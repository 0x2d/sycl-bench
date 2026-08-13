#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Mvt1;
class Mvt2;

void init_arrays(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y_1, DATA_TYPE* y_2, size_t size) {
	const auto N = size;

	for(size_t i = 0; i < N; i++) {
		x1[i] = 0.0;
		x2[i] = 0.0;
		y_1[i] = 1.0;
		y_2[i] = 1.0;

		for(size_t j = 0; j < N; j++) {
			a[i * N + j] = (DATA_TYPE)(i + j + 1.0) / N;
		}
	}
}

void runMvt(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2, size_t size) {
	const auto N = size;

	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < N; j++) {
			x1[i] = x1[i] + a[i * N + j] * y1[j];
		}
	}

	for(size_t k = 0; k < N; k++) {
		for(size_t l = 0; l < N; l++) {
			x2[k] = x2[k] + a[k * N + l] * y2[l];
		}
	}
}

class Polybench_Mvt {
  public:
	Polybench_Mvt(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		a.resize(size * size);
		x1.resize(size);
		x2.resize(size);
		y1.resize(size);
		y2.resize(size);

		init_arrays(a.data(), x1.data(), x2.data(), y1.data(), y2.data(), size);

		a_buffer .initialize(args.device_queue, a.data(), cl::sycl::range<2>(size, size));
		x1_buffer.initialize(args.device_queue, x1.data(), cl::sycl::range<1>(size));
		x2_buffer.initialize(args.device_queue, x2.data(), cl::sycl::range<1>(size));
		y1_buffer.initialize(args.device_queue, y1.data(), cl::sycl::range<1>(size));
		y2_buffer.initialize(args.device_queue, y2.data(), cl::sycl::range<1>(size));
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		// Optimized MVT: x1 = A*y1, x2 = A*y2, A row-major (N x N).
		//
		// This is the [[atax-opt-occupancy-reduction]] / [[bicg-opt-occupancy-reduction]]
		// lever applied to BOTH kernels. Unlike ATAX (where only the A*x kernel had the
		// favorable orientation and A^T*tmp was left at baseline), both Mvt1 and Mvt2 are
		// A*(vector) with one thread per row i/k looping over the contiguous column axis
		// (j/l). So both have the coalesced-but-low-occupancy shape and both get the
		// per-row work-group reduction.
		//
		//   * One work-group per output row, WG threads stride over the column axis.
		//     Consecutive local ids -> consecutive j/l -> A[{row,col}] reads are
		//     contiguous in row-major storage -> coalesced. Thread count N -> N*WG
		//     (16384 -> ~4M) restores occupancy.
		//   * x1/x2 start at 0 (vector value-init) -> discard_write from zero.
		//   * Tree reduction requires a power-of-two group width (--local=256 satisfies it).
		const size_t N = size;
		const size_t WG = args.local_size > 0 ? args.local_size : 256;
		const size_t local = WG;

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto a = a_buffer.get_access<access::mode::read>(cgh);
			auto y1 = y1_buffer.get_access<access::mode::read>(cgh);
			auto x1 = x1_buffer.get_access<access::mode::discard_write>(cgh);
			accessor<DATA_TYPE, 1, access::mode::read_write, access::target::local> scratch{local, cgh};

			cgh.parallel_for<Mvt1>(nd_range<1>{N * local, local}, [=, N_ = N](nd_item<1> item) {
				const size_t lid = item.get_local_id(0);
				const size_t i = item.get_group(0);

				DATA_TYPE acc = DATA_TYPE(0);
				for(size_t jj = lid; jj < N_; jj += local) {
					acc += a[{i, jj}] * y1[jj];
				}
				scratch[lid] = acc;
				item.barrier(access::fence_space::local_space);

				for(size_t s = local / 2; s > 0; s >>= 1) {
					if(lid < s) scratch[lid] += scratch[lid + s];
					item.barrier(access::fence_space::local_space);
				}

				if(lid == 0) x1[i] = scratch[0];
			});
		}));

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto a = a_buffer.get_access<access::mode::read>(cgh);
			auto y2 = y2_buffer.get_access<access::mode::read>(cgh);
			auto x2 = x2_buffer.get_access<access::mode::discard_write>(cgh);
			accessor<DATA_TYPE, 1, access::mode::read_write, access::target::local> scratch{local, cgh};

			cgh.parallel_for<Mvt2>(nd_range<1>{N * local, local}, [=, N_ = N](nd_item<1> item) {
				const size_t lid = item.get_local_id(0);
				const size_t k = item.get_group(0);

				DATA_TYPE acc = DATA_TYPE(0);
				for(size_t ll = lid; ll < N_; ll += local) {
					acc += a[{k, ll}] * y2[ll];
				}
				scratch[lid] = acc;
				item.barrier(access::fence_space::local_space);

				for(size_t s = local / 2; s > 0; s >>= 1) {
					if(lid < s) scratch[lid] += scratch[lid + s];
					item.barrier(access::fence_space::local_space);
				}

				if(lid == 0) x2[k] = scratch[0];
			});
		}));
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		std::vector<DATA_TYPE> x1_cpu(size);
		std::vector<DATA_TYPE> x2_cpu(size);

		// Trigger writeback
		x1_buffer.reset();
		x2_buffer.reset();

		init_arrays(a.data(), x1_cpu.data(), x2_cpu.data(), y1.data(), y2.data(), size);

		runMvt(a.data(), x1_cpu.data(), x2_cpu.data(), y1.data(), y2.data(), size);

		for(size_t i = 0; i < size; i++) {
			auto diff = percentDiff(x1_cpu[i], x1[i]);
			if(diff > ERROR_THRESHOLD) return false;

			diff = percentDiff(x2_cpu[i], x2[i]);
			if(diff > ERROR_THRESHOLD) return false;
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Mvt_Opt"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> a;
	std::vector<DATA_TYPE> x1;
	std::vector<DATA_TYPE> x2;
	std::vector<DATA_TYPE> y1;
	std::vector<DATA_TYPE> y2;

	PrefetchedBuffer<DATA_TYPE, 2> a_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> x1_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> x2_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> y1_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> y2_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Mvt>();
	return 0;
}
