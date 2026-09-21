#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

#ifndef M_PI
#define M_PI 3.14159
#endif

using DATA_TYPE = float;

class Atax1;
class Atax2;

void init_array(DATA_TYPE* x, DATA_TYPE* A, size_t size) {
	const auto NX = size;
	const auto NY = size;

	for(size_t i = 0; i < NX; i++) {
		x[i] = i * M_PI;
		for(size_t j = 0; j < NY; j++) {
			A[i * NY + j] = ((DATA_TYPE)i * (j)) / NX;
		}
	}
}

void atax_cpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp, size_t size) {
	const auto NX = size;
	const auto NY = size;

	for(size_t i = 0; i < NX; i++) {
		for(size_t j = 0; j < NY; j++) {
			tmp[i] += A[i * NY + j] * x[j];
		}

		for(size_t j = 0; j < NY; j++) {
			y[j] += A[i * NY + j] * tmp[i];
		}
	}
}

class Polybench_Atax {
  public:
	Polybench_Atax(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		A.resize(size * size);
		x.resize(size);
		y.resize(size);
		tmp.resize(size);

		init_array(x.data(), A.data(), size);

		A_buffer.initialize(args.device_queue, A.data(), cl::sycl::range<2>{size, size});
		x_buffer.initialize(args.device_queue, x.data(), cl::sycl::range<1>{size});
		y_buffer.initialize(args.device_queue, y.data(), cl::sycl::range<1>{size});
		tmp_buffer.initialize(args.device_queue, tmp.data(), cl::sycl::range<1>{size});
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		// Optimized ATAX: y = A^T * (A * x), A in row-major (N x N).
		//
		// The baseline (polybench/atax.cpp) launches each kernel with
		// parallel_for(range<1>(N)) and an auto-chosen local size (~256). With
		// N=4096 that is only ~16 work-groups — under one wavefront per CU on
		// this DCU — so the device cannot hide memory latency and ATAX runs far
		// below peak bandwidth. The win below is raising occupancy via a
		// per-output work-group reduction, NOT the RMW collapse (the compiler
		// already hoists `tmp[item] +=` / `y[item] +=` into a register, so a
		// plain register-accumulator rewrite measured no gain on its own).
		//
		//   * Atax1 (tmp = A*x): one work-group per row i, WG threads cooperate
		//     over j with a stride of WG. Consecutive local ids map to
		//     consecutive j -> A[{i,j}] reads are contiguous in row-major
		//     storage, i.e. coalesced across the wavefront. Partial sums are
		//     reduced in local memory (tree reduction). Thread count rises from
		//     N to N*WG (4096 -> ~1M), restoring occupancy. This is the kernel
		//     where the win lives: its reduction axis (j) is the contiguous axis
		//     of A, so per-output grouping keeps reads coalesced.
		//   * Atax2 (y = A^T*tmp): the reduction axis is i (rows), which is the
		//     *strided* axis of row-major A. A per-column work-group would make
		//     the A reads strided (one column = stride N), wrecking coalescing —
		//     measured worse. So Atax2 keeps the baseline one-thread-per-column
		//     mapping (reads coalesced *across* work-items, since adjacent
		//     columns j are adjacent in memory) and only collapses the
		//     `y[item] +=` RMW into a register accumulator + single store.
		//     Raising Atax2 occupancy would need shared-memory tiling of A
		//     (load row-tiles coalesced, read them transposed); left for a
		//     future pass since the simple per-column mapping is already
		//     bandwidth-coalesced.
		//   * No access::target::local tiling on Atax1's A (only the tiny
		//     reduction scratch), per the syrk_opt note that shared-memory
		//     tiling of the big matrix regresses on the CPU target.
		const size_t N = size;
		const size_t WG = args.local_size > 0 ? args.local_size : 128;
		// Tree reduction requires a power-of-two group width.
		const size_t local = WG;

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto x = x_buffer.get_access<access::mode::read>(cgh);
			auto tmp = tmp_buffer.get_access<access::mode::discard_write>(cgh);
			accessor<DATA_TYPE, 1, access::mode::read_write, access::target::local> scratch{local, cgh};

			cgh.parallel_for<Atax1>(nd_range<1>{N * local, local}, [=, N_ = N](nd_item<1> item) {
				const size_t lid = item.get_local_id(0);
				const size_t i = item.get_group(0);

				DATA_TYPE acc = DATA_TYPE(0);
				for(size_t jj = lid; jj < N_; jj += local) {
					acc += A[{i, jj}] * x[jj];
				}
				scratch[lid] = acc;
				item.barrier(access::fence_space::local_space);

				for(size_t s = local / 2; s > 0; s >>= 1) {
					if(lid < s) scratch[lid] += scratch[lid + s];
					item.barrier(access::fence_space::local_space);
				}

				if(lid == 0) tmp[i] = scratch[0];
			});
		}));

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto tmp = tmp_buffer.get_access<access::mode::read>(cgh);
			auto y = y_buffer.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<Atax2>(y_buffer.get_range(), [=, N_ = N](item<1> item) {
				const auto j = item[0];

				DATA_TYPE acc = DATA_TYPE(0);
				for(size_t i = 0; i < N_; i++) {
					acc += A[{i, j}] * tmp[i];
				}
				y[item] = acc;
			});
		}));
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		init_array(x.data(), A.data(), size);

		std::vector<DATA_TYPE> y_cpu(size);
		std::vector<DATA_TYPE> tmp_cpu(size);

		atax_cpu(A.data(), x.data(), y_cpu.data(), tmp_cpu.data(), size);

		auto y_acc = y_buffer.get_access<cl::sycl::access::mode::read>();

		for(size_t i = 0; i < size; i++) {
			const auto diff = percentDiff(y_cpu[i], y_acc[i]);
			if(diff > ERROR_THRESHOLD) return false;
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Atax_Opt"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> x;
	std::vector<DATA_TYPE> y;
	std::vector<DATA_TYPE> tmp;

	PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> x_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> y_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> tmp_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Atax>();
	return 0;
}
