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
		// Both kernels are the naive one-thread-per-output mapping, but with the
		// output held in a *register accumulator* instead of read-modify-written
		// in global memory every inner iteration:
		//   * The baseline (polybench/atax.cpp) does `tmp[item] += A*x` / `y[item]
		//     += A*tmp` inside the reduction loop. Under the DSP build's
		//     POCL_EXTRA_BUILD_FLAGS="-cl-opt-disable" the compiler does NOT hoist
		//     tmp[item]/y[item] into a register, so the output is touched N times
		//     per element -- the main inefficiency on the MT3K DSP. Accumulating
		//     in a scalar register and storing once fixes that (the bulk of the
		//     win), exactly analogous to the 1x1 register tile in 2mm_opt/3mm_opt.
		//   * No access::target::local, no barriers, no nd_range. The previous
		//     atax_opt raised occupancy via a per-output work-group reduction in
		//     local memory (tree reduction + barriers); on the MT3K DSP that
		//     hangs (1D local accessors + barriers are not supported by the MOCL
		//     backend), so it is dropped. On CPU/GPU -- where the device compiler
		//     already hoists the `+=` into a register -- the register-accumulator
		//     rewrite is a no-op, so this version is never a regression there.
		//   * Atax1 (tmp = A*x): reduction over j, the contiguous axis of A ->
		//     A[{i,j}] reads coalesced across consecutive j. Output written from
		//     zero (discard_write).
		//   * Atax2 (y = A^T*tmp): reduction over i, the strided axis of A. The
		//     per-column mapping keeps A reads coalesced *across* work-items
		//     (adjacent columns j are adjacent in memory); a per-column work-group
		//     would make the A reads strided (one column = stride N), so the
		//     one-thread-per-column mapping is kept. y is seeded from the existing
		//     y (read_write, host-initialized to 0) to match atax_cpu exactly.

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto x = x_buffer.get_access<access::mode::read>(cgh);
			auto tmp = tmp_buffer.get_access<access::mode::discard_write>(cgh);

			cgh.parallel_for<Atax1>(tmp_buffer.get_range(), [=, N_ = size](item<1> item) {
				const auto i = item[0];

				DATA_TYPE acc = DATA_TYPE(0);
				for(size_t j = 0; j < N_; j++) {
					acc += A[{i, j}] * x[j];
				}
				tmp[item] = acc;
			});
		}));

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto tmp = tmp_buffer.get_access<access::mode::read>(cgh);
			auto y = y_buffer.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<Atax2>(y_buffer.get_range(), [=, N_ = size](item<1> item) {
				const auto j = item[0];

				DATA_TYPE acc = y[item];
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
