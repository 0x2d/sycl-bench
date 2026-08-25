#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Gesummv1;

constexpr DATA_TYPE ALPHA = 1;
constexpr DATA_TYPE BETA = 1;

void init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x, size_t size) {
	const auto N = size;

	for(size_t i = 0; i < N; i++) {
		x[i] = 1;

		for(size_t j = 0; j < N; j++) {
			A[i * N + j] = 2;
			B[i * N + j] = 3;
		}
	}
}

void gesummv(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp, size_t size) {
	const auto N = size;

	for(size_t i = 0; i < N; i++) {
		tmp[i] = 0;
		y[i] = 0;
		for(size_t j = 0; j < N; j++) {
			tmp[i] = A[i * N + j] * x[j] + tmp[i];
			y[i] = B[i * N + j] * x[j] + y[i];
		}

		y[i] = ALPHA * tmp[i] + BETA * y[i];
	}
}

class Polybench_Gesummv {
public:
	Polybench_Gesummv(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		A.resize(size * size);
		B.resize(size * size);
		x.resize(size);
		y.resize(size);
		tmp.resize(size);

		init(A.data(), B.data(), x.data(), size);

		A_buffer.initialize(args.device_queue, A.data(), cl::sycl::range<2>(size, size));
		B_buffer.initialize(args.device_queue, B.data(), cl::sycl::range<2>(size, size));
		x_buffer.initialize(args.device_queue, x.data(), cl::sycl::range<1>(size));
		y_buffer.initialize(args.device_queue, y.data(), cl::sycl::range<1>(size));
		tmp_buffer.initialize(args.device_queue, tmp.data(), cl::sycl::range<1>(size));
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		// Optimized GESUMMV (MT3K-DSP / tianhe): tmp = A*x, y = B*x, then
		// y = ALPHA*tmp + BETA*y, A/B row-major (N x N).
		//
		// This is the atax_opt / bicg_opt register-accumulator lever (see
		// [[mocl-dsp-constraints]]) applied to the fused GEMV pair. The baseline
		// (polybench/gesummv.cpp) does `tmp[item] += A*x` / `y[item] += B*x` inside the
		// j loop and then `y[item] = ALPHA*tmp[item] + BETA*y[item]`. Under the DSP
		// build's POCL_EXTRA_BUILD_FLAGS="-cl-opt-disable" the compiler does NOT hoist
		// tmp[item]/y[item] into registers, so both outputs are touched N times per row
		// -- the main inefficiency on the MT3K DSP. Accumulating tmp/y in two scalar
		// registers (accA, accB) and storing once fixes that (the bulk of the win).
		//
		//   * No access::target::local, no barriers, no nd_range: the orise twin
		//     raises occupancy via a per-row work-group tree reduction in local memory;
		//     that path hangs the MOCL backend on the DSP (1D local accessors +
		//     barriers are not supported) and is dropped. On CPU/GPU -- where the device
		//     compiler already hoists the `+=` into a register -- the register-
		//     accumulator rewrite is a no-op, so this version is never a regression.
		//   * One thread per row i, reduction over the contiguous column axis j ->
		//     A[{i,j}], B[{i,j}] reads coalesced. tmp/y start at 0 (vector value-init),
		//     so discard_write from zero is correct.

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto B = B_buffer.get_access<access::mode::read>(cgh);
			auto x = x_buffer.get_access<access::mode::read>(cgh);
			auto y = y_buffer.get_access<access::mode::discard_write>(cgh);
			auto tmp = tmp_buffer.get_access<access::mode::discard_write>(cgh);

			cgh.parallel_for<Gesummv1>(y.get_range(), [=, N_ = size](item<1> item) {
				const auto i = item[0];

				DATA_TYPE accA = DATA_TYPE(0); // tmp = A*x
				DATA_TYPE accB = DATA_TYPE(0); // y  = B*x
				for(size_t j = 0; j < N_; j++) {
					accA += A[{i, j}] * x[j];
					accB += B[{i, j}] * x[j];
				}

				tmp[item] = accA;
				y[item] = ALPHA * accA + BETA * accB;
			});
		}));
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		// Trigger writeback
		y_buffer.reset();

		std::vector<DATA_TYPE> y_cpu(size);
		std::vector<DATA_TYPE> tmp_cpu(size);

		gesummv(A.data(), B.data(), x.data(), y_cpu.data(), tmp_cpu.data(), size);

		for(size_t i = 0; i < size; i++) {
			const auto diff = percentDiff(y_cpu[i], y[i]);
			if(diff > ERROR_THRESHOLD) return false;
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Gesummv_Opt"; }

private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> B;
	std::vector<DATA_TYPE> x;
	std::vector<DATA_TYPE> y;
	std::vector<DATA_TYPE> tmp;

	PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> B_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> x_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> y_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> tmp_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Gesummv>();
	return 0;
}
