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

		// Optimized GESUMMV: tmp = A*x, y = B*x, then y = ALPHA*tmp + BETA*y, A/B
		// row-major (N x N). This is the [[atax-opt-occupancy-reduction]] /
		// [[bicg-opt-occupancy-reduction]] lever applied to the fused GEMV pair.
		//
		// The baseline (polybench/gesummv.cpp) launches one thread per row i, each
		// serially looping j over both A and B. With N=16384 that is ~64 work-groups
		// of auto-local-256 -> under one wavefront per CU on this DCU, so the device
		// cannot hide memory latency and the kernel runs far below peak bandwidth.
		// The win is raising occupancy via a per-row work-group reduction.
		//
		//   * One work-group per row i, WG threads stride over j with stride WG.
		//     Consecutive local ids map to consecutive j -> A[{i,j}], B[{i,j}] reads
		//     are contiguous in row-major storage, i.e. coalesced across the
		//     wavefront. The reduction axis (j) is the contiguous axis of A and B,
		//     so per-output grouping keeps reads coalesced (the favorable atax1
		//     case).
		//   * Two accumulators (one for tmp=A*x, one for y=B*x) reduced in parallel
		//     in two local scratch arrays; lid 0 then stores tmp[i] and the final
		//     y[i] = ALPHA*tmp + BETA*y. tmp/y start at 0 (vector value-init), so
		//     discard_write from zero is correct.
		//   * Thread count rises from N to N*WG (16384 -> ~4M), restoring occupancy.
		const size_t N = size;
		const size_t WG = args.local_size > 0 ? args.local_size : 256;
		const size_t local = WG; // tree reduction requires a power-of-two group width

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto B = B_buffer.get_access<access::mode::read>(cgh);
			auto x = x_buffer.get_access<access::mode::read>(cgh);
			auto y = y_buffer.get_access<access::mode::discard_write>(cgh);
			auto tmp = tmp_buffer.get_access<access::mode::discard_write>(cgh);

			// NOTE: two reductions (A*x and B*x) are packed into a SINGLE local
			// accessor -- scratch[0..local) holds the A-sums, scratch[local..2*local)
			// holds the B-sums -- rather than two separate local accessors. On this
			// LLVM-MLIR-HIP backend a kernel with two local accessors mis-compiles
			// (verified: the 2-scratch version FAILED verification while the
			// structurally identical 1-scratch mvt_opt PASSES); one accessor is fine.
			accessor<DATA_TYPE, 1, access::mode::read_write, access::target::local> scratch{2 * local, cgh};

			cgh.parallel_for<Gesummv1>(nd_range<1>{N * local, local}, [=, N_ = N](nd_item<1> item) {
				const size_t lid = item.get_local_id(0);
				const size_t i = item.get_group(0);

				DATA_TYPE accA = DATA_TYPE(0); // tmp = A*x
				DATA_TYPE accB = DATA_TYPE(0); // y  = B*x
				for(size_t jj = lid; jj < N_; jj += local) {
					accA += A[{i, jj}] * x[jj];
					accB += B[{i, jj}] * x[jj];
				}
				scratch[lid] = accA;
				scratch[local + lid] = accB;
				item.barrier(access::fence_space::local_space);

				for(size_t s = local / 2; s > 0; s >>= 1) {
					if(lid < s) {
						scratch[lid] += scratch[lid + s];
						scratch[local + lid] += scratch[local + lid + s];
					}
					item.barrier(access::fence_space::local_space);
				}

				if(lid == 0) {
					tmp[i] = scratch[0];
					y[i] = ALPHA * scratch[0] + BETA * scratch[local];
				}
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
