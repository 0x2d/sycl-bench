#include <string>
#include <vector>

#include <cmath>
#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Gramschmidt1;
class Gramschmidt2;
class Gramschmidt3;

void init_array(DATA_TYPE* A, size_t size) {
	const auto M = size;
	const auto N = size;

	for(size_t i = 0; i < M; i++) {
		for(size_t j = 0; j < N; j++) {
			A[i * N + j] = ((DATA_TYPE)(i + 1) * (j + 1)) / (M + 1);
		}
	}
}

void gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q, size_t size) {
	const auto M = size;
	const auto N = size;

	for(size_t k = 0; k < N; k++) {
		DATA_TYPE nrm = 0;
		for(size_t i = 0; i < M; i++) {
			nrm += A[i * N + k] * A[i * N + k];
		}

		R[k * N + k] = sqrt(nrm);
		for(size_t i = 0; i < M; i++) {
			Q[i * N + k] = A[i * N + k] / R[k * N + k];
		}

		for(size_t j = k + 1; j < N; j++) {
			R[k * N + j] = 0;
			for(size_t i = 0; i < M; i++) {
				R[k * N + j] += Q[i * N + k] * A[i * N + j];
			}
			for(size_t i = 0; i < M; i++) {
				A[i * N + j] = A[i * N + j] - Q[i * N + k] * R[k * N + j];
			}
		}
	}
}

class Polybench_Gramschmidt {
  public:
	Polybench_Gramschmidt(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		A.resize(size * size);
		R.resize(size * size);
		Q.resize(size * size);

		init_array(A.data(), size);

		A_buffer.initialize(args.device_queue, A.data(), cl::sycl::range<2>(size, size));
		R_buffer.initialize(args.device_queue, R.data(), cl::sycl::range<2>(size, size));
		Q_buffer.initialize(args.device_queue, Q.data(), cl::sycl::range<2>(size, size));
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		// Optimized Gram-Schmidt (MT3K-DSP / tianhe). k stays a host loop (each k
		// depends on the prior modified-A columns, so it cannot be parallelized).
		// The three baseline kernels are kept as separate launches -- the orise
		// twin fuses Gram1+Gram2 via a work-group tree reduction in local memory,
		// but that path hangs the MOCL backend on the DSP (1D local accessors +
		// barriers unsupported; see [[mocl-dsp-constraints]]), so no fusion here.
		//
		//   * Gram1 (norm): already a scalar-register accumulator (nrm is a thread-
		//     local scalar, summed then stored once). Single-thread serial reduction
		//     (parallel_for<range<2>(1,1)>); O(N) per k, not the bottleneck, and a
		//     work-group reduction would need local memory + barriers (unavailable).
		//     Left at baseline.
		//   * Gram2 (normalize): elementwise Q[i,k] = A[i,k]/R[k,k], single write.
		//     Left at baseline.
		//   * Gram3 (rank-1 update, the dominant O(N^2)-per-k kernel): baseline does
		//     `R[item] = 0; for i: R[item] += Q*A;` (N global RMWs of R[item]) `then
		//     for i: A[{i,j}] -= Q*R[item]` (reads R[item] back N times). Under
		//     -cl-opt-disable the compiler does not hoist R[item] into a register, so
		//     it is reloaded on every i in both loops. The rewrite accumulates the
		//     dot product in a scalar register (accR), stores once, and reuses accR
		//     (not a global reload) in the A update -- collapsing the N-fold R RMW
		//     and the N-fold R reload. The A update itself stays a per-i RMW (each
		//     A[{i,j}] is a distinct address, so there is no CSE/register-reuse win
		//     across i -- the read-modify-write is unavoidable).
		for(size_t k = 0; k < size; k++) {
			// Gram1: column norm -> R[k,k].
			events.push_back(args.device_queue.submit([&](handler& cgh) {
				auto A = A_buffer.get_access<access::mode::read>(cgh);
				auto R = R_buffer.get_access<access::mode::write>(cgh);

				cgh.parallel_for<Gramschmidt1>(range<2>(1, 1), [=, M_ = size, k_ = k](item<2> item) {
					DATA_TYPE nrm = 0;
					for(size_t i = 0; i < M_; i++) {
						nrm += A[{i, k_}] * A[{i, k_}];
					}
					R[{k_, k_}] = cl::sycl::sqrt(nrm);
				});
			}));

			// Gram2: normalize -> Q[i,k] = A[i,k] / R[k,k].
			events.push_back(args.device_queue.submit([&](handler& cgh) {
				auto A = A_buffer.get_access<access::mode::read>(cgh);
				auto R = R_buffer.get_access<access::mode::read>(cgh);
				auto Q = Q_buffer.get_access<access::mode::write>(cgh);

				cgh.parallel_for<Gramschmidt2>(range<2>(size, 1), id<2>(0, k), [=, k_ = k](item<2> item) {
					Q[item] = A[item] / R[{k_, k_}];
				});
			}));

			// Gram3: rank-1 update of R[k,j] and A[i,j] for j > k. accR holds the
			// dot product Q[:,k]·A[:,j] in a register; R is stored once, and accR is
			// reused (not reloaded) for the A update.
			events.push_back(args.device_queue.submit([&](handler& cgh) {
				auto A = A_buffer.get_access<access::mode::read_write>(cgh);
				auto R = R_buffer.get_access<access::mode::write>(cgh);
				auto Q = Q_buffer.get_access<access::mode::read>(cgh);

				cgh.parallel_for<Gramschmidt3>(range<2>(size, 1), [=, M_ = size, N_ = size, k_ = k](item<2> item) {
					const auto j = item[0];

					if(j <= k_ || j >= N_) return;

					DATA_TYPE accR = DATA_TYPE(0);
					for(size_t i = 0; i < M_; i++) {
						accR += Q[{i, k_}] * A[{i, j}];
					}
					R[item] = accR;

					for(size_t i = 0; i < M_; i++) {
						A[{i, j}] -= Q[{i, k_}] * accR;
					}
				});
			}));
		}
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		std::vector<DATA_TYPE> A_cpu(size * size);
		std::vector<DATA_TYPE> R_cpu(size * size);
		std::vector<DATA_TYPE> Q_cpu(size * size);

		// Trigger writeback
		A_buffer.reset();

		init_array(A_cpu.data(), size);

		gramschmidt(A_cpu.data(), R_cpu.data(), Q_cpu.data(), size);

		for(size_t i = 0; i < size; i++) {
			for(size_t j = 0; j < size; j++) {
				const auto diff = percentDiff(A_cpu[i * size + j], A[i * size + j]);
				if(diff > ERROR_THRESHOLD) return false;
			}
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Gramschmidt_Opt"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> R;
	std::vector<DATA_TYPE> Q;

	PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> R_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> Q_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Gramschmidt>();
	return 0;
}
