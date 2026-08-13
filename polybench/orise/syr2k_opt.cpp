#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Syr2k2;

constexpr DATA_TYPE ALPHA = 1;
constexpr DATA_TYPE BETA = 1;

void init_arrays(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, size_t size) {
	const auto N = size;
	const auto M = size;

	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < N; j++) {
			C[i * N + j] = ((DATA_TYPE)i * j + 2) / N;
		}

		for(size_t j = 0; j < M; j++) {
			A[i * N + j] = ((DATA_TYPE)i * j) / N;
			B[i * N + j] = ((DATA_TYPE)i * j + 1) / N;
		}
	}
}

void syr2k(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, size_t size) {
	const auto N = size;
	const auto M = size;

	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < N; j++) {
			C[i * N + j] *= BETA;
		}
	}

	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < N; j++) {
			for(size_t k = 0; k < M; k++) {
				C[i * N + j] += ALPHA * A[i * M + k] * B[j * M + k];
				C[i * N + j] += ALPHA * B[i * M + k] * A[j * M + k];
			}
		}
	}
}

// Register-blocked SYR2K: C := alpha*(A*B' + B*A') + beta*C, A/B in row-major
// (N x M, here square M = N = size). This mirrors the shipped orise syrk_opt
// register micro-kernel (the established SYRK-family pattern) extended to the
// two-term A*B' + B*A' update.
//
//   * Register micro-kernel (BM x BN per thread): each thread computes a BM x BN
//     block of C in register accumulators. For every k it loads BM rows of the
//     i-block of A and B and BN rows of the j-block of A and B once, then
//     performs BM*BN*2 FMAs (the A*B' term and the B*A' term). Arithmetic
//     intensity per load rises from ~0.5 (1x1) to ~2*(BM*BN)/(2*(BM+BN)) for the
//     two terms, converting the bandwidth-bound 1-output-per-thread kernel into
//     a compute-bound FMA reduction.
//   * Single C read / single C write per output: the old `C[item] +=` per k-step
//     did a global read-modify-write on every iteration; now C is read once (for
//     the beta term) and written once.
//   * Edge clamping keeps non-multiple N correct (no-op for default N=1024).
//   * alpha/beta are constexpr so they fold into FMA immediates.
class Polybench_Syr2k {
  public:
	Polybench_Syr2k(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		A.resize(size * size);
		B.resize(size * size);
		C.resize(size * size);

		init_arrays(A.data(), B.data(), C.data(), size);

		A_buffer.initialize(args.device_queue, A.data(), cl::sycl::range<2>(size, size));
		B_buffer.initialize(args.device_queue, B.data(), cl::sycl::range<2>(size, size));
		C_buffer.initialize(args.device_queue, C.data(), cl::sycl::range<2>(size, size));
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		// Register-tile edge. BM x BN outputs per thread; matches the syrk_opt
		// tuning (8x8 = intensity ~4/load, 64 accumulators/thread).
		constexpr size_t BM = 8;
		constexpr size_t BN = 8;

		const size_t N = size;
		const size_t M = size;
		const size_t gi_range = (N + BM - 1) / BM;
		const size_t gj_range = (N + BN - 1) / BN;

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto B = B_buffer.get_access<access::mode::read>(cgh);
			auto C = C_buffer.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<Syr2k2>(range<2>(gi_range, gj_range), [=, N_ = N, M_ = M](item<2> item) {
				const size_t bi = item[0] * BM; // first row of this thread's C-block
				const size_t bj = item[1] * BN; // first col of this thread's C-block

				DATA_TYPE regC[BM][BN];

				// Seed from C*beta (one global read per output). Rows/cols past N-1
				// (only possible for the last block when N % BM != 0) are clamped
				// to a valid index so the A/B reads below stay in range; their
				// results are discarded by the guarded store at the end.
				auto clampRow = [N_](size_t r) { return r < N_ ? r : N_ - 1; };
				auto clampCol = [N_](size_t c) { return c < N_ ? c : N_ - 1; };

				for(size_t a = 0; a < BM; a++) {
					const size_t ia = clampRow(bi + a);
					for(size_t b = 0; b < BN; b++) {
						const size_t jb = clampCol(bj + b);
						regC[a][b] = C[{ia, jb}] * BETA;
					}
				}

				// Inner reduction: load each needed A/B row once per k, reuse
				// across the whole BN (or BM) width via registers -> FMA chain.
				// Term1: A[i,k]*B[j,k]  (avA[a] * bvB[b])
				// Term2: B[i,k]*A[j,k]  (avB[a] * bvA[b])
				for(size_t k = 0; k < M_; k++) {
					DATA_TYPE avA[BM];
					DATA_TYPE avB[BM];
					DATA_TYPE bvA[BN];
					DATA_TYPE bvB[BN];
					for(size_t a = 0; a < BM; a++) {
						const size_t ia = clampRow(bi + a);
						avA[a] = A[{ia, k}];
						avB[a] = B[{ia, k}];
					}
					for(size_t b = 0; b < BN; b++) {
						const size_t jb = clampCol(bj + b);
						bvA[b] = A[{jb, k}];
						bvB[b] = B[{jb, k}];
					}

					for(size_t a = 0; a < BM; a++) {
						for(size_t b = 0; b < BN; b++) {
							regC[a][b] += ALPHA * (avA[a] * bvB[b] + avB[a] * bvA[b]);
						}
					}
				}

				// Single guarded write per output cell.
				for(size_t a = 0; a < BM; a++) {
					const size_t ia = bi + a;
					if(ia >= N_) break;
					for(size_t b = 0; b < BN; b++) {
						const size_t jb = bj + b;
						if(jb >= N_) break;
						C[{ia, jb}] = regC[a][b];
					}
				}
			});
		}));
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		std::vector<DATA_TYPE> C_cpu(size * size);

		init_arrays(A.data(), B.data(), C_cpu.data(), size);

		// Trigger writeback
		C_buffer.reset();

		syr2k(A.data(), B.data(), C_cpu.data(), size);

		for(size_t i = 0; i < size; i++) {
			for(size_t j = 0; j < size; j++) {
				const auto diff = percentDiff(C_cpu[i * size + j], C[i * size + j]);
				if(diff > ERROR_THRESHOLD) return false;
			}
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Syr2k_Opt"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> B;
	std::vector<DATA_TYPE> C;

	PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> B_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> C_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Syr2k>();
	return 0;
}
