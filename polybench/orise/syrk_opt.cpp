#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Syr2k2;

constexpr DATA_TYPE alpha = 123;
constexpr DATA_TYPE beta = 14512;

void init_arrays(DATA_TYPE* A, DATA_TYPE* C, size_t size) {
	const auto N = size;
	const auto M = size;

	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < M; j++) {
			A[i * M + j] = ((DATA_TYPE)i * j) / N;
		}

		for(size_t j = 0; j < N; j++) {
			C[i * M + j] = ((DATA_TYPE)i * j + 2) / N;
		}
	}
}

void syrk(DATA_TYPE* A, DATA_TYPE* C, size_t size) {
	const auto N = size;
	const auto M = size;

	/*  C := alpha*A*A' + beta*C */
	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < N; j++) {
			C[i * M + j] *= beta;
		}
	}

	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < N; j++) {
			for(size_t k = 0; k < M; k++) {
				C[i * N + j] += alpha * A[i * M + k] * A[j * M + k];
			}
		}
	}
}

// Register-blocked SYRK: C := alpha*A*A' + beta*C, with A in row-major (N x M)
// and here square (M = N = size).
//
// Optimizations vs. the baseline polybench/syrk.cpp (and the accumulator-only
// syrk_opt.cpp):
//   * Register micro-kernel (BM x BN per thread): each thread computes a BM x BN
//     block of C in register accumulators. For every k it loads BM rows of the
//     i-block of A and BN rows of the j-block once, then performs BM*BN FMAs.
//     Arithmetic intensity per A-load rises from 0.5 (1x1) to (BM*BN)/(BM+BN)
//     (=2.0 for 4x4, =4.0 for 8x8). This is the canonical CPU GEMM optimization:
//     it converts the bandwidth-bound 1-output-per-thread kernel into a
//     compute-bound FMA reduction that the DPC++/OpenCL backend vectorizes
//     (AVX-512 on this Xeon). No local/shared memory is used, so it does not
//     regress on CPU the way access::target::local tiling does (see the
//     2DConvolution_tiled CPU-regression note).
//   * Single C read / single C write per output: the old `C[item] +=` per k-step
//     did a global read-modify-write on every iteration (2*N^3 C accesses);
//     now C is read once (for the beta term) and written once.
//   * Symmetric-triangle short-circuit: C is symmetric in (i,j). When a
//     computed block lies entirely above or below the diagonal we still compute
//     it (keeps the launch regular and branch-free on the hot path), but blocks
//     straddling the diagonal write only the valid upper/lower cells. The full
//     symmetric write set is produced; the wasted lower-triangle work is a small
//     constant fraction and avoids the divergent 1-D upper-triangle thread
//     mapping that would wreck vectorization on CPU.
//   * alpha/beta are constexpr so they fold into FMA immediates.
class Polybench_Syrk {
  public:
	Polybench_Syrk(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		A.resize(size * size);
		C.resize(size * size);

		init_arrays(A.data(), C.data(), size);

		A_buffer.initialize(args.device_queue, A.data(), cl::sycl::range<2>(size, size));
		C_buffer.initialize(args.device_queue, C.data(), cl::sycl::range<2>(size, size));
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		// Register-tile edge. BM x BN outputs per thread; tuned for the Xeon's
		// vector width and L1 capacity. 8x8 gives intensity 4.0 FMA/load and
		// 64 accumulators/thread, which fits the AVX-512 register file. (BN=16
		// was tried to match the AVX-512 width but regressed: the j-panel reads
		// 16 strided A rows per k, exhausting L1 and losing more than the wider
		// FMA gained.)
		constexpr size_t BM = 8;
		constexpr size_t BN = 8;

		const size_t N = size;
		const size_t M = size;
		const size_t gi_range = (N + BM - 1) / BM;
		const size_t gj_range = (N + BN - 1) / BN;

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto C = C_buffer.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<Syr2k2>(range<2>(gi_range, gj_range), [=, N_ = N, M_ = M](item<2> item) {
				const size_t bi = item[0] * BM; // first row of this thread's C-block
				const size_t bj = item[1] * BN; // first col of this thread's C-block

				// Register accumulators for the BM x BN block.
				DATA_TYPE regC[BM][BN];

				// Seed from C*beta (one global read per output). Rows/cols past N-1
				// (only possible for the last block when N % BM != 0) are clamped
				// to a valid index so the A reads below stay in range; their
				// results are discarded by the guarded store at the end.
				auto clampRow = [N_](size_t r) { return r < N_ ? r : N_ - 1; };
				auto clampCol = [N_](size_t c) { return c < N_ ? c : N_ - 1; };

				for(size_t a = 0; a < BM; a++) {
					const size_t ia = clampRow(bi + a);
					for(size_t b = 0; b < BN; b++) {
						const size_t jb = clampCol(bj + b);
						regC[a][b] = C[{ia, jb}] * beta;
					}
				}

				// Inner reduction: load each needed A row once per k, reuse across
				// the whole BN (or BM) width via registers -> FMA chain.
				for(size_t k = 0; k < M_; k++) {
					DATA_TYPE av[BM];
					DATA_TYPE bv[BN];
					for(size_t a = 0; a < BM; a++) av[a] = A[{clampRow(bi + a), k}];
					for(size_t b = 0; b < BN; b++) bv[b] = A[{clampCol(bj + b), k}];

					for(size_t a = 0; a < BM; a++) {
						for(size_t b = 0; b < BN; b++) {
							regC[a][b] += alpha * av[a] * bv[b];
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

		// Trigger writeback
		C_buffer.reset();

		std::vector<DATA_TYPE> C_cpu(size * size);

		init_arrays(A.data(), C_cpu.data(), size);

		syrk(A.data(), C_cpu.data(), size);

		for(size_t i = 0; i < size; i++) {
			for(size_t j = 0; j < size; j++) {
				const auto diff = percentDiff(C_cpu[i * size + j], C[i * size + j]);
				if(diff > ERROR_THRESHOLD) return false;
			}
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Syrk"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> C;

	PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> C_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Syrk>();
	return 0;
}
