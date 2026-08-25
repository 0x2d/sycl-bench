#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Polybench_Syr2k1;

// Unique kernel-name type per (BM, BN). The LLVM-MLIR backend requires a distinct
// type for every parallel_for instantiation; we instantiate the register-tiled
// kernel once per selectable tile shape.
template <size_t BM, size_t BN>
class Polybench_Syr2k_Tile;

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

// Register-tiled SYR2K: C := alpha*(A*B' + B*A') + beta*C, A/B in row-major (N x M,
// here square M = N = size). Each work-item computes a BM x BN block of outputs in
// register accumulators. For every k it loads BM rows of the i-block of A and B and
// BN rows of the j-block of A and B once, then performs BM*BN*2 FMAs (the A*B' term
// and the B*A' term), reusing each loaded operand across the whole block.
template <size_t BM, size_t BN, typename AccA, typename AccB, typename AccC>
static void submitRegSyr2kImpl(cl::sycl::handler& cgh, AccA A, AccB B, AccC C, size_t N) {
	using namespace cl::sycl;
	using KernelName = Polybench_Syr2k_Tile<BM, BN>;

	const size_t gi = (N + BM - 1) / BM;
	const size_t gj = (N + BN - 1) / BN;

	cgh.parallel_for<KernelName>(range<2>{gi, gj}, [=, N_ = N](item<2> item) {
		const size_t bi = item[0] * BM; // first row of this thread's C-block
		const size_t bj = item[1] * BN; // first col of this thread's C-block

		auto clampR = [N_](size_t r) { return r < N_ ? r : N_ - 1; };
		auto clampC = [N_](size_t c) { return c < N_ ? c : N_ - 1; };

		// Seed from C*beta (read_write: C read once per output for the beta term).
		DATA_TYPE regC[BM][BN];
		for(size_t a = 0; a < BM; a++) {
			const size_t ia = clampR(bi + a);
			for(size_t b = 0; b < BN; b++) {
				const size_t jb = clampC(bj + b);
				regC[a][b] = C[{ia, jb}] * BETA;
			}
		}

		// Inner reduction: load each needed A/B row once per k, reuse across the
		// whole BN (or BM) width via registers -> FMA chain.
		//   Term1: A[i,k]*B[j,k]  (avA[a] * bvB[b])
		//   Term2: B[i,k]*A[j,k]  (avB[a] * bvA[b])
		for(size_t k = 0; k < N_; k++) {
			DATA_TYPE avA[BM];
			DATA_TYPE avB[BM];
			DATA_TYPE bvA[BN];
			DATA_TYPE bvB[BN];
			for(size_t a = 0; a < BM; a++) {
				const size_t ia = clampR(bi + a);
				avA[a] = A[{ia, k}];
				avB[a] = B[{ia, k}];
			}
			for(size_t b = 0; b < BN; b++) {
				const size_t jb = clampC(bj + b);
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
}

// Register-tiled SYR2K dispatcher. Same rationale / DSP tuning story as 2mm_opt /
// gemm_opt: the naive kernel does a global read-modify-write of C every K-step
// (twice -- once per term); under -cl-opt-disable the compiler does not hoist
// C[item] into a register, so C is touched 2*N times per output. The 1x1 register
// tile reads C once (for the beta term), accumulates in a scalar register, and
// stores once -- the bulk of the win on the MT3K DSP. Larger tiles (2x2, 4x4) reuse
// each A/B load across more outputs and can win on CPU/GPU where the device
// compiler keeps the arrays in registers; on the DSP they are slower (stack arrays
// under -cl-opt-disable) and tiles >= 8x8 overflow the tiny work-item stack. The
// default is therefore 1x1; set SYCL_SYR2K_BM / SYCL_SYR2K_BN to 2 or 4 to enlarge
// the tile on CPU/GPU.
//
//   * No access::target::local, no barriers, no nd_range: the orise twin's 8x8
//     register block is structurally DSP-safe (no shared memory) but the 8x8 stack
//     overflows the DSP's work-item stack -> silent verification FAIL; this version
//     caps the tile at 4x4 and defaults to 1x1.
//   * Edge handling: the launch is padded up to a multiple of BM/BN. Out-of-range
//     rows/cols clamp their loads to a valid index (results discarded) and the
//     store is guarded, so non-multiple N stays correct.
template <typename KernelName, typename AccA, typename AccB, typename AccC>
static void submitRegSyr2k(cl::sycl::handler& cgh, AccA A, AccB B, AccC C, size_t N, size_t BM, size_t BN) {
	if(BM == 1 && BN == 1) return submitRegSyr2kImpl<1, 1>(cgh, A, B, C, N);
	if(BM == 2 && BN == 2) return submitRegSyr2kImpl<2, 2>(cgh, A, B, C, N);
	if(BM == 4 && BN == 4) return submitRegSyr2kImpl<4, 4>(cgh, A, B, C, N);
	// Fallback: 1x1.
	return submitRegSyr2kImpl<1, 1>(cgh, A, B, C, N);
}

class Polybench_Syr2k {
  public:
	Polybench_Syr2k(const BenchmarkArgs& args) : args(args), size(args.problem_size) {
		// Register-tile edge (BM output rows x BN output cols per work-item). Tunable
		// via SYCL_SYR2K_BM / SYCL_SYR2K_BN (defaults 1/1). See submitRegSyr2k.
		const auto readEnv = [](const char* name, size_t def) {
			const char* e = std::getenv(name);
			const long v = e ? std::strtol(e, nullptr, 10) : (long)def;
			return v > 0 ? (size_t)v : def;
		};
		bm = readEnv("SYCL_SYR2K_BM", 1);
		bn = readEnv("SYCL_SYR2K_BN", 1);
	}

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

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto B = B_buffer.get_access<access::mode::read>(cgh);
			auto C = C_buffer.get_access<access::mode::read_write>(cgh);

			submitRegSyr2k<Polybench_Syr2k1>(cgh, A, B, C, size, bm, bn);
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
	size_t bm{1};
	size_t bn{1};
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
