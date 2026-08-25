#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Polybench_Syrk1;

// Unique kernel-name type per (BM, BN). The LLVM-MLIR backend requires a distinct
// type for every parallel_for instantiation; we instantiate the register-tiled
// kernel once per selectable tile shape.
template <size_t BM, size_t BN>
class Polybench_Syrk_Tile;

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

// Register-tiled SYRK: C := alpha*A*A' + beta*C, A in row-major (N x M, here square
// M = N = size). Each work-item computes a BM x BN block of outputs in register
// accumulators. For every k it loads BM rows of the i-block of A and BN rows of the
// j-block once, then performs BM*BN FMAs, reusing each loaded operand across the
// whole block.
template <size_t BM, size_t BN, typename AccA, typename AccC>
static void submitRegSyrkImpl(cl::sycl::handler& cgh, AccA A, AccC C, size_t N) {
	using namespace cl::sycl;
	using KernelName = Polybench_Syrk_Tile<BM, BN>;

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
				regC[a][b] = C[{ia, jb}] * beta;
			}
		}

		// Inner reduction: load each needed A row once per k, reuse across the whole
		// BN (or BM) width via registers -> FMA chain.
		for(size_t k = 0; k < N_; k++) {
			DATA_TYPE av[BM];
			DATA_TYPE bv[BN];
			for(size_t a = 0; a < BM; a++) av[a] = A[{clampR(bi + a), k}];
			for(size_t b = 0; b < BN; b++) bv[b] = A[{clampC(bj + b), k}];

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
}

// Register-tiled SYRK dispatcher. Same rationale / DSP tuning story as 2mm_opt /
// gemm_opt / syr2k_opt: the naive kernel does a global read-modify-write of C every
// K-step; under -cl-opt-disable the compiler does not hoist C[item] into a register,
// so C is touched N times per output. The 1x1 register tile reads C once (for the
// beta term), accumulates in a scalar register, and stores once -- the bulk of the
// win on the MT3K DSP. Larger tiles (2x2, 4x4) reuse each A load across more
// outputs and can win on CPU/GPU where the device compiler keeps the arrays in
// registers; on the DSP they are slower (stack arrays under -cl-opt-disable) and
// tiles >= 8x8 overflow the tiny work-item stack. The default is therefore 1x1;
// set SYCL_SYRK_BM / SYCL_SYRK_BN to 2 or 4 to enlarge the tile on CPU/GPU.
//
//   * No access::target::local, no barriers, no nd_range: the orise twin's 8x8
//     register block is structurally DSP-safe (no shared memory) but the 8x8 stack
//     overflows the DSP's work-item stack -> silent verification FAIL; this version
//     caps the tile at 4x4 and defaults to 1x1.
//   * Edge handling: the launch is padded up to a multiple of BM/BN. Out-of-range
//     rows/cols clamp their loads to a valid index (results discarded) and the
//     store is guarded, so non-multiple N stays correct.
template <typename KernelName, typename AccA, typename AccC>
static void submitRegSyrk(cl::sycl::handler& cgh, AccA A, AccC C, size_t N, size_t BM, size_t BN) {
	if(BM == 1 && BN == 1) return submitRegSyrkImpl<1, 1>(cgh, A, C, N);
	if(BM == 2 && BN == 2) return submitRegSyrkImpl<2, 2>(cgh, A, C, N);
	if(BM == 4 && BN == 4) return submitRegSyrkImpl<4, 4>(cgh, A, C, N);
	// Fallback: 1x1.
	return submitRegSyrkImpl<1, 1>(cgh, A, C, N);
}

class Polybench_Syrk {
  public:
	Polybench_Syrk(const BenchmarkArgs& args) : args(args), size(args.problem_size) {
		// Register-tile edge (BM output rows x BN output cols per work-item). Tunable
		// via SYCL_SYRK_BM / SYCL_SYRK_BN (defaults 1/1). See submitRegSyrk.
		const auto readEnv = [](const char* name, size_t def) {
			const char* e = std::getenv(name);
			const long v = e ? std::strtol(e, nullptr, 10) : (long)def;
			return v > 0 ? (size_t)v : def;
		};
		bm = readEnv("SYCL_SYRK_BM", 1);
		bn = readEnv("SYCL_SYRK_BN", 1);
	}

	void setup() {
		A.resize(size * size);
		C.resize(size * size);

		init_arrays(A.data(), C.data(), size);

		A_buffer.initialize(args.device_queue, A.data(), cl::sycl::range<2>(size, size));
		C_buffer.initialize(args.device_queue, C.data(), cl::sycl::range<2>(size, size));
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto C = C_buffer.get_access<access::mode::read_write>(cgh);

			submitRegSyrk<Polybench_Syrk1>(cgh, A, C, size, bm, bn);
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
	size_t bm{1};
	size_t bn{1};
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
