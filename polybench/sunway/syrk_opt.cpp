#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

// Unique kernel-name type per (BM, BN). The LLVM-MLIR backend requires a
// distinct type for every parallel_for instantiation; the register-tiled
// kernel is instantiated once per selectable tile shape. Template parameters
// are compile-time (never captured into the kernel lambda), which also
// sidesteps the swsycl closure-size bug noted below.
template <size_t BM, size_t BN>
class Polybench_Syrk_Tile;

// Register-tile edges selectable at run time (compile machine != run machine,
// so the sweep must not require a rebuild). Returns def when unset/invalid.
static size_t tileEdgeFromEnv(const char* env, size_t def) {
	const char* e = std::getenv(env);
	const long v = e ? std::strtol(e, nullptr, 10) : 0;
	return (v == 1 || v == 2 || v == 4 || v == 8) ? (size_t)v : def;
}

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
//     (=2.0 for 4x4, =4.0 for 8x8): it converts the bandwidth-bound
//     1-output-per-thread kernel into a compute-bound FMA reduction. No
//     local/shared memory and no barriers are used, so nothing here depends on
//     how the athread backend maps local accessors onto the CPE LDM.
//   * Single C read / single C write per output: the old `C[item] +=` per k-step
//     did a global read-modify-write on every iteration (2*N^3 C accesses);
//     now C is read once (for the beta term) and written once.
//   * Regular launch over the full N x N output: every block (including the
//     mirrored lower-triangle blocks) is computed and written, keeping the
//     launch branch-free on the hot path. The duplicated lower-triangle work
//     is a small constant fraction and avoids the divergent 1-D
//     upper-triangle thread mapping that would wreck vectorization.
//   * alpha/beta are constexpr so they fold into FMA immediates.
//   * BM/BN are selected at run time via SYCL_SYRK_BM / SYCL_SYRK_BN (1/2/4/8;
//     default 8x8 = intensity 4.0 FMA/load with 64 accumulators per thread --
//     drop to 1x1 or 2x2 if the device compiler spills the accumulator block).
template <size_t BM, size_t BN, typename AccA, typename AccC>
static void submitRegSyrkImpl(cl::sycl::handler& cgh, AccA A, AccC C, size_t N) {
	using namespace cl::sycl;
	using KernelName = Polybench_Syrk_Tile<BM, BN>;

	const size_t gi_range = (N + BM - 1) / BM;
	const size_t gj_range = (N + BN - 1) / BN;

	cgh.parallel_for<KernelName>(range<2>(gi_range, gj_range), [=, N_ = N, M_ = N](item<2> item) {
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
}

// Register-tile dispatcher: instantiates the kernel once per selectable shape.
template <typename AccA, typename AccC>
static void submitRegSyrk(cl::sycl::handler& cgh, AccA A, AccC C, size_t N, size_t BM, size_t BN) {
	if(BM == 1 && BN == 1) return submitRegSyrkImpl<1, 1>(cgh, A, C, N);
	if(BM == 2 && BN == 2) return submitRegSyrkImpl<2, 2>(cgh, A, C, N);
	if(BM == 4 && BN == 4) return submitRegSyrkImpl<4, 4>(cgh, A, C, N);
	// Default / fallback: 8x8.
	return submitRegSyrkImpl<8, 8>(cgh, A, C, N);
}

class Polybench_Syrk {
  public:
	Polybench_Syrk(const BenchmarkArgs& args)
	: args(args), size(args.problem_size), bm(tileEdgeFromEnv("SYCL_SYRK_BM", 8)), bn(tileEdgeFromEnv("SYCL_SYRK_BN", 8)) {}

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

			submitRegSyrk(cgh, A, C, size, bm, bn);
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

	// The BM/BN suffix keeps env-swept runs (SYCL_SYRK_BM/BN=1/2/4/8)
	// distinguishable in the shared CSV.
	static std::string getBenchmarkName(BenchmarkArgs& args) {
		return "Polybench_Syrk_Opt_BM" + std::to_string(tileEdgeFromEnv("SYCL_SYRK_BM", 8)) + "BN" + std::to_string(tileEdgeFromEnv("SYCL_SYRK_BN", 8));
	}

  private:
	BenchmarkArgs args;

	const size_t size;
	const size_t bm; // register-tile edges (SYCL_SYRK_BM / SYCL_SYRK_BN)
	const size_t bn;
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
