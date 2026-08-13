#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Polybench_2mm_2;
class Polybench_2mm_1;

// Unique kernel-name type per (TM, TN, IsK1). The LLVM-MLIR backend requires a
// distinct type for every parallel_for instantiation (two parallel_for<Name>
// with different lambdas otherwise collide on KernelInfo<Name>), and we
// instantiate the register-tiled kernel once per selectable tile shape.
template <size_t TM, size_t TN, bool IsK1>
class Polybench_2mm_Tile;

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, size_t size) {
	const auto NI = size;
	const auto NJ = size;
	const auto NK = size;
	const auto NL = size;

	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NK; j++) {
			A[i * NI + j] = ((DATA_TYPE)i * j) / NI;
		}
	}

	for(size_t i = 0; i < NK; i++) {
		for(size_t j = 0; j < NJ; j++) {
			B[i * NK + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
		}
	}

	for(size_t i = 0; i < NL; i++) {
		for(size_t j = 0; j < NJ; j++) {
			C[i * NL + j] = ((DATA_TYPE)i * (j + 3)) / NL;
		}
	}

	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NL; j++) {
			D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
		}
	}
}

void mm2_cpu(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, size_t size) {
	const auto NI = size;
	const auto NJ = size;
	const auto NK = size;
	const auto NL = size;

	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NJ; j++) {
			for(size_t k = 0; k < NK; ++k) {
				C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
			}
		}
	}

	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NL; j++) {
			E[i * NL + j] = 0;
			for(size_t k = 0; k < NJ; ++k) {
				E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
			}
		}
	}
}

template <size_t TM, size_t TN, bool SeedFromC, typename AccA, typename AccB, typename AccC>
static void submitRegGemmImpl(cl::sycl::handler& cgh, AccA A, AccB B, AccC C, size_t N) {
	using namespace cl::sycl;
	using KernelName = Polybench_2mm_Tile<TM, TN, SeedFromC>;

	const size_t gi = (N + TM - 1) / TM;
	const size_t gj = (N + TN - 1) / TN;

	cgh.parallel_for<KernelName>(range<2>{gi, gj}, [=, N_ = N](item<2> item) {
		const size_t bi = item[0] * TM;
		const size_t bj = item[1] * TN;

		auto clampR = [N_](size_t r) { return r < N_ ? r : N_ - 1; };
		auto clampC = [N_](size_t c) { return c < N_ ? c : N_ - 1; };

		DATA_TYPE acc[TM][TN];
		for(size_t ii = 0; ii < TM; ii++) {
			for(size_t jj = 0; jj < TN; jj++) {
				if constexpr(SeedFromC) {
					acc[ii][jj] = C[{clampR(bi + ii), clampC(bj + jj)}];
				} else {
					acc[ii][jj] = DATA_TYPE(0);
				}
			}
		}

		for(size_t k = 0; k < N_; k++) {
			// A column for this K: a[ii] = A[bi+ii][k] (clamped row).
			DATA_TYPE a[TM];
			for(size_t ii = 0; ii < TM; ii++) {
				a[ii] = A[{clampR(bi + ii), k}];
			}
			// B row for this K: b[jj] = B[k][bj+jj] -- consecutive cols, coalesced.
			DATA_TYPE b[TN];
			for(size_t jj = 0; jj < TN; jj++) {
				b[jj] = B[{k, clampC(bj + jj)}];
			}
			// TM x TN FMAs reusing the loaded a/b in registers.
			for(size_t ii = 0; ii < TM; ii++) {
				for(size_t jj = 0; jj < TN; jj++) {
					acc[ii][jj] += a[ii] * b[jj];
				}
			}
		}

		for(size_t ii = 0; ii < TM; ii++) {
			for(size_t jj = 0; jj < TN; jj++) {
				if(bi + ii < N_ && bj + jj < N_) {
					C[{bi + ii, bj + jj}] = acc[ii][jj];
				}
			}
		}
	});
}

// Register-tiled GEMM used for both 2mm kernels (C = A x B, A/B/C all N x N
// row-major). Each work-item computes a TM x TN block of outputs held in
// registers; per K-step it loads one column of A (TM values) and one row of B
// (TN values) and does TM*TN FMAs reusing them across the block.
//
// Why this beats the naive baseline -- and why the tile is small on the DSP:
//
// The naive kernel does `C[item] += A[{i,k}] * B[{k,j}]` inside the K loop, so
// every iteration issues a global read-modify-write of C. Under the DSP build's
// POCL_EXTRA_BUILD_FLAGS="-cl-opt-disable" the compiler does NOT hoist C[item]
// into a register, so C is touched N times per output (≈⅓ of all traffic). The
// 1x1 register tile reads C once (or zero), accumulates in a scalar register,
// and stores once -- that alone is the bulk of the win on the MT3K DSP.
//
// Larger tiles (2x2, 4x4) reuse each A/B load across more outputs and can win
// on CPU/GPU where the device compiler promotes the acc/a/b arrays to
// registers. On the DSP they do not help: -cl-opt-disable lowers those stack
// arrays to real memory traffic, and the extra per-K A/B loads are not
// pipelined, so the reuse benefit is cancelled. Tiles >= 8x8 are not offered:
// their per-work-item stack (acc+a+b) overflows the DSP's tiny work-item stack
// and silently corrupts results. The default is therefore 1x1 (best on the DSP
// and never a regression elsewhere); set SYCL_2MM_TM / SYCL_2MM_TN to 2 or 4
// to enlarge the tile on CPU/GPU.
//
//   * No access::target::local, no barriers: the MT3K DSP's per-work-group
//     on-chip local memory is too small to hold a useful shared tile (only a
//     TS=2 tile -- 4 threads, 32 B -- fits; TS>=4 trips a "DDR Addr Overstep"
//     as the local accessor is allocated past the tiny local-memory window).
//   * SeedFromC: kernel 1 adds into the existing C (read_write, seeded with
//     C_init from init_array); kernel 2 writes E from zero (discard_write).
//   * Edge handling: the launch is padded up to a multiple of TM/TN. Out-of-
//     range rows/cols clamp their loads to a valid index (results discarded)
//     and the store is guarded, so non-multiple N stays correct.
//
// TM/TN are selected at runtime from env vars SYCL_2MM_TM / SYCL_2MM_TN
// (defaults 1/1) over a small compile-time set, so the tile can be matched to
// the target without rebuilding.
// Hand-unrolled scalar 2x2 register tile -- no arrays, every accumulator and
// loaded operand is a named scalar so the DSP codegen (which runs under
// -cl-opt-disable and will not promote stack arrays to registers) is forced to
// keep them in registers. This is the same reuse as the 2x2 array path but
// without the per-element stack load/store overhead that makes the array path
// slower than the 1x1 baseline on the MT3K DSP.
template <bool SeedFromC, typename AccA, typename AccB, typename AccC>
static void submitScalarGemm_2x2(cl::sycl::handler& cgh, AccA A, AccB B, AccC C, size_t N) {
	using namespace cl::sycl;
	using KernelName = Polybench_2mm_Tile<2, 2, SeedFromC>;

	const size_t gi = (N + 1) / 2;
	const size_t gj = (N + 1) / 2;

	cgh.parallel_for<KernelName>(range<2>{gi, gj}, [=, N_ = N](item<2> item) {
		const size_t i0 = item[0] * 2;
		const size_t j0 = item[1] * 2;
		const size_t i1 = i0 + 1;
		const size_t j1 = j0 + 1;
		const size_t ci0 = i0 < N_ ? i0 : N_ - 1;
		const size_t ci1 = i1 < N_ ? i1 : N_ - 1;
		const size_t cj0 = j0 < N_ ? j0 : N_ - 1;
		const size_t cj1 = j1 < N_ ? j1 : N_ - 1;

		DATA_TYPE a0, a1, b0, b1;
		DATA_TYPE acc00, acc01, acc10, acc11;
		if constexpr(SeedFromC) {
			acc00 = C[{ci0, cj0}];
			acc01 = C[{ci0, cj1}];
			acc10 = C[{ci1, cj0}];
			acc11 = C[{ci1, cj1}];
		} else {
			acc00 = acc01 = acc10 = acc11 = DATA_TYPE(0);
		}

		for(size_t k = 0; k < N_; k++) {
			a0 = A[{ci0, k}];
			a1 = A[{ci1, k}];
			b0 = B[{k, cj0}];
			b1 = B[{k, cj1}];
			acc00 += a0 * b0;
			acc01 += a0 * b1;
			acc10 += a1 * b0;
			acc11 += a1 * b1;
		}

		if(i0 < N_ && j0 < N_) C[{i0, j0}] = acc00;
		if(i0 < N_ && j1 < N_) C[{i0, j1}] = acc01;
		if(i1 < N_ && j0 < N_) C[{i1, j0}] = acc10;
		if(i1 < N_ && j1 < N_) C[{i1, j1}] = acc11;
	});
}

template <typename KernelName, bool SeedFromC, typename AccA, typename AccB, typename AccC>
static void submitRegGemm(cl::sycl::handler& cgh, AccA A, AccB B, AccC C, size_t N, size_t TM, size_t TN) {
	if(TM == 1 && TN == 1) return submitRegGemmImpl<1, 1, SeedFromC>(cgh, A, B, C, N);
	if(TM == 2 && TN == 2) return submitScalarGemm_2x2<SeedFromC>(cgh, A, B, C, N);
	if(TM == 4 && TN == 4) return submitRegGemmImpl<4, 4, SeedFromC>(cgh, A, B, C, N);
	// Fallback: 1x1.
	return submitRegGemmImpl<1, 1, SeedFromC>(cgh, A, B, C, N);
}

class Polybench_2mm {
  public:
	Polybench_2mm(const BenchmarkArgs& args) : args(args), size(args.problem_size) {
		// Register-tile shape (TM output rows x TN output cols per work-item).
		// Tunable via SYCL_2MM_TM / SYCL_2MM_TN (defaults 1/1). The 1x1 default
		// fixes the naive kernel's per-K global read-modify-write of C (the main
		// DSP win under -cl-opt-disable) and is never a regression elsewhere.
		// Larger tiles (4x4, 8x8) reuse each A/B load across more outputs and can
		// pay off on CPU/GPU, where the device compiler keeps the tile in
		// registers; on the DSP they are slower (see submitRegGemm above).
		const auto readEnv = [](const char* name, size_t def) {
			const char* e = std::getenv(name);
			const long v = e ? std::strtol(e, nullptr, 10) : (long)def;
			return v > 0 ? (size_t)v : def;
		};
		tm = readEnv("SYCL_2MM_TM", 1);
		tn = readEnv("SYCL_2MM_TN", 1);
	}

	void setup() {
		A.resize(size * size);
		B.resize(size * size);
		C.resize(size * size);
		D.resize(size * size);
		E.resize(size * size);

		init_array(A.data(), B.data(), C.data(), D.data(), size);

		A_buffer.initialize(args.device_queue, A.data(), cl::sycl::range<2>(size, size));
		B_buffer.initialize(args.device_queue, B.data(), cl::sycl::range<2>(size, size));
		C_buffer.initialize(args.device_queue, C.data(), cl::sycl::range<2>(size, size));
		D_buffer.initialize(args.device_queue, D.data(), cl::sycl::range<2>(size, size));
		E_buffer.initialize(args.device_queue, E.data(), cl::sycl::range<2>(size, size));
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		// Kernel 1: C += A * B  (seed from C_init -> read_write).
		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto B = B_buffer.get_access<access::mode::read>(cgh);
			auto C = C_buffer.get_access<access::mode::read_write>(cgh);

			submitRegGemm<Polybench_2mm_1, true>(cgh, A, B, C, size, tm, tn);
		}));

		// Kernel 2: E = C * D  (write from zero -> discard_write).
		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto C = C_buffer.get_access<access::mode::read>(cgh);
			auto D = D_buffer.get_access<access::mode::read>(cgh);
			auto E = E_buffer.get_access<access::mode::discard_write>(cgh);

			submitRegGemm<Polybench_2mm_2, false>(cgh, C, D, E, size, tm, tn);
		}));
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		init_array(A.data(), B.data(), C.data(), D.data(), size);

		std::vector<DATA_TYPE> E_cpu(size * size);
		mm2_cpu(A.data(), B.data(), C.data(), D.data(), E_cpu.data(), size);

		auto E_acc = E_buffer.get_access<cl::sycl::access::mode::read>();

		for(size_t i = 0; i < size; i++) {
			for(size_t j = 0; j < size; j++) {
				const auto diff = percentDiff(E_cpu[i * size + j], E_acc.get_pointer()[i * size + j]);
				if(diff > ERROR_THRESHOLD) return false;
			}
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_2mm_Opt"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	size_t tm{1};
	size_t tn{1};
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> B;
	std::vector<DATA_TYPE> C;
	std::vector<DATA_TYPE> D;
	std::vector<DATA_TYPE> E;

	PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> B_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> C_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> D_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> E_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_2mm>();
	return 0;
}
