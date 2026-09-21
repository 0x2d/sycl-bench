#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

// Unique kernel-name type per TS. The LLVM-MLIR backend requires a distinct
// type for every parallel_for instantiation; the tiled kernel is instantiated
// once per selectable tile shape.
template <size_t TS>
class Polybench_Gemm1_T;

// Tile edge selectable at run time (compile machine != run machine, so the
// sweep must not require a rebuild). Returns def when unset/invalid.
static size_t tileEdgeFromEnv(const char* env, size_t def) {
	const char* e = std::getenv(env);
	const long v = e ? std::strtol(e, nullptr, 10) : 0;
	return (v == 4 || v == 8 || v == 16 || v == 32) ? (size_t)v : def;
}

// ALPHA/BETA are macros in the reference (polybench/gemm.cpp); make them constexpr
// here so they fold into FMA immediates.
constexpr DATA_TYPE ALPHA = 32412;
constexpr DATA_TYPE BETA = 2123;

void init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, size_t size) {
	const auto NI = size;
	const auto NJ = size;
	const auto NK = size;

	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NK; j++) {
			A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
		}
	}

	for(size_t i = 0; i < NK; i++) {
		for(size_t j = 0; j < NJ; j++) {
			B[i * NJ + j] = ((DATA_TYPE)i * j + 1) / NJ;
		}
	}

	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NJ; j++) {
			C[i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
		}
	}
}

void gemm(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, size_t size) {
	const auto NI = size;
	const auto NJ = size;
	const auto NK = size;

	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NJ; j++) {
			C[i * NJ + j] *= BETA;

			for(size_t k = 0; k < NK; ++k) {
				C[i * NJ + j] += ALPHA * A[i * NK + k] * B[k * NJ + j];
			}
		}
	}
}

// Shared-memory tiled GEMM: C = ALPHA*A*B + BETA*C, A/B/C all N x N row-major.
//
// This is the 2mm_opt / 3mm_opt tiled-GEMM lever (see [[2mm-opt-shared-mem-tile]]):
// one work-group computes a TS x TS output tile, a TS x TK tile of A and a TK x TS
// tile of B are staged in local memory per K-step, then each thread walks the TK
// depth doing TS FMA reusing those shared tiles. The baseline (one thread per
// output, strided inner reduction) reloads each A row / B column N times from
// global memory; tiling cuts that traffic ~TSx, turning the bandwidth-bound
// kernel into a compute-bound FMA loop.
//
//   * SeedFromC is true here: the accumulator is seeded from C*BETA (the baseline
//     does `C *= BETA` then `C += ALPHA*A*B`), so C is read_write and read once
//     per output. The FMA loop adds ALPHA*a*b.
//   * A reads: aTile[{ty,tx}] = A[{irow, kt+tx}] -- consecutive tx -> consecutive
//     columns -> coalesced; aTile[{ty,kk}] broadcasts across the tx row.
//   * B reads: bTile[{ty,tx}] = B[{kt+ty, jcol}] -- consecutive tx -> consecutive
//     columns -> coalesced; bTile[{kk,tx}] is a contiguous read across tx.
//   * Edge clamping keeps non-multiple N correct, and the K-loop tail is
//     bounded by kbnd so clamped loads never contribute to the accumulator
//     (no-op for the suite's --size=512, which every selectable TS divides).
//
// TS (== TK) is selected at run time via SYCL_GEMM_TS (4/8/16/32; default 8 ->
// a 64-thread work-group, one work-item per CPE of a 64-CPE Sunway core
// group). The kernel is instantiated once per shape so every loop bound stays
// a compile-time constant.
template <size_t TS, template <size_t> class KernelNameT, typename AccA, typename AccB, typename AccC>
static void submitTiledGemm(cl::sycl::handler& cgh, AccA A, AccB B, AccC C, size_t N) {
	using namespace cl::sycl;
	using KernelName = KernelNameT<TS>;

	constexpr size_t TK = TS; // K-depth staged in local memory per step (== TS so
	                          // the TS*TS threads fill each tile with one load)

	const size_t gi = (N + TS - 1) / TS;
	const size_t gj = (N + TS - 1) / TS;

	// aTile[TS x TK] (rows of A) and bTile[TK x TS] (cols of B) are packed into a
	// SINGLE local accessor lmem[2*TS x TS]: aTile occupies rows [0,TS), bTile
	// occupies rows [TS,2*TS). Structure kept from the orise twin, where a kernel
	// with two local accessors mis-compiled on the LLVM-MLIR-HIP backend (the
	// 2-accessor tiled GEMM FAILED verification while the structurally identical
	// 1-accessor reductions PASSED); one accessor is fine everywhere.
	accessor<DATA_TYPE, 2, access::mode::read_write, access::target::local> lmem{range<2>(2 * TS, TS), cgh};

	cgh.parallel_for<KernelName>(nd_range<2>{range<2>(gi * TS, gj * TS), range<2>(TS, TS)},
		[=, N_ = N](nd_item<2> item) {
			const size_t ty = item.get_local_id(0);
			const size_t tx = item.get_local_id(1);
			const size_t bi = item.get_group(0) * TS;
			const size_t bj = item.get_group(1) * TS;
			const size_t irow = bi + ty;
			const size_t jcol = bj + tx;

			auto clampR = [N_](size_t r) { return r < N_ ? r : N_ - 1; };
			auto clampC = [N_](size_t c) { return c < N_ ? c : N_ - 1; };

			// Seed from C * BETA (read_write). Rows/cols past N-1 (only possible for
			// the last block when N % TS != 0) clamp to a valid index so the A
			// reads below stay in range; their results are discarded by the
			// guarded store at the end.
			DATA_TYPE acc = C[{clampR(irow), clampC(jcol)}] * BETA;

			for(size_t kt = 0; kt < N_; kt += TK) {
				const size_t kbnd = (kt + TK <= N_) ? TK : (N_ - kt);
				// aTile[{ty,tx}] = lmem[{ty,tx}]; bTile[{ty,tx}] = lmem[{TS+ty,tx}]
				lmem[{ty, tx}] = A[{clampR(irow), clampC(kt + tx)}];
				lmem[{TS + ty, tx}] = B[{clampC(kt + ty), clampC(jcol)}];

				item.barrier(access::fence_space::local_space);

				// aTile[{ty,kk}] = lmem[{ty,kk}]; bTile[{kk,tx}] = lmem[{TS+kk,tx}]
				for(size_t kk = 0; kk < kbnd; kk++) {
					acc += ALPHA * lmem[{ty, kk}] * lmem[{TS + kk, tx}];
				}

				item.barrier(access::fence_space::local_space);
			}

			if(irow < N_ && jcol < N_) {
				C[{irow, jcol}] = acc;
			}
		});
}

class Polybench_Gemm {
  public:
	Polybench_Gemm(const BenchmarkArgs& args) : args(args), size(args.problem_size), ts(tileEdgeFromEnv("SYCL_GEMM_TS", 8)) {}

	void setup() {
		A.resize(size * size);
		B.resize(size * size);
		C.resize(size * size);

		init(A.data(), B.data(), C.data(), size);

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

			if(ts == 4) submitTiledGemm<4, Polybench_Gemm1_T>(cgh, A, B, C, size);
			else if(ts == 16) submitTiledGemm<16, Polybench_Gemm1_T>(cgh, A, B, C, size);
			else if(ts == 32) submitTiledGemm<32, Polybench_Gemm1_T>(cgh, A, B, C, size);
			else submitTiledGemm<8, Polybench_Gemm1_T>(cgh, A, B, C, size);
		}));
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		// Trigger writeback
		C_buffer.reset();

		std::vector<DATA_TYPE> C_cpu(size * size);

		init(A.data(), B.data(), C_cpu.data(), size);

		gemm(A.data(), B.data(), C_cpu.data(), size);

		for(size_t i = 0; i < size; i++) {
			for(size_t j = 0; j < size; j++) {
				const auto diff = percentDiff(C_cpu[i * size + j], C[i * size + j]);
				if(diff > ERROR_THRESHOLD) return false;
			}
		}

		return true;
	}

	// The TS suffix keeps env-swept runs (SYCL_GEMM_TS=4/8/16/32) distinguishable
	// in the shared CSV.
	static std::string getBenchmarkName(BenchmarkArgs& args) {
		return "Polybench_Gemm_Opt_TS" + std::to_string(tileEdgeFromEnv("SYCL_GEMM_TS", 8));
	}

private:
	BenchmarkArgs args;

	const size_t size;
	const size_t ts; // shared-memory tile edge (SYCL_GEMM_TS; see submitTiledGemm)
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> B;
	std::vector<DATA_TYPE> C;

	PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> B_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> C_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Gemm>();
	return 0;
}
