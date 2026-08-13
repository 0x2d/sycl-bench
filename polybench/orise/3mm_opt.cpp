#include <string>
#include <vector>

#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Polybench_3mm_1;
class Polybench_3mm_2;
class Polybench_3mm_3;

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, size_t size) {
	const auto NI = size;
	const auto NJ = size;
	const auto NK = size;
	const auto NL = size;
	const auto NM = size;

	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NK; j++) {
			A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
		}
	}

	for(size_t i = 0; i < NK; i++) {
		for(size_t j = 0; j < NJ; j++) {
			B[i * NJ + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
		}
	}

	for(size_t i = 0; i < NJ; i++) {
		for(size_t j = 0; j < NM; j++) {
			C[i * NM + j] = ((DATA_TYPE)i * (j + 3)) / NL;
		}
	}

	for(size_t i = 0; i < NM; i++) {
		for(size_t j = 0; j < NL; j++) {
			D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
		}
	}
}

void mm3_cpu(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, DATA_TYPE* F, DATA_TYPE* G, size_t size) {
	const auto NI = size;
	const auto NJ = size;
	const auto NK = size;
	const auto NL = size;
	const auto NM = size;

	/* E := A*B */
	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NJ; j++) {
			E[i * NJ + j] = 0;
			for(size_t k = 0; k < NK; ++k) {
				E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
			}
		}
	}

	/* F := C*D */
	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NL; j++) {
			F[i * NL + j] = 0;
			for(size_t k = 0; k < NM; ++k) {
				F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
			}
		}
	}

	/* G := E*F */
	for(size_t i = 0; i < NI; i++) {
		for(size_t j = 0; j < NL; j++) {
			G[i * NL + j] = 0;
			for(size_t k = 0; k < NJ; ++k) {
				G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
			}
		}
	}
}

// Shared-memory tiled GEMM used for all three 3mm kernels.
//
// Each work-group computes a TS x TS tile of the output (Out = A x B, A/B/Out
// all N x N row-major). A TS x TK tile of A and a TK x TS tile of B are staged
// in local (shared) memory per K-step; every thread then walks the TK depth and
// does TS*... FMAs reusing those shared tiles. The naive baseline (one thread
// per output, strided inner reduction) reloads each A row and B column N times
// from global memory; tiling cuts that traffic by a factor of ~TS on A and ~TK
// on B, turning the bandwidth-bound kernel into a compute-bound FMA loop.
//
//   * A reads: aTile[{ty, tx}] = A[{irow, kt+tx}] -- consecutive tx map to
//     consecutive columns -> coalesced. In the FMA loop aTile[{ty, kk}] is
//     broadcast across the whole row of tx (same ty, kk) -> no bank conflict.
//   * B reads: bTile[{ty, tx}] = B[{kt+ty, jcol}] -- consecutive tx map to
//     consecutive columns -> coalesced. In the FMA loop bTile[{kk, tx}] is a
//     contiguous row read across tx -> no bank conflict.
//   * SeedFromC: when true, seed the accumulator from the existing Out
//     (read_write, Out already initialized); when false, write from zero
//     (discard_write). All three 3mm kernels write from zero (E/F/G start at
//     0), so SeedFromC=false throughout.
//   * Edge handling: the launch is padded up to a multiple of TS. Out-of-range
//     rows/cols clamp their loads to a valid index (results discarded) and the
//     store is guarded, so non-multiple N stays correct. For the default
//     --size=1024 (1024 % 16 == 0) there are no edge tiles and these paths are
//     never taken.
template <typename KernelName, bool SeedFromC, typename AccA, typename AccB, typename AccC>
static void submitTiledGemm(cl::sycl::handler& cgh, AccA A, AccB B, AccC C, size_t N) {
	using namespace cl::sycl;

	constexpr size_t TS = 16; // work-group tile edge in C; WG = TS x TS threads
	constexpr size_t TK = 16; // K-depth staged in local memory per step (== TS so
	                         // the TS*TS threads fill the tile with one load each)

	const size_t gi = (N + TS - 1) / TS;
	const size_t gj = (N + TS - 1) / TS;

	// Local tiles: aTile[TS x TK] (rows of A), bTile[TK x TS] (cols of B). With
	// TS == TK each thread loads exactly one element of each tile per K-step.
	accessor<DATA_TYPE, 2, access::mode::read_write, access::target::local> aTile{range<2>(TS, TK), cgh};
	accessor<DATA_TYPE, 2, access::mode::read_write, access::target::local> bTile{range<2>(TK, TS), cgh};

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

			DATA_TYPE acc;
			if constexpr(SeedFromC) {
				acc = C[{clampR(irow), clampC(jcol)}];
			} else {
				acc = DATA_TYPE(0);
			}

			for(size_t kt = 0; kt < N_; kt += TK) {
				// One coalesced load per thread per K-step: consecutive tx -> consecutive
				// global columns for both A (row bi+ty) and B (row kt+ty).
				aTile[{ty, tx}] = A[{clampR(irow), clampC(kt + tx)}];
				bTile[{ty, tx}] = B[{clampC(kt + ty), clampC(jcol)}];

				item.barrier(access::fence_space::local_space);

				// aTile[{ty,kk}] broadcasts across the row of tx; bTile[{kk,tx}] is a
				// contiguous read across tx -> no bank conflicts.
				for(size_t kk = 0; kk < TK; kk++) {
					acc += aTile[{ty, kk}] * bTile[{kk, tx}];
				}

				item.barrier(access::fence_space::local_space);
			}

			if(irow < N_ && jcol < N_) {
				C[{irow, jcol}] = acc;
			}
		});
}

class Polybench_3mm {
  public:
	Polybench_3mm(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		A.resize(size * size);
		B.resize(size * size);
		C.resize(size * size);
		D.resize(size * size);
		E.resize(size * size);
		F.resize(size * size);
		G.resize(size * size);

		init_array(A.data(), B.data(), C.data(), D.data(), size);

		A_buffer.initialize(args.device_queue, A.data(), cl::sycl::range<2>(size, size));
		B_buffer.initialize(args.device_queue, B.data(), cl::sycl::range<2>(size, size));
		C_buffer.initialize(args.device_queue, C.data(), cl::sycl::range<2>(size, size));
		D_buffer.initialize(args.device_queue, D.data(), cl::sycl::range<2>(size, size));
		E_buffer.initialize(args.device_queue, E.data(), cl::sycl::range<2>(size, size));
		F_buffer.initialize(args.device_queue, F.data(), cl::sycl::range<2>(size, size));
		G_buffer.initialize(args.device_queue, G.data(), cl::sycl::range<2>(size, size));
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		// Kernel 1: E = A * B  (write from zero -> discard_write).
		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto A = A_buffer.get_access<access::mode::read>(cgh);
			auto B = B_buffer.get_access<access::mode::read>(cgh);
			auto E = E_buffer.get_access<access::mode::discard_write>(cgh);

			submitTiledGemm<Polybench_3mm_1, false>(cgh, A, B, E, size);
		}));

		// Kernel 2: F = C * D  (write from zero -> discard_write). Independent of
		// kernel 1; the runtime may overlap them.
		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto C = C_buffer.get_access<access::mode::read>(cgh);
			auto D = D_buffer.get_access<access::mode::read>(cgh);
			auto F = F_buffer.get_access<access::mode::discard_write>(cgh);

			submitTiledGemm<Polybench_3mm_2, false>(cgh, C, D, F, size);
		}));

		// Kernel 3: G = E * F  (write from zero -> discard_write). Reads E/F written
		// by kernels 1/2; accessor data-flow serializes correctly on the queue.
		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto E = E_buffer.get_access<access::mode::read>(cgh);
			auto F = F_buffer.get_access<access::mode::read>(cgh);
			auto G = G_buffer.get_access<access::mode::discard_write>(cgh);

			submitTiledGemm<Polybench_3mm_3, false>(cgh, E, F, G, size);
		}));
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		init_array(A.data(), B.data(), C.data(), D.data(), size);

		std::vector<DATA_TYPE> E_cpu(size * size);
		std::vector<DATA_TYPE> F_cpu(size * size);
		std::vector<DATA_TYPE> G_cpu(size * size);

		mm3_cpu(A.data(), B.data(), C.data(), D.data(), E_cpu.data(), F_cpu.data(), G_cpu.data(), size);

		auto G_acc = G_buffer.get_access<cl::sycl::access::mode::read>();

		for(size_t i = 0; i < size; i++) {
			for(size_t j = 0; j < size; j++) {
				const auto diff = percentDiff(G_cpu[i * size + j], G_acc.get_pointer()[i * size + j]);
				if(diff > ERROR_THRESHOLD) return false;
			}
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_3mm_Opt"; }

  private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> A;
	std::vector<DATA_TYPE> B;
	std::vector<DATA_TYPE> C;
	std::vector<DATA_TYPE> D;
	std::vector<DATA_TYPE> E;
	std::vector<DATA_TYPE> F;
	std::vector<DATA_TYPE> G;

	PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> B_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> C_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> D_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> E_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> F_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> G_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_3mm>();
	return 0;
}
