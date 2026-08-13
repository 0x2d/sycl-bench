#include <string>
#include <vector>

#include <cmath>
#include <cstdlib>

#include <CL/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class CovarianceMean;
class CovarianceReduce;
class CovarianceCovar;

constexpr DATA_TYPE float_n = 3214212.01;

// Shared-memory tiled covariance matrix: symmat = D^T * D, where D is the
// [1..size] x [1..size] submatrix of `data` (already centered by the prior
// Reduce kernel). C[j1,j2] = sum_i D[i,j1] * D[i,j2].
//
// This is the [[2mm-opt-shared-mem-tile]] / correlation_opt lever: the baseline
// launches one work-item per column j1 (only ~size items, each serially looping
// j2 then i) -> ~size work-groups of parallelism on a device that wants ~1M
// threads, and it reloads every column of D O(M) times from global memory.
// Tiling restores occupancy and cuts D traffic ~TSx, turning the bandwidth-bound
// kernel into a compute-bound FMA loop.
//
//   * One work-group per TS x TS output tile; a TK-deep slice of D's i-axis is
//     staged in local memory twice per K-step (aTile for the j1 operand, bTile
//     for the j2 operand -- both equal to D since the product is D^T*D).
//   * Loads: aTile[{ty,tx}] = D[{kt+ty, bi+tx}], bTile[{ty,tx}] = D[{kt+ty,
//     bj+tx}]. Consecutive tx -> consecutive columns of D at a fixed row ->
//     coalesced (D is row-major). aTile[{kk,ty}] broadcasts across the tx row,
//     bTile[{kk,tx}] is a contiguous read across tx -> no bank conflicts.
//   * OFFSET=1 because polybench stores the active submatrix at indices
//     [1..size] of a (size+1)x(size+1) buffer.
//   * SYMMETRY: symmat is symmetric, so only the upper triangle (block row
//     bi <= block col bj) is computed; the lower triangle is filled by a
//     transposed mirror store. Diagonal blocks (bi==bj) write every cell
//     directly (both halves live in the square block); off-diagonal upper
//     blocks write the upper cell and mirror it into the lower triangle.
//   * Unlike correlation_opt, the diagonal is NOT forced to 1.0 here -- for
//     covariance the diagonal is the variance sum_i D[i,j]^2, which the tiled
//     kernel computes directly as `acc` when irow==jcol (the baseline's
//     `symmat[{j1,j1}]=1.0` is dead-overwritten; the CPU ref has no 1.0).
template <typename KernelName, typename AccD, typename AccC>
static void submitTiledCovar(cl::sycl::handler& cgh, AccD data, AccC symmat, size_t size) {
	using namespace cl::sycl;

	constexpr size_t TS = 16;  // work-group tile edge in symmat; WG = TS x TS
	constexpr size_t TK = 16;  // K-depth staged in local memory per step (== TS so
	                           // the TS*TS threads fill each tile with one load)
	constexpr size_t OFFSET = 1;

	const size_t N = size;
	const size_t gi = (N + TS - 1) / TS;
	const size_t gj = (N + TS - 1) / TS;

	// aTile[TK x TS] and bTile[TK x TS] are packed into a SINGLE local accessor
	// lmem[2*TK x TS] (TK == TS == 16): aTile occupies rows [0,TK), bTile rows
	// [TK,2*TK). On this LLVM-MLIR-HIP backend a kernel with two local accessors
	// mis-compiles (the 2-accessor tiled covar FAILED verification while the
	// 1-accessor reductions in mvt_opt / gramschmidt_opt PASS); one accessor works.
	accessor<DATA_TYPE, 2, access::mode::read_write, access::target::local> lmem{range<2>(2 * TK, TS), cgh};

	cgh.parallel_for<KernelName>(nd_range<2>{range<2>(gi * TS, gj * TS), range<2>(TS, TS)},
		[=, N_ = N](nd_item<2> item) {
			const size_t ty = item.get_local_id(0);
			const size_t tx = item.get_local_id(1);
			const size_t bi = item.get_group(0) * TS;  // tile's j1 origin
			const size_t bj = item.get_group(1) * TS;  // tile's j2 origin

			// Lower-triangle tile: its result is produced by the mirror store of the
			// symmetric upper tile (bj, bi). Skip to avoid computing it twice.
			if(bi > bj) return;

			const size_t irow = bi + ty;  // logical j1
			const size_t jcol = bj + tx;  // logical j2
			const bool diag_block = (bi == bj);

			auto clampC = [N_](size_t c) { return c < N_ ? c : N_ - 1; };

			DATA_TYPE acc = DATA_TYPE(0);

			for(size_t kt = 0; kt < N_; kt += TK) {
				const size_t kbnd = (kt + TK <= N_) ? TK : (N_ - kt);
				// Coalesced load: consecutive tx -> consecutive columns of D (fixed row).
				// aTile[{ty,tx}] = lmem[{ty,tx}]; bTile[{ty,tx}] = lmem[{TK+ty,tx}]
				const size_t ri = OFFSET + (kt + ty < N_ ? (kt + ty) : (N_ - 1));
				lmem[{ty, tx}] = data[{ri, OFFSET + clampC(bi + tx)}];
				lmem[{TK + ty, tx}] = data[{ri, OFFSET + clampC(bj + tx)}];

				item.barrier(access::fence_space::local_space);

				// aTile[{kk,ty}] = lmem[{kk,ty}]; bTile[{kk,tx}] = lmem[{TK+kk,tx}]
				for(size_t kk = 0; kk < kbnd; kk++) {
					acc += lmem[{kk, ty}] * lmem[{TK + kk, tx}];
				}

				item.barrier(access::fence_space::local_space);
			}

			if(irow < N_ && jcol < N_) {
				const size_t gj1 = OFFSET + irow;
				const size_t gj2 = OFFSET + jcol;

				if(diag_block) {
					// Diagonal block: write the whole TS x TS tile directly (both halves
					// live here). The diagonal cell gets acc (the variance), not 1.0.
					symmat[{gj1, gj2}] = acc;
				} else {
					// Upper off-diagonal block (bi < bj => irow < jcol): write the upper
					// element and mirror it into the lower triangle. One global read tile
					// feeds two output tiles.
					symmat[{gj1, gj2}] = acc;
					symmat[{gj2, gj1}] = acc;
				}
			}
		});
}

void init_arrays(DATA_TYPE* data, size_t size) {
	const auto M = size;
	const auto N = size;

	for(size_t i = 0; i < M; i++) {
		for(size_t j = 0; j < N; j++) {
			data[i * (N + 1) + j] = ((DATA_TYPE)i * j) / M;
		}
	}
}

void covariance(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean, size_t size) {
	const auto M = size;
	const auto N = size;

	// Determine mean of column vectors of input data matrix
	for(size_t j = 1; j <= M; j++) {
		mean[j] = 0.0;
		for(size_t i = 1; i <= N; i++) {
			mean[j] += data[i * (M + 1) + j];
		}
		mean[j] /= float_n;
	}

	// Center the column vectors.
	for(size_t i = 1; i <= N; i++) {
		for(size_t j = 1; j <= M; j++) {
			data[i * (M + 1) + j] -= mean[j];
		}
	}

	// Calculate the m * m covariance matrix.
	for(size_t j1 = 1; j1 <= M; j1++) {
		for(size_t j2 = j1; j2 <= M; j2++) {
			symmat[j1 * (M + 1) + j2] = 0.0;
			for(size_t i = 1; i <= N; i++) {
				symmat[j1 * (M + 1) + j2] += data[i * (M + 1) + j1] * data[i * (M + 1) + j2];
			}
			symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
		}
	}
}

class Polybench_Covariance {
public:
	Polybench_Covariance(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

	void setup() {
		data.resize((size + 1) * (size + 1));
		symmat.resize((size + 1) * (size + 1));
		mean.resize(size + 1);

		init_arrays(data.data(), size);

		data_buffer.initialize(args.device_queue, data.data(), cl::sycl::range<2>(size + 1, size + 1));
		symmat_buffer.initialize(args.device_queue, symmat.data(), cl::sycl::range<2>(size + 1, size + 1));
		mean_buffer.initialize(args.device_queue, mean.data(), cl::sycl::range<1>(size + 1));
	}

	void run(std::vector<cl::sycl::event>& events) {
		using namespace cl::sycl;

		// Mean of column vectors (baseline -- O(N^2), not the bottleneck).
		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto data = data_buffer.get_access<access::mode::read>(cgh);
			auto mean = mean_buffer.get_access<access::mode::discard_write>(cgh);

			cgh.parallel_for<CovarianceMean>(range<1>(size), id<1>(1), [=, N_ = size](item<1> item) {
				const auto j = item[0];

				mean[item] = 0;
				for(size_t i = 1; i <= N_; i++) {
					mean[item] += data[{i, j}];
				}
				mean[item] /= float_n;
			});
		}));

		// Center the column vectors (baseline -- covariance only subtracts the
		// mean, no division by stddev/sqrt(float_n) unlike correlation).
		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto mean = mean_buffer.get_access<access::mode::read>(cgh);
			auto data = data_buffer.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<CovarianceReduce>(range<2>(size, size), id<2>(1, 1), [=](item<2> item) {
				const auto j = item[1];
				data[item] -= mean[j];
			});
		}));

		// Covariance matrix: symmat = D^T * D via shared-memory tiled GEMM.
		events.push_back(args.device_queue.submit([&](handler& cgh) {
			auto data = data_buffer.get_access<access::mode::read>(cgh);
			auto symmat = symmat_buffer.get_access<access::mode::discard_write>(cgh);

			submitTiledCovar<CovarianceCovar>(cgh, data, symmat, size);
		}));
	}

	bool verify(VerificationSetting&) {
		constexpr auto ERROR_THRESHOLD = 0.05;

		std::vector<DATA_TYPE> data_cpu((size + 1) * (size + 1));
		std::vector<DATA_TYPE> symmat_cpu((size + 1) * (size + 1));
		std::vector<DATA_TYPE> mean_cpu(size + 1);

		// Trigger writeback
		symmat_buffer.reset();

		init_arrays(data_cpu.data(), size);

		covariance(data_cpu.data(), symmat_cpu.data(), mean_cpu.data(), size);

		for(size_t i = 1; i < size + 1; i++) {
			for(size_t j = 1; j < size + 1; j++) {
				const auto diff = percentDiff(symmat_cpu[i * (size + 1) + j], symmat[i * (size + 1) + j]);
				if(diff > ERROR_THRESHOLD) return false;
			}
		}

		return true;
	}

	static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Covariance_Opt"; }

private:
	BenchmarkArgs args;

	const size_t size;
	std::vector<DATA_TYPE> data;
	std::vector<DATA_TYPE> symmat;
	std::vector<DATA_TYPE> mean;

	PrefetchedBuffer<DATA_TYPE, 2> data_buffer;
	PrefetchedBuffer<DATA_TYPE, 2> symmat_buffer;
	PrefetchedBuffer<DATA_TYPE, 1> mean_buffer;
};

int main(int argc, char** argv) {
	BenchmarkApp app(argc, argv);
	app.run<Polybench_Covariance>();
	return 0;
}
