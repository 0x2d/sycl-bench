#include <string>
#include <vector>

#include <cmath>
#include <cstdlib>

#include <sycl/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

#define FLOAT_N 3214212.01
#define EPS 0.005

#define sqrt_of_array_cell(x, j) sqrt(x[j])

using DATA_TYPE = float;

class CorrelationMean;
class CorrelationStd;
class CorrelationReduce;
class CorrelationCorr;

// Shared-memory tiled correlation matrix: symmat = D^T * D, where D is the
// [1..size] x [1..size] submatrix of `data` (already centered/reduced by the
// prior Reduce kernel). C[j1,j2] = sum_i D[i,j1] * D[i,j2].
//
// The baseline launches one work-item per column j1 (only ~size items, each
// serially looping j2 then i) -> ~size work-groups of parallelism on a device
// that wants ~1M threads, and it reloads every column of D O(M) times from
// global memory. Tiling restores occupancy and cuts D traffic ~TSx, turning the
// bandwidth-bound kernel into a compute-bound FMA loop -- the same lever that
// won in 2mm_opt / 3mm_opt.
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
//   * SYMMETRY: symmat is symmetric (C[j1,j2] == C[j2,j1]), so only the upper
//     triangle (block row bi <= block col bj) is computed; the lower triangle
//     is filled by a transposed mirror store. This halves both the FMA count
//     and the global D traffic on the dominant O(N^3) kernel (~2x), with no
//     change to the LDS tile / TK / register footprint -- deliberately avoiding
//     the failure modes the 2mm/atax memories document (deeper TK, register
//     blocking, cutting threads-per-WG all regressed on this occupancy-sensitive
//     device). The gi*gi grid is still launched; the bi>bj half early-returns
//     (a no-op WG is a few cycles, negligible against ~1e9 FMAs) so no triangle
//     index math is needed.
//   * The diagonal is *defined* to be 1.0 (a variable correlates perfectly with
//     itself), so the diagonal store forces 1.0 regardless of acc -- this also
//     subsumes the baseline's separate Correlation5 diagonal kernel.
template <typename KernelName, typename AccD, typename AccC>
static void submitTiledCorr(sycl::handler& cgh, AccD data, AccC symmat, size_t size) {
  using namespace sycl;

  constexpr size_t TS = 16;  // work-group tile edge in symmat; WG = TS x TS
  constexpr size_t TK = 16;  // K-depth staged in local memory per step (== TS so
                             // the TS*TS threads fill each tile with one load)
  constexpr size_t OFFSET = 1;

  const size_t N = size;
  const size_t gi = (N + TS - 1) / TS;
  const size_t gj = (N + TS - 1) / TS;

  // aTile[TK x TS] holds D[i, j1] for the tile; bTile[TK x TS] holds D[i, j2].
  accessor<DATA_TYPE, 2, access::mode::read_write, access::target::local> aTile{range<2>(TK, TS), cgh};
  accessor<DATA_TYPE, 2, access::mode::read_write, access::target::local> bTile{range<2>(TK, TS), cgh};

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
        const size_t ri = OFFSET + (kt + ty < N_ ? (kt + ty) : (N_ - 1));
        aTile[{ty, tx}] = data[{ri, OFFSET + clampC(bi + tx)}];
        bTile[{ty, tx}] = data[{ri, OFFSET + clampC(bj + tx)}];

        item.barrier(access::fence_space::local_space);

        // aTile[{kk,ty}] broadcasts across the tx row; bTile[{kk,tx}] is a
        // contiguous read across tx -> no bank conflicts.
        for(size_t kk = 0; kk < kbnd; kk++) {
          acc += aTile[{kk, ty}] * bTile[{kk, tx}];
        }

        item.barrier(access::fence_space::local_space);
      }

      if(irow < N_ && jcol < N_) {
        const size_t gj1 = OFFSET + irow;
        const size_t gj2 = OFFSET + jcol;

        if(diag_block) {
          // Diagonal block: write the whole TS x TS tile directly (both halves
          // live here), forcing the diagonal to 1.0.
          symmat[{gj1, gj2}] = (irow == jcol) ? DATA_TYPE(1.0) : acc;
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

  for(size_t i = 0; i <= M; i++) {
    for(size_t j = 0; j <= N; j++) {
      data[i * N + j] = ((DATA_TYPE)i * j) / (M + 1);
    }
  }
}

void correlation(DATA_TYPE* data, DATA_TYPE* mean, DATA_TYPE* stddev, DATA_TYPE* symmat, size_t size) {
  const auto M = size;
  const auto N = size;

  // Determine mean of column vectors of input data matrix
  for(size_t j = 1; j <= M; j++) {
    mean[j] = 0.0;

    for(size_t i = 1; i <= N; i++) {
      mean[j] += data[i * (M + 1) + j];
    }

    mean[j] /= (DATA_TYPE)FLOAT_N;
  }

  // Determine standard deviations of column vectors of data matrix.
  for(size_t j = 1; j <= M; j++) {
    stddev[j] = 0.0;

    for(size_t i = 1; i <= N; i++) {
      stddev[j] += (data[i * (M + 1) + j] - mean[j]) * (data[i * (M + 1) + j] - mean[j]);
    }

    stddev[j] /= FLOAT_N;
    stddev[j] = sqrt_of_array_cell(stddev, j);
    stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
  }

  // Center and reduce the column vectors.
  for(size_t i = 1; i <= N; i++) {
    for(size_t j = 1; j <= M; j++) {
      data[i * (M + 1) + j] -= mean[j];
      data[i * (M + 1) + j] /= sqrt(FLOAT_N);
      data[i * (M + 1) + j] /= stddev[j];
    }
  }

  // Calculate the m * m correlation matrix.
  for(size_t j1 = 1; j1 <= M - 1; j1++) {
    symmat[j1 * (M + 1) + j1] = 1.0;

    for(size_t j2 = j1 + 1; j2 <= M; j2++) {
      symmat[j1 * (M + 1) + j2] = 0.0;

      for(size_t i = 1; i <= N; i++) {
        symmat[j1 * (M + 1) + j2] += (data[i * (M + 1) + j1] * data[i * (M + 1) + j2]);
      }

      symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
    }
  }

  symmat[M * (M + 1) + M] = 1.0;
}

class Polybench_Correlation {
public:
  Polybench_Correlation(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

  void setup() {
    data.resize((size + 1) * (size + 1));
    mean.resize(size + 1);
    stddev.resize(size + 1);
    symmat.resize((size + 1) * (size + 1));

    init_arrays(data.data(), size);

    data_buffer.initialize(args.device_queue, data.data(), sycl::range<2>(size + 1, size + 1));
    mean_buffer.initialize(args.device_queue, mean.data(), sycl::range<1>(size + 1));
    stddev_buffer.initialize(args.device_queue, stddev.data(), sycl::range<1>(size + 1));
    symmat_buffer.initialize(args.device_queue, symmat.data(), sycl::range<2>(size + 1, size + 1));
  }

  void run(std::vector<sycl::event>& events) {
    using namespace sycl;

    events.push_back(args.device_queue.submit([&](handler& cgh) {
      auto data = data_buffer.get_access<access::mode::read>(cgh);
      auto mean = mean_buffer.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<CorrelationMean>(range<1>(size), [=, N_ = size](id<1> gid) {
        const id<1> offset(1);
        const auto j = gid[0] + offset[0];

        for(size_t i = 1; i <= N_; i++) {
          mean[gid + offset] += data[{i, j}];
        }
        mean[gid + offset] /= ((DATA_TYPE)FLOAT_N);
      });
    }));

    events.push_back(args.device_queue.submit([&](handler& cgh) {
      auto data = data_buffer.get_access<access::mode::read>(cgh);
      auto mean = mean_buffer.get_access<access::mode::read>(cgh);
      auto stddev = stddev_buffer.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<CorrelationStd>(range<1>(size), [=, N_ = size](id<1> gid) {
        const id<1> offset(1);
        const auto adj_id = gid + offset;
        const auto j = gid[0] + offset[0];

        for(size_t i = 1; i <= N_; i++) {
          stddev[adj_id] += (data[{i, j}] - mean[adj_id]) * (data[{i, j}] - mean[adj_id]);
        }

        stddev[adj_id] /= FLOAT_N;
        stddev[adj_id] = sycl::sqrt(stddev[adj_id]);
        stddev[adj_id] = stddev[adj_id] <= EPS ? 1.0 : stddev[adj_id];
      });
    }));

    events.push_back(args.device_queue.submit([&](handler& cgh) {
      auto data = data_buffer.get_access<access::mode::read_write>(cgh);
      auto mean = mean_buffer.get_access<access::mode::read>(cgh);
      auto stddev = stddev_buffer.get_access<access::mode::read>(cgh);

      cgh.parallel_for<CorrelationReduce>(range<2>(size, size), [=](id<2> gid) {
        const id<2> offset(1, 1);
        const auto adj_id = gid + offset;
        const auto j = gid[1] + offset[1];

        data[adj_id] -= mean[j];
        data[adj_id] /= sycl::sqrt(FLOAT_N);
        data[adj_id] /= stddev[j];
      });
    }));

    // Correlation matrix: symmat = D^T * D via shared-memory tiled GEMM, with
    // the diagonal forced to 1.0 inside the store (see submitTiledCorr).
    events.push_back(args.device_queue.submit([&](handler& cgh) {
      auto data = data_buffer.get_access<access::mode::read>(cgh);
      auto symmat = symmat_buffer.get_access<access::mode::discard_write>(cgh);

      submitTiledCorr<CorrelationCorr>(cgh, data, symmat, size);
    }));
  }

  bool verify(VerificationSetting&) {
    constexpr auto ERROR_THRESHOLD = 0.05;

    std::vector<DATA_TYPE> data_cpu((size + 1) * (size + 1));
    std::vector<DATA_TYPE> mean_cpu(size + 1);
    std::vector<DATA_TYPE> stddev_cpu(size + 1);
    std::vector<DATA_TYPE> symmat_cpu((size + 1) * (size + 1));

    // Trigger writeback
    symmat_buffer.reset();

    init_arrays(data_cpu.data(), size);
    correlation(data_cpu.data(), mean_cpu.data(), stddev_cpu.data(), symmat_cpu.data(), size);

    for(size_t i = 1; i < size + 1; i++) {
      for(size_t j = 1; j < size + 1; j++) {
        const auto diff = percentDiff(symmat_cpu[i * (size + 1) + j], symmat[i * (size + 1) + j]);
        if(diff > ERROR_THRESHOLD)
          return false;
      }
    }

    return true;
  }

  static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_Correlation_Opt"; }

private:
  BenchmarkArgs args;

  const size_t size;
  std::vector<DATA_TYPE> data;
  std::vector<DATA_TYPE> mean;
  std::vector<DATA_TYPE> stddev;
  std::vector<DATA_TYPE> symmat;

  PrefetchedBuffer<DATA_TYPE, 2> data_buffer;
  PrefetchedBuffer<DATA_TYPE, 1> mean_buffer;
  PrefetchedBuffer<DATA_TYPE, 1> stddev_buffer;
  PrefetchedBuffer<DATA_TYPE, 2> symmat_buffer;
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<Polybench_Correlation>();
  return 0;
}