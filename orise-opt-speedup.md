# `polybench/orise` optimized kernels — speedup vs. baseline

Measured on the DCU GPU ("Device 66a1", gfx906) via the `build-mlir` toolchain
(`SYCL_IMPL=LLVM-MLIR-HIP`, `DARCH_OPT=orise`, clang++ from
`/public/home/liuying/sycl-mlir/build/install`). Each `*_opt` kernel is built
alongside its reference kernel by `CMakeLists.txt`'s `file(GLOB
polybench/${ARCH_OPT}/*_opt.cpp)`. All six opt kernels **PASS verification**
(≤5% CPU-vs-GPU percent-diff) at the sizes below.

Sizes and `--local` follow the `bin/run-suite-orise` `default_profile`:
`gesummv`/`mvt` → 16384 (their `individual-benchmark-options`), the rest → 1024
(default); `--local=256 --num-runs=10` everywhere. `ONEAPI_DEVICE_SELECTOR=hip:*`
selects the HIP backend. All measurements are wall-clock run-time from
`BenchmarkManager` (the kernel-time metric is `N/A` on this backend — queue
profiling is not enabled).

## Results

| Benchmark | N | opt median [s] | base median [s] | opt min [s] | base min [s] | **Speedup (median)** | Verify |
|---|--:|--:|--:|--:|--:|--:|:--:|
| `gemm` | 1024 | 0.001857 | 0.007640 | 0.001754 | 0.007504 | **4.1×** | PASS |
| `covariance` | 1024 | 0.001907 | 1.064390 | 0.001761 | 1.060521 | **558×** | PASS |
| `gesummv` | 16384 | 0.003888 | 0.227760 | 0.003859 | 0.215612 | **58.6×** | PASS |
| `mvt` | 16384 | 0.004174 | 0.188326 | 0.003585 | 0.160673 | **45.1×** | PASS |
| `syr2k` | 1024 | 0.167477 | 0.247521 | 0.151137 | 0.245739 | **1.48×** | PASS |
| `gramschmidt` | 1024 | 1.002046 | 1.109399 | 1.000290 | 1.109026 | **1.11×** | PASS |

Speedup = baseline / opt (both median, the robust central tendency for the
bimodal DCU; min-based speedup is in the same ballpark and is listed for
reproducibility). Both opt and baseline PASS verification in every row, so the
speedups are apples-to-apples correct-output comparisons.

## Optimization per kernel

- **`gemm_opt` (4.1×)** — shared-memory tiled GEMM, `C = ALPHA·A·B + BETA·C`.
  One work-group computes a 16×16 output tile; a 16×16 A-tile and a 16×16 B-tile
  are staged in local memory per K-step, then each thread walks the 16-deep K
  reduction reusing those shared tiles. Cuts global A/B traffic ~16×, turning
  the bandwidth-bound baseline into a compute-bound FMA loop. Same lever as the
  shipped `2mm_opt`/`3mm_opt`. *(Single local accessor — see caveat.)*
- **`covariance_opt` (558×)** — the O(N³) covar kernel (`symmat = Dᵀ·D`) rewritten
  as a shared-memory tiled symmetric GEMM, mirroring `correlation_opt`: one
  work-group per 16×16 output tile, TK-deep slices of D staged twice per
  K-step, upper-triangle computation + transposed mirror store (halves FMA
  count and D traffic). The baseline launches one work-item per column j1 and
  serially loops j2 then i — almost no parallelism and O(M) reloads of every
  D column — so the tiling win here is extreme. The diagonal is the variance
  `Σᵢ D[i,j]²` (not forced to 1.0, unlike correlation). *(Single local
  accessor.)*
- **`gesummv_opt` (58.6×)** — two fused GEMVs (`tmp = A·x`, `y = B·x`) computed
  with one per-row work-group each, WG threads striding over j with a local-mem
  tree reduction producing both sums; `lid 0` stores `tmp[i]` and
  `y = ALPHA·tmp + BETA·y`. The baseline's one-thread-per-row launch gives only
  ~N/256 work-groups — under one wavefront per CU — so the device cannot hide
  memory latency; raising thread count N→N·WG restores occupancy. *(Single
  local accessor.)*
- **`mvt_opt` (45.1×)** — per-row work-group GEMV reduction applied to **both**
  kernels (`x1 = A·y1`, `x2 = A·y2`), unlike atax (where only the A·x kernel
  was favorable). Same occupancy+coalescing lever as `gesummv_opt`.
- **`syr2k_opt` (1.48×)** — register-blocked micro-kernel (BM=BN=8, 64
  accumulators/thread) for `C = alpha·(A·Bᵀ + B·Aᵀ) + beta·C`. Two-term FMA
  per k; single C read (for the beta term) and single C write per output. No
  local memory — mirrors the shipped `syrk_opt` SYRK-family pattern.
- **`gramschmidt_opt` (1.11×)** — Gram-Schmidt is sequential in k, so k stays a
  host loop. The win is fusing the baseline's two tiny per-k kernels (norm
  Gram1 — a single work-item serially summing N products — and normalize Gram2)
  into one parallel-reduction kernel: one work-group of 256 threads striding
  over i, local-mem tree reduction → `R[k,k] = sqrt(nrm)`, barrier, then the
  same threads write `Q[i,k] = A[i,k]/R[k,k]`. The dominant Gram3 rank-1 update
  is left at baseline (its two i-loops cannot fuse). Modest but safe win.

## ⚠️ Important caveat — the two-local-accessor backend bug

On this `build-mlir` (LLVM-MLIR-HIP) toolchain, **a SYCL kernel that declares
two or more `accessor<T,N,access::mode::read_write,access::target::local>`
scratch buffers mis-compiles**: it runs but produces wrong results, failing
verification even when the algorithm is byte-identical to a passing
single-accessor version. This is why `gemm_opt`, `covariance_opt`, and
`gesummv_opt` above each pack their two logical scratch tiles into **one**
local accessor (`lmem[2·TS, TS]` with the second tile at rows `[TS, 2·TS)`, or
`scratch[2·local]` with the second reduction at `[local, 2·local)`).

Confirmed in the same run that produced the table above:

| shipped opt | local accessors | Verify on `build-mlir` |
|---|--:|:--:|
| `2mm_opt` | 2 (aTile, bTile) | **FAIL** |
| `correlation_opt` | 2 (aTile, bTile) | **FAIL** |
| `gemm_opt` (this work, fixed) | 1 (packed `lmem`) | PASS |
| `covariance_opt` (this work, fixed) | 1 (packed `lmem`) | PASS |
| `gesummv_opt` (this work, fixed) | 1 (packed `scratch`) | PASS |
| `mvt_opt` (this work) | 1 | PASS |
| `gramschmidt_opt` (this work) | 1 | PASS |
| `syr2k_opt` (this work) | 0 (register-blocked) | PASS |

**Consequence:** the shipped `2mm_opt`, `3mm_opt`, and `correlation_opt` (all
two-accessor tiled kernels) — which were validated and are correct on the
`build/build.sh` LLVM-HIP toolchain — **fail verification on `build-mlir`**.
If you want those working on this path, they need the same single-accessor
packing fix. The six kernels documented here are the ones that pass on
`build-mlir` today.

## Reproduce

```
cd /public/home/liuying/sycl-bench/build-mlir
bash build.sh                       # builds all opts + reference kernels
sbatch run.slurm                    # runs all six opt + baseline pairs
```

`run.slurm` is configured with the per-test sizes above. Slurm output lands in
`slurm-<jobid>.out`; each block prints `run-time-median` / `run-time-min` /
`Verification: PASS|FAIL`.
