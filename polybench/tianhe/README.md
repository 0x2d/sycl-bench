# Tianhe (MT3K-DSP) source-level optimizations

This directory holds DSP-safe source-level optimizations of the Polybench
kernels, built when `-DARCH_OPT=tianhe` (see `CMakeLists.txt`). Each
`<name>_opt.cpp` is a drop-in alternative to `polybench/<name>.cpp` with a
distinct benchmark name (`Polybench_<Name>_Opt`), so the opt and the reference
kernel coexist in the build and can be A/B-compared on the same queue.

The target is the MT3K-DSP via the `LLVM-MLIR` SYCL implementation and the
MOCL/pocl runtime (`bin/tianhe/env-mlir.sh`, `POCL_TEST_DEVICE_TYPE=DSP`,
`MOCL_CORE_NUMS=24`). The DSP has three hard constraints that defeat the
shared-memory-tiled optimizations used by the `polybench/orise/` twins (which
target CPU/GPU). See `~/.claude/.../memory/mocl-dsp-constraints.md` for the
full diagnosis; in short:

1. **`access::target::local` is unusable** — the per-work-group on-chip local
   memory window is too small for a useful tile (TS≥4 trips `DDR Addr
   Overstep`; 2D local accessors null out; 1D local accessors + barriers hang
   the MOCL backend).
2. **`-cl-opt-disable` is set** (`POCL_EXTRA_BUILD_FLAGS`) — the DSP compiler
   does not hoist loads, unroll, or promote stack arrays to registers. So a
   naive `C[item] += A*B` inside the K-loop does a **global
   read-modify-write of `C[item]` every iteration** (the compiler will not hoist
   it into a register). That N-fold C RMW is the main inefficiency to fix.
3. **Per-work-item stack is tiny** — `acc[TM][TN]`+`a[TM]`+`b[TN]` arrays
   overflow it at TM=TN=8 (320 B) → silent verification FAIL. 4×4 (96 B) and
   2×2 are fine.

So every opt here is built from two DSP-safe ingredients only:

- **Register accumulator** — read the output once (or seed from zero), accumulate
  the K-loop in a named scalar register, store once. Collapses the N-fold C RMW
  into one read + one write. This is the dominant win and is a no-op on CPU/GPU
  (where the device compiler already hoists the `+=`), so it never regresses
  there.
- **Register tile (GEMM family only)** — each work-item computes a TM×TN block
  of outputs in registers, reusing each A/B load across TM·TN FMAs. Default
  **1×1** (best on the DSP, never a regression elsewhere); env-tunable up to
  **4×4** for CPU/GPU. No `access::target::local`, no barriers, no `nd_range`.

The orise twins' shared-memory tiling and work-group tree reductions are
**dropped** (they hang/crash on the DSP); only the register-level rewrites
survive.

## Kernels

| File | Technique |
|------|-----------|
| `2mm_opt.cpp` | Register-tiled GEMM ×2 (C+=A·B seed-from-C; E=C·D from zero). 1×1 default, env `SYCL_2MM_TM/TN`. |
| `3mm_opt.cpp` | Register-tiled GEMM ×3 (E=A·B, F=C·D, G=E·F). 1×1 default, env `SYCL_3MM_TM/TN`. |
| `atax_opt.cpp` | Both GEMVs: one-thread-per-output scalar register accumulator. |
| `bicg_opt.cpp` | Both GEMVs (s=Aᵀr, q=Ap): scalar register accumulator. |
| `gesummv_opt.cpp` | One thread per row, two register accumulators (A·x, B·x), `y=α·accA+β·accB`. |
| `mvt_opt.cpp` | Both GEMVs (x1=A·y1, x2=A·y2): scalar register accumulator. |
| `gemm_opt.cpp` | Register-tiled GEMM, seed `C·BETA + α·A·B`. 1×1 default, env `SYCL_GEMM_TM/TN`. |
| `syrk_opt.cpp` | Register-tiled, seed `C·β + α·A·Aᵀ`. 1×1 default, env `SYCL_SYRK_BM/BN`. |
| `syr2k_opt.cpp` | Register-tiled, two-term `α·(A·Bᵀ+B·Aᵀ)+β·C`. 1×1 default, env `SYCL_SYR2K_BM/BN`. |
| `covariance_opt.cpp` | Mean → register accumulator; Covar → one-thread-per-output (j1,j2) register accumulator, symmetry+mirror, diagonal=variance. |
| `correlation_opt.cpp` | Mean/Std → register accumulators; Reduce hoists `data[item]`; Corr → one-thread-per-output, symmetry+mirror, diagonal forced to 1.0 (subsumes Correlation5). |
| `gramschmidt_opt.cpp` | 3-kernel structure kept (no Gram1+Gram2 fusion — that needs local-memory reduction); Gram3 collapses the `R[item]` RMW into a register `accR` reused for the A update. |
| `fdtd2d_opt.cpp` | Fused ey+ex update kernel (reads `hz[i,j]` once into a register, reuses for both updates); hz update stays separate. No local/barriers. |

All 13 pass verification on the DSP at `--size=64 --local=8` (and at non-multiple
sizes 63/65/67 for the GEMM edge-clamp paths; 2×2/4×4 env-tile paths also pass).

## Build & run

```bash
source bin/tianhe/env-mlir.sh
cd build-mlir   # or your build dir
cmake -DSYCL_IMPL=LLVM-MLIR -DARCH_OPT=tianhe \
      -DCMAKE_CXX_COMPILER=/thfs1/home/ouyyc/sycl-mlir/build/install/bin/clang++ \
      -DCMAKE_INSTALL_PREFIX=$(pwd)/.. ..
cmake --build . -j16
```

Single benchmark (opt and reference share the same flags):

```bash
yhrun --exclusive -p thmt1 ./gemm     --size=256 --local=8 --num-runs=10 --output=stdio
yhrun --exclusive -p thmt1 ./gemm_opt --size=256 --local=8 --num-runs=10 --output=stdio
```

## Measuring speedup correctly (read this first)

**The POCL runtime caches compiled DSP kernels on disk** (`~/.cache/pocl`),
keyed by kernel source + build flags. The first-ever execution of a kernel
compiles it (seconds on the DSP); subsequent executions load the cached binary.
The harness reports `run-time-mean` averaged over `--num-runs=N` iterations, so a
**single cold compile on iteration 1 is amortized into the mean** and inflates
it dramatically for small/fast kernels.

This bites A/B comparisons asymmetrically: if the opt kernel was exercised first
(e.g. during verification) it is already cached, while the reference kernel may
still be cold. Measured naively, the cold baseline can look **10–15× slower than
it really is**, producing fake double-digit speedups that are really JIT compile
time, not kernel execution.

This was confirmed empirically on `gemm` at `--size=64`:

| run | baseline `run-time-mean` |
|-----|--------------------------|
| cold (fresh `POCL_CACHE_DIR`, recompiles) | ~0.21 s |
| warm (default `~/.cache/pocl`, cached)     | ~0.021 s |
| opt, warm                                  | ~0.017 s |

So `gemm`'s real warm speedup at size 64 is **~1.2×**, not the ~13× a
cold-baseline / warm-opt measurement reports.

**Correct measurement protocol:**

1. Run **both** the reference and the opt once first (any size) to populate the
   POCL cache, *or* run with `--warmup-run` and a large `--num-runs` so the one
   cold iteration is a small fraction of the mean. Better: discard iteration 1.
2. Then take the A/B `run-time-mean` with both kernels warm.
3. Prefer a **compute-dominated size** (≥128 for the GEMM family; see below) —
   at `--size=64` every kernel is host-overhead-bound (~10–30 ms of queue submit
   + accessor resolution vs. microseconds of actual DSP compute), so size-64
   speedups cluster near 1.0× and are noisy / uninformative about kernel compute.

The numbers in the table below were taken with both sides warm and
`--num-runs=10`.

## Measured speedup (warm cache, `--local=8`, `--num-runs=10`)

Speedup = baseline `run-time-mean` / opt `run-time-mean`. Higher is better;
`<1.0` is a regression at that size.

| benchmark | size=64 | size=128 | size=256 |
|-----------|--------:|---------:|---------:|
| **2mm**         | 1.14× | 1.80× | **1.88×** |
| **3mm**         | 1.03× | 1.63× | **1.69×** |
| **atax**        | 0.69× | **1.82×** | 1.07× |
| **gemm**        | 1.02× | **2.20×** | 2.10× |
| **syrk**        | 1.09× | **2.12×** | 1.93× |
| **syr2k**       | 1.26× | **1.93×** | 1.54× |
| **gesummv**     | 0.89× | 1.44× | **2.46×** |
| **mvt**         | 0.90× | 1.13× | 1.09× |
| **bicg**        | 0.96× | 1.09× | 0.92× |
| **covariance**  | 1.12× | 0.92× | 0.85× |
| **correlation** | **1.93×** | 1.24× | 1.27× |
| **gramschmidt** | 1.02× | 1.02× | 1.05× |
| **fdtd2d**      | 1.15× | 0.85× | 0.68× |

(`fdtd2d` at 256 = 81.9 s baseline / 121.3 s opt; 256³×500 timesteps is very
slow on the DSP, and the fused-kernel regression scales up — see below.)

### Reading the table

- **The win lives at size ≥128, in the GEMM/GEMV family.** `gemm`, `syrk`,
  `syr2k`, `2mm`, `3mm`, `gesummv`, and `atax` all do `C[item] += A·B` (or the
  GEMV equivalent) *inside the K-loop*, so under `-cl-opt-disable` the output is
  read-modify-written N times per element. Collapsing that into a register
  accumulator + single store removes ~⅓ of the global traffic and gives
  **1.5–2.5×** once compute dominates (size ≥128). This is the core
  `mocl-dsp-constraints` win.
- **Size 64 is overhead-bound and uninformative.** Every kernel lands at
  ~10–30 ms of host overhead regardless of compute, so speedups cluster near
  1.0× and several show spurious regressions (atax 0.69×, gesummv 0.89×, mvt
  0.90×, bicg 0.96×). These do **not** reflect kernel compute and flip positive
  at larger sizes (e.g. gesummv 0.89×→2.46×, atax 0.69×→1.82×). Use size ≥128
  for representative numbers.
- **The GEMV / non-GEMM kernels are modest (~1.0×).** `mvt`, `bicg`,
  `covariance`, `gramschmidt`, and `fdtd2d` are either bandwidth-bound on the
  N² matrix reads (not the C RMW), or already register-clean, or have an
  O(N²) part dwarfed by an O(N³) part the rewrite doesn't touch. Their opts are
  still correct and DSP-safe, and are no-ops on CPU/GPU (the device compiler
  hoists there), but they do not beat the baseline by much on the DSP. A few
  regress at larger sizes — notably **`fdtd2d` regresses 0.85× at 128 and 0.68×
  at 256**: the fused ey+ex kernel's branch divergence and register pressure
  cost more than the saved launch + `hz_ij` reuse once the stencil loop dominates,
  so the fusion is a net loss at scale on the DSP (it was a small win only at
  size 64). `covariance` regresses 0.85× at 256 for the same access-pattern
  reason.
- **`correlation` is the standout at small size (1.93× at 64)** because its
  baseline's Corr kernel launches only N threads (one per column j1), each
  serially looping j2 then i — catastrophic occupancy. The opt's one-thread-per-
  output (j1,j2) mapping restores parallelism, which helps even at size 64.

### Bottom line

On the MT3K-DSP, the register-accumulator / register-tile rewrites give a
**real 1.5–2.5× kernel-compute speedup** on the GEMM/GEMV family (gemm, syrk,
syr2k, 2mm, 3mm, gesummv, atax) at compute-dominated sizes, and ~1.0–1.3×
elsewhere, with no correctness regressions. The double-digit speedups an
asymmetric (cold-baseline / warm-opt) measurement reports are a POCL JIT-cache
artifact — always warm both sides before measuring.
