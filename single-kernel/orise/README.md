# `single-kernel/orise` optimized kernels — speedup vs. baseline

Measured on the DCU GPU ("Device 66a1", gfx906) via the same toolchain as
`polybench/orise`: `SYCL_IMPL=LLVM-HIP`, `ARCH_OPT=orise`, clang++ from
`/public/home/liuying/sycl-mlir/build/install`, built in the `build/` tree.
Sizes follow `bin/run-suite`'s `default_profile['individual-benchmark-options']`
(`kmeans`/`lin_reg_coeff` → 2^20, `lin_reg_error` → 2^16), `--local=256
--num-runs=10`, `ONEAPI_DEVICE_SELECTOR=hip:*`. Each pair was measured in 3
interleaved rounds (baseline and opt as separate fresh processes alternating
in the same job); the table reports the median of the three per-round
medians. All measurements are wall-clock run-time around `run()` (H2D is
triggered eagerly in `setup()` via `prefetched_buffer` and excluded;
kernel-time is `N/A` — queue profiling is not enabled on this path).

## Results (final, fresh-process A/B)

All runs **PASS verification** (0 FAIL). `lin_reg_*` rows: 3 interleaved
rounds × num-runs=10 medians. `kmeans` rows: matched fresh-process
**num-runs=50** probes (the median lands in the device's fast state — see
the methodology note below; 6 matched base/opt pairs across 3 jobs).

| Benchmark | size | opt median [s] | base median [s] | **Speedup (median)** | Verify |
|---|--:|--:|--:|--:|:--:|
| `lin_reg_coeff` fp32 | 2^20 | 0.000878 | 0.004077 | **4.64×** | PASS |
| `lin_reg_coeff` fp64 | 2^20 | 0.000869 | 0.004613 | **5.31×** | PASS |
| `lin_reg_error` fp32 | 2^16 | 0.003615 | 0.003920 | **1.08×** (see below — fp32 is verify-pinned) | PASS |
| `lin_reg_error` fp64 | 2^16 | 0.000327 | 0.007410 | **22.7×** | PASS |
| `kmeans` fp32 | 2^20 | 0.000135 | 0.000166 | **1.23×** (fast-floor 0.096 vs 0.130, 1.35×) | PASS |
| `kmeans` fp64 | 2^20 | 0.000130 | 0.000142 | **1.09×** (fast-floor 0.110 vs 0.130, 1.19×) | PASS |

## Optimization per kernel

- **`lin_reg_coeff_opt` (4.64× / 5.31×)** — the baseline `run()` executes
  **14 kernel launches and 4 full-buffer host synchronizations** per measured
  run: 2 elementwise product kernels (`vec_product`) plus 4 multi-pass tree
  reductions (`reduce()` ×4 for ss_xy, ss_xx, sum_x, sum_y), and *every*
  `reduce()` call ends in a host accessor over the whole input buffer (a
  4 MB D2H per call) to read element [0]. The opt fuses all of that into
  **one multi-sum kernel + one small finalize kernel + one 32-byte read**:
  each work-group of 256 threads strided-covers a 2048-element block,
  accumulating FOUR partial sums per element in registers (`x*y`, `x*x`,
  `x`, `y`) — the two input streams are read once instead of being re-read
  by 2 product kernels and 4 reduce passes — then tree-reduces the four
  partials in a single packed local accessor (region `s` at
  `[s*WG, s*WG+WG)`, one barrier per tree level advances all four regions).
  A second 256-thread kernel reduces the four per-group partial rows to
  four grand totals, and the host derives the coefficients from the
  32-byte result with the baseline's exact expression order. Numerics are
  exact for the fixture (all-1.0/2.0 data: every partial sum is a power of
  two ≤ 2^21, so any reduction order is bit-identical to verify()'s
  sequential host sums — the 1e-5 coefficient comparison is exact).
- **`lin_reg_error_opt` (fp64 22.7×, fp32 1.08×)** — the baseline is O(N²):
  each of N work-items serially accumulates its (alpha,beta) pair's error
  over all N points. The opt is **type-conditional** because the verify
  contract splits the legal optimization space by precision:
    - **fp64 — algebraic O(N) decomposition (22.7×)**: expand
      `err[i] = Σ_j (a_i·x_j + b_i − y_j)²` to
      `a²Sxx + 2abSx + b²N − 2aSxy − 2bSy + Syy` — five sums (Sxx, Sx,
      Sxy, Sy, Syy) computed ONCE by a fused multi-sum kernel
      (lin_reg_coeff_opt's structure with five partials per element, EPT=8
      strided, packed 5·WG local accessor, tree reduction), one 256-thread
      finalize kernel, then one work-item per output evaluating 10
      FMAs from the five sums. O(N²)→O(N): 7.41ms → 0.33ms. Why it
      verifies at fp64 but not fp32: verify() compares against a host
      loop summed sequentially in T with a 1e-6 relative-L2 norm. At fp64
      the host reference is accurate to ~1e-11 relative, so ANY
      value-correct computation (including the identity, whose
      cancellation error is also ~1e-11 at fp64) agrees within tolerance.
      At fp32 the host reference itself carries ~1.5e-5 relative
      sequential-summation error — a MORE accurate device value (exact
      fp64, reassociated, or the identity) differs from the host by that
      ~1.5e-5 and FAILs by ~10×; only replicating the host's rounding
      order passes. MEASURED at fp32: identity 1.11ms FAIL; 2-way
      block-split 3.60ms FAIL; FMA passes but slower (3.40 vs 3.38ms);
      baseline-order unroll passes.
    - **fp32 — same-order staged-register unroll, U=32 (1.08×)**: keeps
      the baseline mapping (one work-item per output, full N work-items —
      the latency-hiding mechanism on this DCU) and unrolls the i-loop
      with register-staged loads; FP chain bit-identical to the baseline
      per output. In-process sweeps suggested U=16→32 goes 3.38→3.11ms;
      fresh-process they are indistinguishable (~3.61ms both) — another
      in-process position artifact (see methodology note). ~2.4 T
      uniform L2 loads/s ≈ 9 TB/s effective L2 bandwidth is the fp32
      ceiling for this pinned-order kernel.
    - Other measured fp32 levers (all rejected): LDS staging of in1/in2
      chunks (barrier-bound, 3.84–3.85ms; wins only at fp64: 7.15ms);
      register blocking over outputs (G=2 1.6× worse, G=8 ~3× worse —
      work-item count is the latency-hiding mechanism); explicit
      nd_range shape (slightly worse than plain range).
- **`kmeans_opt` (fp32 1.23× / fp64 1.09×, matched num-runs=50 probes)** —
  three levers, measured separately (variant ladder + position-controlled
  6-round A/B, 25s cooldowns):
    - **Dead-transfer elimination in setup() (THE original win, ~1.2×)**:
      the baseline H2Ds 24 MB per harness iteration but the kernel only ever
      addresses `clusters[i*nfeatures+l]` for `i<nclusters, l<nfeatures`
      — 6 values, 24 B; the other ~12 MB (upstream's oversized
      `cluster_size = nclusters*problem_size` buffer) is never read by
      kernel or verify. And `membership` is `discard_write`'n in full
      before any read, so its 4 MB H2D of zeros is dead too. The opt
      transfers features + a 6-element clusters buffer and allocates
      membership without transfer: 8 MB + 24 B instead of 24 MB per
      iteration — data-independent (holds for any
      nfeatures/nclusters/data), outputs identical. Although setup() is
      outside the timed window, the dead transfers' L2 writeback contends
      with the timed kernel.
    - **Wide vector accesses (v4, +~8% medians, +~4% fast-floor)**: each
      work-item owns 4 consecutive points and touches them with ONE 16B
      (fp32, 4×float) / 32B (fp64, 2×double2-equivalent) load per feature
      stream plus one 16B int4 store, instead of 8 scalar loads + 4 scalar
      stores. The clue was in the old samples: the fp64 kernel moved 20MB
      in LESS time than fp32 moved 12MB — scalar 8B accesses got ~75%
      higher effective bandwidth than 4B ones, i.e. fp32 was limited by
      memory-instruction/sector throughput, not DRAM bytes. Vectorization
      attacks exactly that. Semantics preserved via the `index = gid`
      quirk: the output only depends on whether any cluster distance is
      `< 500000`, computed per 4-lane vector as an OR of the three
      per-cluster predicates — each with the baseline's exact distance
      expression — so output is identical for any data. Requires
      `N % 4 == 0` (suite sizes are powers of two); other sizes take a
      scalar fallback kernel (verified PASS at N=1027, N=131).
    - Kernel-level (kept): 3×2 distance math unrolled for the fixed
      C=3/F=2. With the vector path the fast-state floor is
      12MB @ ~127GB/s (fp32), 20MB @ ~190GB/s (fp64) — matching the
      vec_add streaming reference (platform ceiling for this R/W mix).
    - REJECTED (all measured position-controlled): explicit nd_range
      launch shape (slightly worse than plain `range`); short-circuit
      centroid evaluation — skip clusters 1/2 once cluster 0 decided the
      predicate — ~neutral-to-worse, the kernel is memory-bound and the
      skip branch costs what the skipped math saves; **centroid capture**
      (the 6 centroid values as by-value kernel args instead of the 24B
      device buffer) — **~25-30% WORSE**: on this LLVM-HIP/DCU stack
      by-value captures lower to slower code than L2-cached uniform
      buffer loads; B=8 blocking — ~15% worse (wave count is the
      latency-hiding mechanism, same lesson as lin_reg_error).
    - The upstream `index = gid` argmin quirk is kept VERBATIM per point
      (kernel and verify) — output bit-identical for any data.

## Methodology note (important for sub-ms kernels on this DCU)

Two measurement traps, both discovered while optimizing kmeans:

1. **The DCU has a fast/slow state machine (~0.5 ms per launch apart).**
   The per-run samples (`run-time-samples`) show the device runs a sub-ms
   kernel at its true cost for the first ~30–40 back-to-back runs of a
   fresh process, then transitions to a ~+0.5 ms fixed slow state (e.g.
   kmeans fp32: `0.121 0.122 … 0.188 | 0.606 0.611 …`), recovering over
   tens of seconds. The default num-runs=10 window lands *entirely* in
   whichever state the device is in at job start — a 10-run median can
   read 0.62 ms or 0.16 ms for the same binary. **num-runs=50 medians
   (which mix ~2/3 fast samples and land in the fast region) are the
   robust figure of merit for sub-ms kernels; always record the samples,
   not just the median.** The ~0.5 ms slow-state penalty is per-launch
   and hits baseline and opt equally (10-run rounds: base 0.63 vs opt
   0.60 — an artifact, not a speedup).
2. **In-process multi-variant sweeps mislead absolutely.** Measuring
   several variants in one process (one `app.run` each) makes later
   variants read up to ~4× fast on sub-ms kernels (state + warmup
   interaction). Use them only for *ranking*; validate winners with
   fresh-process interleaved A/B (one binary per process, alternating
   rounds) — which is what `bin/run-suite` does anyway.

## Reproduce

```
cd /public/home/liuying/sycl-bench/build
source <toolchain env from build/build.sh>   # dtk module + LD_LIBRARY_PATH + SYCL_AMDGCN_*
cmake -DSYCL_IMPL=LLVM-HIP \
      -DCMAKE_CXX_COMPILER=/public/home/liuying/sycl-mlir/build/install/bin/clang++ \
      -DCMAKE_INSTALL_PREFIX=/public/home/liuying/sycl-bench ..
cmake --build . --target kmeans kmeans_opt lin_reg_coeff lin_reg_coeff_opt \
                        lin_reg_error lin_reg_error_opt -j16
sbatch run-skopt-meas.sh       # lin_reg_coeff pairs: 3 interleaved rounds, num-runs=10
sbatch run-kmeans-final.sh     # kmeans pair: 3 matched num-runs=50 probes + 3x10-run rounds
sbatch run-lrerr-final.sh      # lin_reg_error pair: 3 interleaved rounds, num-runs=10
```

`run-skopt-meas.sh` / `run-kmeans-final.sh` / `run-lrerr-final.sh` print
per-round `run-time-median` (and samples) plus `Verification: PASS|FAIL`;
slurm output lands in `kmeans-final-<jobid>.out` / `lrerr-final-<jobid>.out`.
The kmeans v4 (vector-access) rework was validated with
`run-kmeans-fin2.sh` and `run-kmeans-fin3.sh` (6 position-controlled
rounds + matched probes + N%4!=0 fallback probes; outputs
`kmeans-fin2-<jobid>.out` / `kmeans-fin3-<jobid>.out`).
