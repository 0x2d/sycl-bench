# `polybench/sunway` optimized kernels — tunable tiling

Optimized Polybench kernels for the Sunway target, built when
`-DARCH_OPT=sunway -DSYCL_IMPL=LLVM-Sunway` (the `swsycl` driver lowering to
the `athread` offload target; see `CMakeLists.txt`). Each `<name>_opt.cpp` is a
drop-in alternative to `polybench/<name>.cpp` with a distinct benchmark name
(`Polybench_<Name>_Opt_...`), so the opt and the reference kernel coexist in
the build and can be A/B-compared on the same queue.

The kernels are ported from `polybench/orise/` (DCU/gfx906) with the tile
shapes reduced for the Sunway core group and, crucially, **all tile sizes made
run-time selectable via environment variables** — this machine only compiles
(the binaries run on the Sunway cluster), so re-tuning a tile must not require
a rebuild. Each kernel is a template instantiated once per selectable shape
(distinct kernel-name type per shape — required by the LLVM-MLIR backend), so
every loop bound stays a compile-time constant regardless of the shape chosen
at run time.

The suite is run at `--size=512 --local=128` (see `bin/run-suite-sw.sh`);
512 is divisible by every selectable tile, so edge-tile paths are inactive.

## Tiling knobs

Shared-memory tiled kernels (one work-group per TS×TS output tile, TS-deep
K-slices staged in local memory; work-group = TS×TS threads). Env var
selects TS ∈ {4, 8, 16, 32}:

| Benchmark    | Env var               | Default | Work-group size |
|--------------|-----------------------|---------|-----------------|
| `2mm_opt`    | `SYCL_2MM_TS`         | 8       | 64 threads      |
| `3mm_opt`    | `SYCL_3MM_TS`         | 8       | 64 threads      |
| `gemm_opt`   | `SYCL_GEMM_TS`        | 8       | 64 threads      |
| `covariance_opt`    | `SYCL_COVARIANCE_TS`    | 8 | 64 threads |
| `correlation_opt`   | `SYCL_CORRELATION_TS`   | 8 | 64 threads |

Register-tiled kernels (each work-item computes a BM×BN output block in
register accumulators; no local memory, no barriers, no `nd_range`). Env vars
select BM, BN ∈ {1, 2, 4, 8} (square shapes are instantiated; invalid values
fall back to the default):

| Benchmark  | Env vars                     | Default |
|------------|------------------------------|---------|
| `syrk_opt` | `SYCL_SYRK_BM` / `SYCL_SYRK_BN`   | 8×8 |
| `syr2k_opt`| `SYCL_SYR2K_BM` / `SYCL_SYR2K_BN` | 8×8 |

The reduction-based kernels (`atax`, `bicg`, `mvt`, `gesummv`,
`gramschmidt`) take their work-group size from the harness `--local` flag
(128 in the suite; their tree reductions need a power of two). `fdtd2d` has
no tiling (a fused-kernel rewrite only).

## Choosing TS / BM / BN on the target

The default TS=8 gives a 64-thread work-group (one work-item per CPE of a
64-CPE core group). If the backend instead vectorizes consecutive work-items
per CPE, larger TS (16/32) amortizes the two K-loop barriers over 2–4× more
FMA work and cuts the barrier count in half per K-step; if the barrier/local
-memory path is slow, or 1024-thread work-groups (TS=32) exceed the backend's
limit, drop to 4. Sweep without rebuilding, e.g. inside the `bsub` job:

```sh
for ts in 4 8 16 32; do
    SYCL_GEMM_TS=$ts ./benchmarks/gemm_opt --size=512 --local=128 --num-runs=10 --output=sycl-bench.csv
done
```

The selected tile is appended to the benchmark name written to the CSV
(`Polybench_Gemm_Opt_TS8`, `Polybench_Syrk_Opt_BM8BN8`, …), so swept runs in
one CSV stay distinguishable.

## Notes

- The K-loop tail is bounded (`kbnd`) in the tiled kernels, so clamped edge
  loads never contribute to the accumulator — any size/tile combination is
  correct, not just multiples of the tile.
- `gemm_opt` / `covariance_opt` pack their two local tiles into a single
  local accessor (structure kept from the orise twin, where two local
  accessors mis-compiled); `2mm`/`3mm`/`correlation` use two accessors.
- BM/BN live as template parameters (not function-local constexpr): the
  swsycl host compiler (swg++/GCC 11.2) implicitly captures function-local
  constexpr into the kernel lambda, growing the closure past what the device
  compiler records ("Unexpected kernel lambda size" static_assert in
  `handler.hpp`); template parameters are never captured.
