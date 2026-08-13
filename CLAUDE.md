# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SYCL-Bench: a cross-platform SYCL benchmark suite. Each benchmark is a standalone executable; build picks one SYCL implementation per configuration (set via `-DSYCL_IMPL=...`).

## Build

CMake out-of-source build. The required flag is `-DSYCL_IMPL`, with one of: `ComputeCpp`, `hipSYCL`, `LLVM` (Intel DPC++), `LLVM-MLIR-HIP`, `LLVM-HIP`, `LLVM-CUDA`, `triSYCL`. The chosen implementation drives both `find_package` and the compiler flags emitted in `CMakeLists.txt`.

A working local script is `build/build.sh` — it loads the user's ROCm/DTK modules, sets `LD_LIBRARY_PATH` for a custom LLVM/SYCL install, then runs:

```
cmake -DSYCL_IMPL=LLVM-HIP \
      -DCMAKE_CXX_COMPILER=/public/home/liuying/sycl-mlir/build/install/bin/clang++ \
      -DCMAKE_INSTALL_PREFIX=/public/home/liuying/sycl-bench ..
cmake --build . -j16
make install
```

`make install` lays binaries out under `<prefix>/bin/benchmarks/` — that layout is what `bin/run-suite` expects, so install before invoking the runner.

When adding a new benchmark `.cpp`, append it to the `benchmarks` list in `CMakeLists.txt`. Files in `single-kernel/`, `pattern/`, `runtime/`, `micro/`, etc. that aren't listed there will not be built (e.g. `pattern/scan.cpp`, `runtime/ranges.cpp` exist but are intentionally excluded).

## Running

Single benchmark — every executable accepts the same flags (defined in `include/command_line.h`):

```
./vec_add --device=gpu --size=1048576 --local=256 --num-runs=10 --output=out.csv
```

Common flags: `--size`, `--local`, `--num-runs`, `--device={cpu|gpu|default}`, `--output={stdio|<file.csv>}`, `--no-verification`, `--no-ndrange-kernels`, `--warmup-run`, `--verification-begin=x,y,z`, `--verification-range=x,y,z`.

Full suite — `bin/run-suite <profile>` (run from inside the install tree, since it discovers binaries via `dirname($0)/benchmarks`). Profiles defined in that script: `default`, `quicktest`, `cpu`, `cpu-warmup`, `gpu`, `gpu-warmup`, `cpu-noverify`, `cpu-nondrange`, `cpu-noverify-nondrange`, `gpu-noverify`. The runner refuses to start if `./sycl-bench.csv` already exists in the cwd, so delete or move the previous output first. Per-benchmark size overrides (e.g. polybench, reductions need bigger inputs than the 1024 default) live in `default_profile['individual-benchmark-options']`.

## Architecture

The harness is header-only (`include/`) and is what unifies all benchmarks; understanding it is the fastest path to making a change in any benchmark.

- `include/common.h` defines `BenchmarkApp` and `BenchmarkManager<Benchmark>`. Each benchmark `.cpp` writes a class with `setup()`, `run(...)`, optional `verify(VerificationSetting&)`, and a static `getBenchmarkName(args)`, then calls `app.run<MyBench<T>>()` from `main`. A typical file (e.g. `single-kernel/vec_add.cpp`) calls `app.run` once per type (int / long long / float / double, gating fp64 on `app.deviceSupportsFP64()`).
- `BenchmarkManager::run` constructs the benchmark fresh per iteration, calls `setup()` then `run()`, and times the wall clock between `preKernel`/`postKernel` hooks. If the benchmark's `run` takes `std::vector<cl::sycl::event>&`, queue profiling can also report aggregate kernel-time — but this is only emitted when `SYCL_BENCH_ENABLE_QUEUE_PROFILING` is defined (currently only the ComputeCpp + ptx64 path sets it). Otherwise kernel-time is reported as `N/A`.
- `include/benchmark_traits.h` uses SFINAE to detect whether a benchmark exposes `run(events)`, `verify`, or `getThroughputMetric`. The harness branches on these traits — adding/removing those methods silently changes what gets measured and reported.
- `include/prefetched_buffer.h` wraps `cl::sycl::buffer` so initial host-to-device transfer can be triggered eagerly via a dummy kernel; benchmarks call `buf.initialize(queue, ptr, range)` in `setup()` so the H2D copy isn't accidentally counted in the measured `run()` time.
- `include/result_consumer.h` (constructed by `BenchmarkCommandLine` in `command_line.h`) writes to either stdio or csv depending on `--output`. CSV output is shared across multiple benchmarks invoked from the same process — the consumer batches a row per `app.run<...>()` call.
- `include/benchmark_hook.h` is the extension point (e.g. `nv_energy_meas.h`, gated by `-DNV_ENERGY_MEAS`).

Benchmark categories:
- `micro/` — synthetic microbenchmarks (arith, DRAM, L2, local memory, special functions, host↔device bandwidth).
- `single-kernel/` — single-kernel apps (vec_add, sobel, kmeans, nbody, etc.).
- `pattern/` — parallel patterns (reduction, segmented reduction). `prefixsum`, `scan`, `segmentedscan` exist but are not in the build list.
- `runtime/` — multi-kernel / DAG-stress benchmarks for SYCL runtime overhead (`dag_task_throughput_*`, `blocked_transform`, `matmulchain`).
- `polybench/` — Polybench/GPU ports. They include `polybench/common/polybenchUtilFuncts.h` (added to the include path globally in `CMakeLists.txt`).
- `compiletime/` — compile-time evaluation harness; `compiletime.cpp` is intentionally excluded from the default build (the Ruby scripts drive a separate experiment).

## SYCL implementation conventions

- Code uses the legacy `<CL/sycl.hpp>` header and the `cl::sycl` namespace. The comment in `vec_add.cpp` is repo-wide policy: do not `using namespace cl::sycl;` because hipSYCL breaks under it; alias as `namespace s = cl::sycl;` instead.
- `__LLVM_SYCL__`, `__LLVM_SYCL_CUDA__`, `__TRISYCL__` are defined per-implementation by `CMakeLists.txt`. `__HIPSYCL__` and `__COMPUTECPP__` come from those toolchains. `getSyclImplementation()` in `common.h` is the canonical mapping.
- C++17 is required and `-std=c++17` is forced for hipSYCL via `CMAKE_SYCL_FLAGS`.
