# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SYCL-Bench is a benchmark suite for heterogeneous computing using the SYCL standard. It evaluates performance across different SYCL implementations (ComputeCpp, hipSYCL, LLVM/DPC++, triSYCL).

## Build Commands

```bash
mkdir build && cd build
cmake -DSYCL_IMPL=<impl> -DCMAKE_CXX_COMPILER=<compiler> ..
cmake --build .
```

Where `<impl>` is one of: `ComputeCpp`, `hipSYCL`, `LLVM`, `LLVM-MLIR`, `LLVM-CUDA`, `triSYCL`.

For packaging: `cmake -Bbuild -DCPACK_GENERATOR="TGZ;ZIP" && cmake --build build --target package`

## Running Benchmarks

Run a single benchmark:
```bash
./build/<benchmark_name> --device=gpu --size=3072 --local=256
```

Run full suite via Python script:
```bash
./bin/run-suite <profile>
```

Profiles: `default`, `quicktest`, `cpu`, `gpu`, `cpu-warmup`, `gpu-warmup`, `cpu-noverify`, etc.

## Key Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--size=<N>` | Problem size (global range) | 3072 |
| `--local=<N>` | Work group size | 256 |
| `--num-runs=<N>` | Number of runs for averaging | 5 |
| `--device=<d>` | Device selector (`cpu`, `gpu`, `default`) | `default` |
| `--output=<f>` | Output file or `stdio` | `stdio` |
| `--no-verification` | Disable verification | - |
| `--warmup-run` | Run once before timing | - |

## Architecture

**Benchmark directories:**
- `single-kernel/` — Standard single-kernel benchmarks (vec_add, sobel, median, nbody, kmeans, etc.)
- `micro/` — Microbenchmarks (arith, DRAM, host_device_bandwidth, local_mem, pattern_L2, sf)
- `pattern/` — Pattern benchmarks (reduction, segmentedreduction)
- `runtime/` — Runtime benchmarks (blocked_transform, matmulchain, dag_task_throughput)
- `polybench/` — PolyBench numerical algorithms (gemm, 2DConvolution, atax, bicg, covariance, fdtd2d, etc.)
- `include/` — Core headers

**Benchmark interface** (all benchmarks follow this pattern):
```cpp
template <typename DataType> class BenchmarkName {
public:
    BenchmarkName(BenchmarkArgs& args);
    void setup();
    void run();
    void verify() { } // optional
    static std::string getBenchmarkName();
};
```

**Core classes** (`include/common.h`):
- `BenchmarkApp` — Entry point, handles CLI and device selection
- `BenchmarkManager<Benchmark>` — Runs benchmark, collects timing metrics
- `PrefetchedBuffer<T, D>` — SYCL buffer wrapper with forced data transfer

**Namespace convention:** Use `namespace s = cl::sycl;` (not `using namespace cl::sycl;`) for cross-SYCL-compatibility with hipSYCL/CUDA/HIP.

## File Structure

- `CMakeLists.txt` — Main build config (lists all benchmarks)
- `cmake/` — CMake modules for different SYCL implementations
- `bin/run-suite` — Python script to run the full benchmark suite
- `.clang-format` — Code formatting (llvm style, 120 col)