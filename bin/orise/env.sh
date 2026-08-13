# orise platform environment for SYCL-Bench (AMD GPU / gfx906 via DPC++ + sycl-mlir).
#
# Source this before building and running:
#   source bin/orise/env.sh
#
# It loads the DTK/ROCm module stack, points the toolchain at the custom
# sycl-mlir LLVM install, and exports the AMDGCN codegen tools that the
# sycl-mlir backend invokes during device compilation.

# Drop any stale module stacks that conflict with the DTK/ROCm toolchain.
module unload compiler/devtoolset/7.3.1
module unload mpi/hpcx/2.11.0/gcc-7.3.1
module unload compiler/rocm/dtk/22.10.1
module load compiler/rocm/dtk/25.04.2

# gcc 11.2.0 toolchain (matches --gcc-toolchain in CMakeLists LLVM-HIP/LLVM-MLIR-HIP).
source ~/Tools/setgcc.sh

# Custom sycl-mlir LLVM/SYCL install (provides clang++ and runtime libs).
export SYCL_MLIR_ROOT=/public/home/liuying/sycl-mlir/build/install
export PATH=${SYCL_MLIR_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${SYCL_MLIR_ROOT}/lib:$LD_LIBRARY_PATH

# AMDGCN codegen tools the sycl-mlir backend shells out to (DTK 25.04.2).
export SYCL_AMDGCN_OPT=/public/software/compiler/dtk/dtk-25.04.2/dcc/bin/opt
export SYCL_AMDGCN_OPT_FLAGS=-O2
export SYCL_AMDGCN_LLC=/public/software/compiler/dtk/dtk-25.04.2/dcc/bin/llc

# Force the oneAPI runtime onto the HIP (AMD GPU) backend for benchmark runs.
export ONEAPI_DEVICE_SELECTOR='hip:*'
