cmake_minimum_required (VERSION 3.5)
project(sycl-bench)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake Build Type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Due to CMake limitations, hipSYCL requires C++ standard to be set manually
set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -std=c++17")

if(CMAKE_GENERATOR STREQUAL "Ninja")
  set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -fdiagnostics-color=always")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
  set(COMPUTECPP_USER_FLAGS "${COMPUTECPP_USER_FLAGS} -fdiagnostics-color=always")
endif()

include(InstallRequiredSystemLibraries)
include(CPack)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/polybench/common)

set(supported_implementations
  ComputeCpp
  hipSYCL
  LLVM
  LLVM-MLIR
  LLVM-MLIR-HIP
  LLVM-HIP
  LLVM-CUDA
  LLVM-Sunway
  triSYCL
)

list(FIND supported_implementations ${SYCL_IMPL} impl_idx)
if(NOT SYCL_IMPL OR impl_idx EQUAL -1)
  message(FATAL_ERROR "Please specify SYCL_IMPL (one of: ${supported_implementations})")
endif()

if(SYCL_IMPL STREQUAL "ComputeCpp")
  find_package(ComputeCpp MODULE REQUIRED)
elseif(SYCL_IMPL STREQUAL "hipSYCL")
  find_package(hipSYCL CONFIG REQUIRED)
elseif(SYCL_IMPL STREQUAL "LLVM")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
elseif(SYCL_IMPL STREQUAL "LLVM-MLIR")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir")
elseif(SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=/public/home/liuying/Tools/gcc/11.2.0 -fsycl -fsycl-targets=amdgcn-amd-amdhsa-syclmlir -Xsycl-target-backend --offload-arch=gfx906")
elseif(SYCL_IMPL STREQUAL "LLVM-HIP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=/public/home/liuying/Tools/gcc/11.2.0 -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906")
elseif(SYCL_IMPL STREQUAL "LLVM-CUDA")
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice")
elseif(SYCL_IMPL STREQUAL "LLVM-Sunway")
  # Sunway (SW) target: DPC++/sycl-mlir lowering to the athread offload target.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl --target=athread")
elseif(SYCL_IMPL STREQUAL "triSYCL")
  find_package(TriSYCL MODULE REQUIRED)
endif()

set(ARCH_OPT "" CACHE STRING "Architecture-specific optimized kernels to build (e.g. tianhe, orise). When set, the matching polybench/<ARCH_OPT>/*_opt.cpp and single-kernel/<ARCH_OPT>/*_opt.cpp kernels are compiled in place of the reference kernels.")
set_property(CACHE ARCH_OPT PROPERTY STRINGS "" tianhe orise sunway)

# CovarianceCovar: register-blocked loop reorder (hoist column-j1 load).
# LLVM-MLIR-HIP only; on by default.
option(COVARIANCE_LOOP_REORDER "Enable the sycl-covariance-loop-reorder cgeist pass on the covariance target (LLVM-MLIR-HIP only)" ON)

# Syr2k2: register accumulator (single C read/write, scf.while->scf.for, hoist i/j/M).
# LLVM-MLIR-HIP only; on by default.
option(SYRK_REGISTER_ACCUMULATOR "Enable the sycl-syrk-register-accumulator cgeist pass on the syrk target (LLVM-MLIR-HIP only)" ON)

# Syr2k2: paired device+host passes — --sycl-syrk-register-tile (8x8 register tile
# at (item_id*8, item_id*8)) and -fsycl-rewrite-syrk-range (range<2>(N,N) ->
# range<2>(ceil(N/8), ceil(N/8))). Reproduces polybench/orise/syrk_opt.cpp (~1.46x
# on gfx906) without touching benchmark source. LLVM-MLIR-HIP only; on by default.
option(SYRK_LAUNCH_TILE "Enable the paired sycl-syrk-register-tile (device) + sycl-rewrite-syrk-range (host launch shrink) passes on the syrk target (LLVM-MLIR-HIP only)" ON)

# Syr2k1: paired device+host passes — --sycl-syr2k-register-tile (8x8 register tile
# at (item_id*8, item_id*8), 16 loads per k feeding 2 FMAs per cell for the A*B^T
# and B*A^T terms) and -fsycl-rewrite-syr2k-range (range<2>(N,N) ->
# range<2>(ceil(N/8), ceil(N/8))). Reproduces polybench/orise/syr2k_opt.cpp without
# touching benchmark source. LLVM-MLIR-HIP only; on by default.
option(SYR2K_LAUNCH_TILE "Enable the paired sycl-syr2k-register-tile (device) + sycl-rewrite-syr2k-range (host launch shrink) passes on the syr2k target (LLVM-MLIR-HIP only)" ON)

# 2mm: paired device+host+runtime passes — --sycl-2mm-local-tile (2mm_opt's
# LDS tiling: 16x16 work-groups cooperatively stage 16x16 A/B tiles in local
# memory with rocdl.barrier sync; one output cell + scalar accumulator per
# work-item, for BOTH GEMM kernels; kernel 1 seeded from C, kernel 2 seeded
# from zero) and -fsycl-rewrite-2mm-range (range<2>(N,N) ->
# range<2>(ceil16(N), ceil16(N)) on BOTH launches). The (16,16) work-group
# shape is supplied at RUN time by SYCL_FORCE_LOCAL_SIZE=16,16 (this tree's
# runtime; set it when running the tiled binary, leave it unset for A/B
# baselines). Reproduces polybench/orise/2mm_opt.cpp without touching
# benchmark source. LLVM-MLIR-HIP only; on by default.
option(TWO_MM_LAUNCH_TILE "Enable the paired sycl-2mm-local-tile (device) + sycl-rewrite-2mm-range (host launch ceil16 padding) passes on the 2mm target (LLVM-MLIR-HIP only; run with SYCL_FORCE_LOCAL_SIZE=16,16)" ON)

# 3mm: paired device+host+runtime passes -- --sycl-3mm-local-tile (3mm_opt's
# LDS tiling: 16x16 work-groups cooperatively stage 16x16 A/B tiles in local
# memory with rocdl.barrier sync; one output cell + scalar accumulator per
# work-item, for ALL THREE GEMM kernels, each seeded from the existing
# read_write output) and -fsycl-rewrite-3mm-range (range<2>(N,N) ->
# range<2>(ceil16(N), ceil16(N)) on ALL THREE launches). The (16,16)
# work-group shape is supplied at RUN time by SYCL_FORCE_LOCAL_SIZE=16,16
# (this tree's runtime; set it when running the tiled binary, leave it unset
# for A/B baselines). Reproduces polybench/orise/3mm_opt.cpp without
# touching benchmark source. LLVM-MLIR-HIP only; on by default.
option(THREE_MM_LAUNCH_TILE "Enable the paired sycl-3mm-local-tile (device) + sycl-rewrite-3mm-range (host launch ceil16 padding) passes on the 3mm target (LLVM-MLIR-HIP only; run with SYCL_FORCE_LOCAL_SIZE=16,16)" ON)

# gemm: paired device+host+runtime passes -- --sycl-gemm-local-tile
# (gemm_opt's LDS tiling: 16x16 work-groups cooperatively stage 16x16 A/B
# tiles in local memory with rocdl.barrier sync; one output cell + scalar
# accumulator per work-item, seeded from C*BETA with ALPHA-scaled FMAs) and
# -fsycl-rewrite-gemm-range (range<2>(N,N) ->
# range<2>(ceil16(N),ceil16(N))). The (16,16) work-group shape is supplied
# at RUN time by SYCL_FORCE_LOCAL_SIZE=16,16 (this tree's runtime; set it
# when running the tiled binary, leave it unset for A/B baselines).
# Reproduces polybench/orise/gemm_opt.cpp without touching benchmark
# source. LLVM-MLIR-HIP only; on by default.
option(GEMM_LAUNCH_TILE "Enable the paired sycl-gemm-local-tile (device) + sycl-rewrite-gemm-range (host launch ceil16 padding) passes on the gemm target (LLVM-MLIR-HIP only; run with SYCL_FORCE_LOCAL_SIZE=16,16)" ON)

# atax: paired device+host+runtime passes -- --sycl-atax-wg-reduction
# (atax_opt's per-output work-group reduction: ONE 256-wide work-group per
# row of A, lanes striding the j axis (coalesced row-major reads) with the
# partial sums tree-reduced in local memory via rocdl.barrier; Atax2 is
# left untouched -- its RMW change measured no gain) and
# -fsycl-rewrite-atax-range (Atax1's range<1>(N) -> range<1>(N*256), ONE
# launch). The 256-wide work-group shape is supplied at RUN time by
# SYCL_FORCE_LOCAL_SIZE=256 (this tree's runtime; set it when running the
# rewritten binary, leave it unset for A/B baselines -- without it the
# i = gid/256 math is WRONG). Reproduces
# polybench/orise/atax_opt.cpp (15.2x at N=4096) without touching
# benchmark source. LLVM-MLIR-HIP only; on by default.
option(ATAX_WG_REDUCTION "Enable the paired sycl-atax-wg-reduction (device) + sycl-rewrite-atax-range (host launch N -> N*256 expansion) passes on the atax target (LLVM-MLIR-HIP only; run with SYCL_FORCE_LOCAL_SIZE=256)" ON)

# bicg: paired device+host+runtime passes -- --sycl-bicg-wg-reduction
# (bicg_opt's work-group reduction for Bicg2 only: one 256-wide work-group
# per row of A, lanes striding the j axis (coalesced row-major reads) with
# the partial sums tree-reduced in local memory via rocdl.barrier; Bicg1
# is left untouched -- its reduction axis is A's strided axis, so a
# per-column work-group would wreck its already-coalesced baseline) and
# -fsycl-rewrite-bicg-range (Bicg2's range<1>(N) -> range<1>(N*256), ONE
# launch). The 256-wide work-group shape is supplied at RUN time by
# SYCL_FORCE_LOCAL_SIZE=256 (this tree's runtime; set it when running the
# rewritten binary, leave it unset for A/B baselines -- without it the
# i = gid/256 math is WRONG). Reproduces
# polybench/orise/bicg_opt.cpp (~18x at N=16384) without touching
# benchmark source. LLVM-MLIR-HIP only.
option(BICG_WG_REDUCTION "Enable the paired sycl-bicg-wg-reduction (device) + sycl-rewrite-bicg-range (host launch N -> N*256 expansion) passes on the bicg target (LLVM-MLIR-HIP only; run with SYCL_FORCE_LOCAL_SIZE=256)" ON)

# mvt: paired device+host+runtime passes -- --sycl-mvt-wg-reduction
# (mvt_opt's per-output work-group reduction for BOTH kernels -- Mvt1
# (x1 = A*y1) AND Mvt2 (x2 = A*y2), unlike atax/bicg's single kernel: both
# have the row-contiguous orientation. ONE 256-wide work-group per row of
# A, lanes striding the j/l axis (coalesced row-major reads) with the
# partial sums tree-reduced in local memory via rocdl.barrier; lane 0
# stores the row total through the read_write accessor -- x1/x2 are
# zero-initialized by init_arrays, so discarding the RMW start value is
# exactly mvt_opt's discard_write) and -fsycl-rewrite-mvt-range (BOTH
# range<1>(N) launches -> range<1>(N*256)). The 256-wide work-group shape
# is supplied at RUN time by SYCL_FORCE_LOCAL_SIZE=256 (this tree's
# runtime; set it when running the rewritten binary, leave it unset for
# A/B baselines -- without it the i = gid/256 math is WRONG). Reproduces
# polybench/orise/mvt_opt.cpp without touching benchmark source.
# LLVM-MLIR-HIP only.
option(MVT_WG_REDUCTION "Enable the paired sycl-mvt-wg-reduction (device) + sycl-rewrite-mvt-range (host launch N -> N*256 expansion) passes on the mvt target (LLVM-MLIR-HIP only; run with SYCL_FORCE_LOCAL_SIZE=256)" ON)

# gesummv: paired device+host+runtime passes -- --sycl-gesummv-wg-reduction
# (gesummv_opt's per-output work-group reduction for the fused GEMV pair:
# ONE 256-wide work-group per row of A/B, lanes striding the j axis
# (coalesced row-major reads of BOTH matrices) with the TWO partial sums
# (tmp = A*x, y = B*x) carried as two scalar iter_args of one stride loop
# and tree-reduced in parallel in two LDS scratches via rocdl.barrier;
# lane 0 stores tmp and the y = ALPHA*tmp + BETA*y combine) and
# -fsycl-rewrite-gesummv-range (Gesummv's range<1>(N) -> range<1>(N*256),
# ONE launch). The 256-wide work-group shape is supplied at RUN time by
# SYCL_FORCE_LOCAL_SIZE=256 (this tree's runtime; set it when running the
# rewritten binary, leave it unset for A/B baselines -- without it the
# i = gid/256 math is WRONG). Reproduces
# polybench/orise/gesummv_opt.cpp (56.3x at N=16384) without touching
# benchmark source. LLVM-MLIR-HIP only.
option(GESUMMV_WG_REDUCTION "Enable the paired sycl-gesummv-wg-reduction (device) + sycl-rewrite-gesummv-range (host launch N -> N*256 expansion) passes on the gesummv target (LLVM-MLIR-HIP only; run with SYCL_FORCE_LOCAL_SIZE=256)" ON)


# kmeans: paired device+host passes -- --sycl-kmeans-point-block
# (kmeans_opt's work-item blocking: B=4 points per work-item with the
# strided gid = tid + k*nt mapping, nt read at RUNTIME from
# sycl.num_work_items so the body is immune to launch rounding; the 6
# centroid loads hoisted out of the k loop; the 3x2 distance math fully
# unrolled with the baseline argmin quirk (index = gid) preserved
# verbatim; no barriers, no LDS, no forced local size -- the opt
# measured forced local sizes WORSE),
# -fsycl-rewrite-kmeans-range (Kmeans' range<1>(N) -> range<1>(ceil(N/4)),
# ONE launch, both fp32/fp64 instantiations), and
# -fsycl-rewrite-kmeans-diet (the setup dead-transfer elimination, the
# DOMINANT part of kmeans_opt's ~1.2x win: the clusters buffer's range
# shrunk from 3*N to the 6 ever-addressed elements -- the runtime stages
# 24B instead of ~12MB -- and the membership buffer's dummy-kernel H2D
# dropped -- the kernel discard_write'n it in full before any read).
# Reproduces single-kernel/orise/kmeans_opt.cpp (~1.2x at 2^20) without
# touching benchmark source. No runtime env var needed. LLVM-MLIR-HIP
# only. KMEANS_POINT_BLOCK carries the device+range pair;
# KMEANS_DIET carries the diet alone (for attribution A/Bs).
option(KMEANS_POINT_BLOCK "Enable the paired sycl-kmeans-point-block (device) + sycl-rewrite-kmeans-range (host launch N -> ceil(N/4) shrink) passes on the kmeans target (LLVM-MLIR-HIP only; no runtime env var needed)" ON)
option(KMEANS_DIET "Enable the sycl-rewrite-kmeans-diet (setup dead-transfer elimination) host pass on the kmeans target (LLVM-MLIR-HIP only; composes with KMEANS_POINT_BLOCK)" ON)


# Host readback narrowing for lin_reg_coeff: reduce()'s whole-buffer host
# read accessor (only element [0] ever read) becomes a 1-element access
# range via the sycl-rewrite-linreg-read-range LLVM pass. The runtime's
# range-limited host-read D2H fast path then serves it with a 4/8-byte
# copy instead of the whole buffer. No device pass, no forced local size.
# LLVM-MLIR-HIP only.
option(LIN_REG_COEFF_READ_RANGE "Enable the -fsycl-rewrite-linreg-read-range host LLVM pass on the lin_reg_coeff target (LLVM-MLIR-HIP only; requires the range-limited host-read D2H runtime fast path)" ON)

# lin_reg_error fp32 staged-register unroll: the LinearRegressionKernel<f32>
# body becomes lin_reg_error_opt's U=32 staged unroll (per 32-element block,
# stage the in1/in2 load pairs into SSA registers, then run the 32
# accumulation steps) via the sycl-linreg-fp32-unroll cgeist pass. The FP
# chain stays bit-identical to the baseline (same summation order, no fma)
# -- the fp32 verify tolerance demands it. Device-only: no paired host
# pass, no forced local size, the launch is unchanged. The fp64
# instantiation is excluded by the pass's f32 struct gate. LLVM-MLIR-HIP
# only.
option(LINREG_FP32_UNROLL "Enable the sycl-linreg-fp32-unroll cgeist pass on the lin_reg_error target (LLVM-MLIR-HIP only; device-only, no paired host pass)" ON)

# lin_reg_error fp64 3-kernel algebraic path: the LinearRegressionKernel
# <double> body becomes lin_reg_error_opt's O(N) identity path via the
# sycl-linreg-fp64-3kernel cgeist pass (K1 five-sum + K2 finalize kernel
# CLONES + the original kernel rewritten to the per-output ten-FMA
# evaluation, all scratch in the existing output buffer). Requires the
# paired host LLVM pass -fsycl-rewrite-linreg-fp64-3kernel (the
# SYCLRewriteLinregFp64ThreeKernel pass, which clones the fp64 submit
# three times with launch shapes ngroups*256 / 256 / N) and the runtime
# env SYCL_FORCE_LOCAL_SIZE=256 at run time. fp64-only; the fp32
# instantiation is the sycl-linreg-fp32-unroll pass's. LLVM-MLIR-HIP only.
option(LINREG_FP64_3KERNEL "Enable the paired sycl-linreg-fp64-3kernel (device) + sycl-rewrite-linreg-fp64-3kernel (host 3-submit clone) passes on the lin_reg_error target (LLVM-MLIR-HIP only; run with SYCL_FORCE_LOCAL_SIZE=256)" ON)


set(known_arch_opts tianhe orise sunway)
if(ARCH_OPT)
  list(FIND known_arch_opts ${ARCH_OPT} arch_idx)
  if(arch_idx EQUAL -1)
    message(FATAL_ERROR "Unknown ARCH_OPT '${ARCH_OPT}'. Supported: ${known_arch_opts} (leave empty to build reference kernels only).")
  endif()
endif()

set(benchmarks
  # micro/arith.cpp
  # micro/DRAM.cpp
  # micro/host_device_bandwidth.cpp
  # micro/pattern_L2.cpp
  # micro/sf.cpp
  # micro/local_mem.cpp
  # single-kernel/median.cpp
  single-kernel/scalar_prod.cpp
  # single-kernel/sobel.cpp
  # single-kernel/sobel5.cpp
  # single-kernel/sobel7.cpp
  # single-kernel/vec_add.cpp
  single-kernel/lin_reg_error.cpp
  single-kernel/lin_reg_coeff.cpp
  single-kernel/kmeans.cpp
  # single-kernel/mol_dyn.cpp
  # single-kernel/nbody.cpp
  # pattern/segmentedreduction.cpp
  # pattern/reduction.cpp
  # runtime/dag_task_throughput_sequential.cpp
  # runtime/dag_task_throughput_independent.cpp
  # runtime/blocked_transform.cpp
  # runtime/matmulchain.cpp
  # polybench/2DConvolution.cpp
  polybench/2mm.cpp
  # polybench/3DConvolution.cpp
  polybench/3mm.cpp
  polybench/atax.cpp
  polybench/bicg.cpp
  # polybench/correlation.cpp
  # polybench/covariance.cpp
  # polybench/fdtd2d.cpp
  polybench/gemm.cpp
  polybench/gesummv.cpp
  # polybench/gramschmidt.cpp
  polybench/mvt.cpp
  polybench/syr2k.cpp
  polybench/syrk.cpp
  #compiletime/compiletime.cpp
)

# Architecture-specific optimized kernels. When ARCH_OPT is set (e.g. -DARCH_OPT=tianhe),
# the opt variants under polybench/<ARCH_OPT>/ and single-kernel/<ARCH_OPT>/ are built
# alongside the reference kernels. Each opt kernel has a distinct target name
# (e.g. 2mm_opt, kmeans_opt) so it never collides with the reference 2mm/kmeans target.
if(ARCH_OPT)
  file(GLOB arch_opt_sources "${CMAKE_SOURCE_DIR}/polybench/${ARCH_OPT}/*_opt.cpp")
  list(SORT arch_opt_sources)
  foreach(src ${arch_opt_sources})
    list(APPEND benchmarks ${src})
  endforeach()

  file(GLOB single_kernel_opt_sources "${CMAKE_SOURCE_DIR}/single-kernel/${ARCH_OPT}/*_opt.cpp")
  list(SORT single_kernel_opt_sources)
  foreach(src ${single_kernel_opt_sources})
    list(APPEND benchmarks ${src})
  endforeach()
endif()

foreach(benchmark IN LISTS benchmarks)
  get_filename_component(target ${benchmark} NAME_WE)

  add_executable(${target} ${benchmark})

  if(SYCL_IMPL STREQUAL "ComputeCpp" OR SYCL_IMPL STREQUAL "hipSYCL")
    add_sycl_to_target(TARGET ${target} SOURCES ${benchmark})
  endif()

  if(SYCL_IMPL STREQUAL "ComputeCpp" AND COMPUTECPP_BITCODE STREQUAL "ptx64")
    target_compile_definitions(${target} PRIVATE SYCL_BENCH_ENABLE_QUEUE_PROFILING)
  endif()

  if(SYCL_IMPL STREQUAL "LLVM")
    target_compile_definitions(${target} PRIVATE __LLVM_SYCL__)
  endif()

  if(SYCL_IMPL STREQUAL "LLVM-Sunway")
    # LLVM/DPC++ lowering to the Sunway athread target. __LLVM_SYCL__ selects the
    # same runtime path as plain LLVM; SYCL_BENCH_SUNWAY gates the Sunway-specific
    # workaround in prefetched_buffer.h (the eager-H2D dummy kernel is unsupported
    # on the athread backend and is skipped there).
    target_compile_definitions(${target} PRIVATE __LLVM_SYCL__ SYCL_BENCH_SUNWAY)
  endif()

  if(SYCL_IMPL STREQUAL "LLVM-CUDA")
    target_compile_definitions(${target} PRIVATE __LLVM_SYCL_CUDA__)
  endif()

  if(SYCL_IMPL STREQUAL "triSYCL")
    add_sycl_to_target(${target})
    target_compile_definitions(${target} PRIVATE __TRISYCL__)
  endif()

  if(COVARIANCE_LOOP_REORDER AND target STREQUAL "covariance" AND SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    target_compile_options(${target} PRIVATE -Xcgeist --sycl-covariance-loop-reorder)
  endif()

  if(SYRK_REGISTER_ACCUMULATOR AND target STREQUAL "syrk" AND SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    # SHELL: keeps "-Xcgeist <flag>" as one unit so CMake does not deduplicate
    # the -Xcgeist token when multiple -Xcgeist options land on the same target.
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-syrk-register-accumulator")
  endif()

  if(SYRK_LAUNCH_TILE AND target STREQUAL "syrk" AND SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-syrk-register-tile" -Xclang -fsycl-rewrite-syrk-range)
  endif()

  if(SYR2K_LAUNCH_TILE AND target STREQUAL "syr2k" AND SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-syr2k-register-tile" -Xclang -fsycl-rewrite-syr2k-range)
  endif()

  if(TWO_MM_LAUNCH_TILE AND target STREQUAL "2mm" AND SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-2mm-local-tile" -Xclang -fsycl-rewrite-2mm-range)
  endif()

  if(THREE_MM_LAUNCH_TILE AND target STREQUAL "3mm" AND SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-3mm-local-tile" -Xclang -fsycl-rewrite-3mm-range)
  endif()

  if(GEMM_LAUNCH_TILE AND target STREQUAL "gemm" AND SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-gemm-local-tile" -Xclang -fsycl-rewrite-gemm-range)
  endif()

  if(ATAX_WG_REDUCTION AND target STREQUAL "atax" AND SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-atax-wg-reduction" -Xclang -fsycl-rewrite-atax-range)
  endif()

  if(BICG_WG_REDUCTION AND target STREQUAL "bicg" AND SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-bicg-wg-reduction" -Xclang -fsycl-rewrite-bicg-range)
  endif()

  if(MVT_WG_REDUCTION AND target STREQUAL "mvt" AND SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-mvt-wg-reduction" -Xclang -fsycl-rewrite-mvt-range)
  endif()

  if(GESUMMV_WG_REDUCTION AND target STREQUAL "gesummv" AND SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-gesummv-wg-reduction" -Xclang -fsycl-rewrite-gesummv-range)
  endif()

  if(KMEANS_POINT_BLOCK AND target STREQUAL "kmeans" AND SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-kmeans-point-block" -Xclang -fsycl-rewrite-kmeans-range)
  endif()

  if(KMEANS_DIET AND target STREQUAL "kmeans" AND SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    target_compile_options(${target} PRIVATE "SHELL:-Xclang -fsycl-rewrite-kmeans-diet")
  endif()

  if(LIN_REG_COEFF_READ_RANGE AND target STREQUAL "lin_reg_coeff" AND SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    target_compile_options(${target} PRIVATE -Xclang -fsycl-rewrite-linreg-read-range)
  endif()

  if(LINREG_FP32_UNROLL AND target STREQUAL "lin_reg_error" AND SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-linreg-fp32-unroll")
  endif()

  if(LINREG_FP64_3KERNEL AND target STREQUAL "lin_reg_error" AND SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-linreg-fp64-3kernel" -Xclang -fsycl-rewrite-linreg-fp64-3kernel)
  endif()

  install(TARGETS ${target} RUNTIME DESTINATION bin/benchmarks/)
  get_filename_component(dir ${benchmark} DIRECTORY)
  set_property(TARGET ${target} PROPERTY FOLDER ${dir})
endforeach(benchmark)

# Per-benchmark floating-point flags.
# Pass e.g. -DSYCL_BENCH_FAST_MATH_BENCHMARKS="arith;gemm;2mm" and
#          -DSYCL_BENCH_MIEEE_BENCHMARKS="reduction;median" to cmake.
# NOTE: -ffast-math affects both host and device compilation, while -mieee is
# forwarded by the swsycl driver only to the swg++ host compiler. Do not list
# the same target in both variables.
set(SYCL_BENCH_FAST_MATH_BENCHMARKS "" CACHE STRING
    "Semicolon-separated benchmark target names compiled with -ffast-math")
set(SYCL_BENCH_MIEEE_BENCHMARKS "" CACHE STRING
    "Semicolon-separated benchmark target names compiled with -mieee")

foreach(bench IN LISTS SYCL_BENCH_FAST_MATH_BENCHMARKS)
  if(TARGET ${bench})
    target_compile_options(${bench} PRIVATE -ffast-math)
  endif()
endforeach()

foreach(bench IN LISTS SYCL_BENCH_MIEEE_BENCHMARKS)
  if(TARGET ${bench})
    target_compile_options(${bench} PRIVATE -mieee)
  endif()
endforeach()

# The "compiletime" target should only be used in the context of the compile time evaluation script
#set_target_properties(compiletime PROPERTIES EXCLUDE_FROM_ALL 1)

install(PROGRAMS bin/run-suite DESTINATION bin/)
install(FILES ${PROJECT_SOURCE_DIR}/share/Brommy.bmp DESTINATION share/)
