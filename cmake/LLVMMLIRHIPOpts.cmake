# LLVM-MLIR-HIP-specific optimization passes.
#
# This file carries (a) every ARCH_OPT-independent option() that gates a
# cgeist device pass / paired -fsycl-rewrite-* host pass on a specific
# benchmark target, and (b) the per-target wiring that turns those options
# into compile flags. It is included once from the top-level CMakeLists.txt;
# the per-benchmark hook is apply_llvm_mlir_hip_opts(<target>), called from
# the benchmarks foreach() loop. Everything here is LLVM-MLIR-HIP only --
# the apply function returns early for every other SYCL_IMPL.

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
# for A/B baselines). Reproduces polybench/orise/3mm_opt.cpp without touching
# benchmark source. LLVM-MLIR-HIP only; on by default.
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

# Per-target wiring: called once per benchmark from the top-level
# CMakeLists.txt foreach() loop. No-op unless SYCL_IMPL is LLVM-MLIR-HIP.
function(apply_llvm_mlir_hip_opts target)
  if(NOT SYCL_IMPL STREQUAL "LLVM-MLIR-HIP")
    return()
  endif()

  if(COVARIANCE_LOOP_REORDER AND target STREQUAL "covariance")
    target_compile_options(${target} PRIVATE -Xcgeist --sycl-covariance-loop-reorder)
  endif()

  if(SYRK_REGISTER_ACCUMULATOR AND target STREQUAL "syrk")
    # SHELL: keeps "-Xcgeist <flag>" as one unit so CMake does not deduplicate
    # the -Xcgeist token when multiple -Xcgeist options land on the same target.
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-syrk-register-accumulator")
  endif()

  if(SYRK_LAUNCH_TILE AND target STREQUAL "syrk")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-syrk-register-tile" -Xclang -fsycl-rewrite-syrk-range)
  endif()

  if(SYR2K_LAUNCH_TILE AND target STREQUAL "syr2k")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-syr2k-register-tile" -Xclang -fsycl-rewrite-syr2k-range)
  endif()

  if(TWO_MM_LAUNCH_TILE AND target STREQUAL "2mm")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-2mm-local-tile" -Xclang -fsycl-rewrite-2mm-range)
  endif()

  if(THREE_MM_LAUNCH_TILE AND target STREQUAL "3mm")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-3mm-local-tile" -Xclang -fsycl-rewrite-3mm-range)
  endif()

  if(GEMM_LAUNCH_TILE AND target STREQUAL "gemm")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-gemm-local-tile" -Xclang -fsycl-rewrite-gemm-range)
  endif()

  if(ATAX_WG_REDUCTION AND target STREQUAL "atax")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-atax-wg-reduction" -Xclang -fsycl-rewrite-atax-range)
  endif()

  if(BICG_WG_REDUCTION AND target STREQUAL "bicg")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-bicg-wg-reduction" -Xclang -fsycl-rewrite-bicg-range)
  endif()

  if(MVT_WG_REDUCTION AND target STREQUAL "mvt")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-mvt-wg-reduction" -Xclang -fsycl-rewrite-mvt-range)
  endif()

  if(GESUMMV_WG_REDUCTION AND target STREQUAL "gesummv")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-gesummv-wg-reduction" -Xclang -fsycl-rewrite-gesummv-range)
  endif()

  if(KMEANS_POINT_BLOCK AND target STREQUAL "kmeans")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-kmeans-point-block" -Xclang -fsycl-rewrite-kmeans-range)
  endif()

  if(KMEANS_DIET AND target STREQUAL "kmeans")
    target_compile_options(${target} PRIVATE "SHELL:-Xclang -fsycl-rewrite-kmeans-diet")
  endif()

  if(LIN_REG_COEFF_READ_RANGE AND target STREQUAL "lin_reg_coeff")
    target_compile_options(${target} PRIVATE -Xclang -fsycl-rewrite-linreg-read-range)
  endif()

  if(LINREG_FP32_UNROLL AND target STREQUAL "lin_reg_error")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-linreg-fp32-unroll")
  endif()

  if(LINREG_FP64_3KERNEL AND target STREQUAL "lin_reg_error")
    target_compile_options(${target} PRIVATE "SHELL:-Xcgeist --sycl-linreg-fp64-3kernel" -Xclang -fsycl-rewrite-linreg-fp64-3kernel)
  endif()
endfunction()
