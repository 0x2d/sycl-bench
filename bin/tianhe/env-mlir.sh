source /thfs1/software/sycl/mocl3.1.3/env.sh
export LLVM_ROOT=/thfs1/software/llvm/llvm17-mt1
export MOCL_CORE_NUMS=${MOCL_CORE_NUMS:-24}
export OCL_ICD_VENDORS=${MOCL3_ROOT}/etc/OpenCL/vendors
export LD_LIBRARY_PATH=/thfs1/software/sycl/opencl-icd-loader/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/thfs1/home/ouyyc/sycl-mlir/build/install/lib:${LD_LIBRARY_PATH}
export POCL_DEBUG=0;