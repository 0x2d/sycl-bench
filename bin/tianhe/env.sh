#opecl-headers
VVV_OCL_HEADERS=/thfs1/software/sycl/opencl-headers
export CPLUS_INCLUDE_PATH=${VVV_OCL_HEADERS}/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=${VVV_OCL_HEADERS}/include:$C_INCLUDE_PATH
export PKG_CONFIG_PATH=${VVV_OCL_HEADERS}/share/pkgconfig:$PKG_CONFIG_PATH

#opencl-icd-loader
VVV_ICD_LOADER=/thfs1/software/sycl/opencl-icd-loader
export PKG_CONFIG_PATH=${VVV_ICD_LOADER}/share/pkgconfig:$PKG_CONFIG_PATH
export LIBRARY_PATH=${VVV_ICD_LOADER}/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${VVV_ICD_LOADER}/lib:$LD_LIBRARY_PATH

#dpcpp
export DPCPP_HOME=/thfs1/software/sycl/dpcpp
BASE_PATH=$DPCPP_HOME
export PATH=${BASE_PATH}/bin:$PATH
export CPLUS_INCLUDE_PATH=${BASE_PATH}/include/:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=${BASE_PATH}/include/:$C_INCLUDE_PATH
#
export CPLUS_INCLUDE_PATH=${BASE_PATH}/include/sycl:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=${BASE_PATH}/include/sycl:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=${BASE_PATH}/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=${BASE_PATH}/lib:$LIBRARY_PATH
export CC=clang CXX=clang++
export SYCL_RT_WARNING_LEVEL=2

#mocl
MOCL3_ROOT=/thfs1/software/sycl/mocl3.1.3

POCL_ROOT=${MOCL3_ROOT}
MT3X_ROOT=${MOCL3_ROOT}/mt3x_toolkit
HT_ROOT=${MOCL3_ROOT}/mt3x_toolkit/hthreads
HWLOC_ROOT=${MOCL3_ROOT}/libdeps/hwloc

export PATH=${MOCL3_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${MOCL3_ROOT}/lib:$LD_LIBRARY_PATH 
export PATH=${HWLOC_ROOT}/bin:${POCL_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${HT_ROOT}/lib:${HWLOC_ROOT}/lib:${POCL_ROOT}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${MT3X_ROOT}/third-party-lib:$LD_LIBRARY_PATH; # dsp compiler deps lib(libmpfr)
export LIBRARY_PATH=${HT_ROOT}/lib:${HWLOC_ROOT}/lib:${POCL_ROOT}/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=${HT_ROOT}/include:${HWLOC_ROOT}/include:${POCL_ROOT}/include:$C_INCLUDE_PATH
export PKG_CONFIG_PATH=${HWLOC_ROOT}/lib/pkgconfig:${POCL_ROOT}/lib/pkgconfig:$PKG_CONFIG_PATH

# LLVM
export LLVM_ROOT=/thfs1/software/llvm/llvm17-mt

# POCL and MT env setting
export POCL_DEBUG=0;
export MT_USERLEVEL=1;
export POCL_EXTRA_BUILD_FLAGS="-cl-opt-disable";
export MOCL_CORE_NUMS=24;
export OCL_ICD_VENDORS=${MOCL3_ROOT}/etc/OpenCL/vendors;

#
export POCL_TEST_DEVICE_TYPE=DSP # just for mocl3 tester