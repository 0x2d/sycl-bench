#!/bin/bash
# Submit every benchmark to the Sunway cluster at --size=512 --local=128.
#
# The polybench *_opt binaries have run-time-selectable tile sizes (no rebuild
# needed; see polybench/sunway/README.md):
#   SYCL_{2MM,3MM,GEMM,COVARIANCE,CORRELATION}_TS   shared-mem tile edge (4/8/16/32, default 8)
#   SYCL_{SYRK,SYR2K}_{BM,BN}                       register tile edge (1/2/4/8, default 8)
# The selected tile is appended to the benchmark name in sycl-bench.csv, so
# swept runs stay distinguishable. To sweep one benchmark, export the vars
# before invoking this script, e.g.:
#   for ts in 4 8 16 32; do SYCL_GEMM_TS=$ts ./run-suite-sw.sh; done
SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]:-$0}")")

export LD_LIBRARY_PATH=/home/export/online1/mdt00/shisuan/swyjs/oyyc/lib:$LD_LIBRARY_PATH

out_args="--output=sycl-bench.csv"
DEFAULT_ARGS="--size=512 --local=128 --num-runs=10"

declare -A ARGS=(
    [2mm]="--size=512 --local=256 --num-runs=10"
    [2DConvolution]="--size=8192 --local=256 --num-runs=10"
    [3mm]="--size=512 --local=256 --num-runs=10"
    [atax]="--size=8192 --local=256 --num-runs=10"
    [bicg]="--size=8192 --local=256 --num-runs=10"
    [correlation]="--size=512 --local=256 --num-runs=10"
    [covariance]="--size=512 --local=256 --num-runs=10"
    [gemm]="--size=512 --local=256 --num-runs=10"
    [syrk]="--size=1024 --local=256 --num-runs=10"
    [syr2k]="--size=1024 --local=256 --num-runs=10"
    [2mm_opt]="--size=512 --local=256 --num-runs=10"
    [3mm_opt]="--size=512 --local=256 --num-runs=10"
    [atax_opt]="--size=8192 --local=256 --num-runs=10"
    [bicg_opt]="--size=8192 --local=256 --num-runs=10"
    [correlation_opt]="--size=512 --local=256 --num-runs=10"
    [covariance_opt]="--size=512 --local=256 --num-runs=10"
    [gemm_opt]="--size=512 --local=256 --num-runs=10"
    [syrk_opt]="--size=1024 --local=256 --num-runs=10"
    [syr2k_opt]="--size=1024 --local=256 --num-runs=10"
    [kmeans]="--size=8388608 --local=256 --num-runs=10"
    [lin_reg_coeff]="--size=2097152 --local=256 --num-runs=10"
    [median]="--size=2048 --local=256 --num-runs=10"
    [sobel]="--size=4096 --local=256 --num-runs=10"
)

for file in ${SCRIPT_DIR}/benchmarks/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        args="${ARGS[$filename]:-$DEFAULT_ARGS}"

        # bsub -J 2mm -I -q q_share -b -m 1 -n 1 -cgsp 64 -share_size 13000 -priv_size 16 -host_stack 1024 -cache_size 128 ./2mm_opt --size=1024 --local=256 --num-runs=10
        cmd=(bsub -J "$filename" -q q_share -b -m 1 -n 1 -cgsp 64 -share_size 13000 -priv_size 16 -host_stack 1024 -cache_size 128 "$file" $args $out_args)
        echo "+ ${cmd[*]}"
        "${cmd[@]}"

        if [ $? -eq 0 ]; then
            echo "=== Successfully submitted: $filename ==="
        else
            echo "=== Failed to submit: $filename ==="
        fi
    fi
done
