#!/bin/bash


if [[ $# -ne 3 ]]; then
    echo "Usage: bash ./scripts/run_distributed_train_gpu.sh [RANK_SIZE] [CONFIG] [OUTDIR]"
    exit 1
fi

rank_size=$1
config=$2
outdir=$3

mkdir -p "${outdir}/log"
mpirun -n $rank_size --allow-run-as-root python tools/train.py $config --outdir $outdir > ${outdir}/log/train.log 2>&1 &
