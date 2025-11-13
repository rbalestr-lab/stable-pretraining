#!/bin/bash

set -e

ALPHAS=(-1.0 0.0 1.0)
MODEL="vit_tiny"
DECODER="base-8b"
EPOCHS=5
SEED=0
BATCH_SIZE=64
NUM_WORKERS=4

echo "${#ALPHAS[@]} runs"

if [ ! -f "benchmarks/imagenette/mae-vit-reconstruction-gap.py" ]; then
    echo "run from repo root"
    exit 1
fi

COUNTER=1
for alpha in "${ALPHAS[@]}"; do
    echo "[$COUNTER/${#ALPHAS[@]}] a=$alpha"

    python benchmarks/imagenette/mae-vit-reconstruction-gap.py \
        --model $MODEL \
        --decoder_type $DECODER \
        --alpha $alpha \
        --epochs $EPOCHS \
        --seed $SEED \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS

    COUNTER=$((COUNTER + 1))
done

python benchmarks/imagenette/analysis.py \
    --log_dir outputs/imagenette-mae-reconstruction-gap/ \
    --output_dir outputs/analysis_quick_test/

echo "done"
