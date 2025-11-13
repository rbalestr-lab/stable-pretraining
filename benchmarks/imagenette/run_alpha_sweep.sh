#!/bin/bash

set -e

ALPHAS=(
    -10.0 -5.0 -2.0 -1.0
    -0.5 -0.1 -0.05 -0.01 -0.005 -0.001
    -0.0001 -0.00001
    0.0
    0.00001 0.0001
    0.001 0.005 0.01 0.05 0.1 0.5
    1.0 2.0 5.0 10.0
)

MODEL="vit_tiny"
DECODER="base-8b"
EPOCHS=50
SEED=0
BATCH_SIZE=64
NUM_WORKERS=4
WANDB_PROJECT="mae-reconstruction-gap-full-sweep"

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
        --num_workers $NUM_WORKERS \
        --project $WANDB_PROJECT

    COUNTER=$((COUNTER + 1))
done

python benchmarks/imagenette/analysis.py \
    --log_dir outputs/imagenette-mae-reconstruction-gap/ \
    --output_dir outputs/analysis_full_sweep/ \
    --wandb_project $WANDB_PROJECT

echo "done"
