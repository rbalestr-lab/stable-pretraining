#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "usage: $0 <dataset_name>"
    exit 1
fi

DATASET="$1"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_NAME="sweep_${TIMESTAMP}"
LOG_DIR="outputs/${DATASET}-${SWEEP_NAME}"

LRS=(1e-4 1e-3)
LAMBDAS=(0.0 1.0)

MODEL="vit_tiny"
ENCODER_DIM=320
ENCODER_DEPTH=8
ENCODER_HEADS=5
PATCH_SIZE=16

DECODER_DIM=192
DECODER_DEPTH=4
DECODER_HEADS=3

EPOCHS=2
SEED=42
BATCH_SIZE=128
NUM_WORKERS=4
WANDB_PROJECT="${DATASET}-${SWEEP_NAME}"

echo "$((${#LRS[@]} * ${#LAMBDAS[@]})) runs"

if [ ! -f "benchmarks/illposed_reconstruction_mae_benchmark/mae-vit-reconstruction-gap.py" ]; then
    echo "run from repo root"
    exit 1
fi

COUNTER=1
TOTAL=$((${#LRS[@]} * ${#LAMBDAS[@]}))

for lr in "${LRS[@]}"; do
    for lambda in "${LAMBDAS[@]}"; do
        echo "[$COUNTER/$TOTAL] lr=$lr, λ=$lambda"

        HF_HUB_ENABLE_HF_TRANSFER=0 python benchmarks/illposed_reconstruction_mae_benchmark/mae-vit-reconstruction-gap.py \
            --dataset $DATASET \
            --model $MODEL \
            --encoder_embed_dim $ENCODER_DIM \
            --encoder_depth $ENCODER_DEPTH \
            --encoder_num_heads $ENCODER_HEADS \
            --patch_size $PATCH_SIZE \
            --decoder_embed_dim $DECODER_DIM \
            --decoder_depth $DECODER_DEPTH \
            --decoder_num_heads $DECODER_HEADS \
            --alpha $lambda \
            --lr $lr \
            --epochs $EPOCHS \
            --seed $SEED \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --project $WANDB_PROJECT \
            --log_dir $LOG_DIR

        COUNTER=$((COUNTER + 1))
    done
done

python benchmarks/illposed_reconstruction_mae_benchmark/analysis.py \
    --dataset $DATASET \
    --wandb_project $WANDB_PROJECT \
    --log_dir $LOG_DIR

echo "done"
