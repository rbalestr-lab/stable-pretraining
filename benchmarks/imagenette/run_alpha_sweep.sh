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

echo "========================================"
echo "MAE Reconstruction Gap Alpha Sweep"
echo "========================================"
echo "Model: $MODEL"
echo "Decoder: $DECODER"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Seed: $SEED"
echo "Total experiments: ${#ALPHAS[@]}"
echo "WandB Project: $WANDB_PROJECT"
echo "========================================"
echo ""

if [ ! -f "benchmarks/imagenette/mae-vit-reconstruction-gap.py" ]; then
    echo "Error: Must run from repository root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

COUNTER=1
for alpha in "${ALPHAS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Experiment $COUNTER/${#ALPHAS[@]}: Alpha = $alpha"
    echo "=========================================="

    python benchmarks/imagenette/mae-vit-reconstruction-gap.py \
        --model $MODEL \
        --decoder_type $DECODER \
        --alpha $alpha \
        --epochs $EPOCHS \
        --seed $SEED \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --project $WANDB_PROJECT

    echo "Experiment $COUNTER/${#ALPHAS[@]} complete!"
    COUNTER=$((COUNTER + 1))
done

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "Running final analysis..."
echo "=========================================="

python benchmarks/imagenette/analysis.py \
    --log_dir outputs/imagenette-mae-reconstruction-gap/ \
    --output_dir outputs/analysis_full_sweep/ \
    --wandb_project $WANDB_PROJECT

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "=========================================="
echo "Local results: outputs/analysis_full_sweep/"
echo "WandB project: $WANDB_PROJECT"
echo "========================================"
echo ""
echo "✓ Reconstruction gap tube plot uploaded to WandB"
echo "✓ All metrics and visualizations available online"
echo "  - tube_scatter_signed_alpha.png"
echo "  - alpha_vs_final_metrics.png"
echo "  - negative_vs_positive_comparison.png"
echo "  - gap_statistics.json"
echo "========================================"
