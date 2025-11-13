#!/bin/bash

set -e

ALPHAS=(-1.0 0.0 1.0)
MODEL="vit_tiny"
DECODER="base-8b"
EPOCHS=5
SEED=0
BATCH_SIZE=64
NUM_WORKERS=4

echo "========================================"
echo "MAE Reconstruction Gap Quick Test"
echo "========================================"
echo "Model: $MODEL"
echo "Decoder: $DECODER"
echo "Epochs: $EPOCHS (quick test)"
echo "Batch size: $BATCH_SIZE"
echo "Alphas: ${ALPHAS[@]}"
echo "Total experiments: ${#ALPHAS[@]}"
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
    echo "Test $COUNTER/${#ALPHAS[@]}: Alpha = $alpha"
    echo "=========================================="

    python benchmarks/imagenette/mae-vit-reconstruction-gap.py \
        --model $MODEL \
        --decoder_type $DECODER \
        --alpha $alpha \
        --epochs $EPOCHS \
        --seed $SEED \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS

    echo "Test $COUNTER/${#ALPHAS[@]} complete!"
    COUNTER=$((COUNTER + 1))
done

echo ""
echo "=========================================="
echo "Quick test complete!"
echo "Running analysis..."
echo "=========================================="

python benchmarks/imagenette/analysis.py \
    --log_dir outputs/imagenette-mae-reconstruction-gap/ \
    --output_dir outputs/analysis_quick_test/

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "Results saved to outputs/analysis_quick_test/"
echo "=========================================="
echo ""
echo "If results look good, run the full sweep:"
echo "  bash benchmarks/imagenette/run_alpha_sweep.sh"
echo "========================================"
