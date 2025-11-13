# MAE Reconstruction Gap Experiment

## Overview

This experiment investigates the **reconstruction gap** in Masked Autoencoders (MAE) using signed alpha regularization. The goal is to understand how adding a supervised classification signal (with varying strengths) during MAE pre-training affects the relationship between reconstruction quality and learned representations.

### Key Research Questions:
1. How does signed alpha regularization affect MAE's reconstruction-representation tradeoff?
2. Is there a "reconstruction gap" where similar reconstruction loss yields different semantic representations?
3. How do negative alpha values (anti-regularization) compare to positive values (regularization)?

## Experiment Setup

### Model Configuration:
- **Encoder**: Vision Transformer Tiny (vit_tiny) - 31.2M parameters
- **Decoder**: Transformer-based, 8 layers, 512 embed_dim, 8 heads
- **Dataset**: Imagenette (10-class subset of ImageNet)
- **Image Size**: 224x224
- **Mask Ratio**: 0.75 (MAE standard)

### Training Configuration:
- **Optimizer**: AdamW (shared between encoder and manual supervised probe)
- **Learning Rate**: 1.5e-4
- **Weight Decay**: 0.05
- **Precision**: 16-bit AMP (Automatic Mixed Precision) - **Compatible with B200 GPUs**
- **Loss**: `total_loss = mae_reconstruction_loss + alpha * supervised_classification_loss`

### Alpha Values Tested:

**Quick Test Sweep** (3 alphas, 10 epochs):
- Alpha = -0.5, 0.0, 0.5

**Full Sweep** (26 alphas, 50 epochs):
- Negative (anti-regularization): -10.0, -5.0, -2.0, -1.0, -0.5, -0.1, -0.05, -0.01, -0.005, -0.001, -0.0001, -0.00001
- Baseline: 0.0 (pure MAE)
- Positive (regularization): 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0

## Installation

### Prerequisites:
- Python 3.12.3
- PyTorch 2.8.0+cu128 (REQUIRED for B200 GPUs - do NOT change)
- CUDA 12.8

### Install Dependencies:

```bash
# From repository root
cd /workspace/ship3copy

# Install frozen requirements (compatible with PyTorch 2.8)
pip install -r requirements_frozen_v45.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Key Packages:
- `torch==2.8.0+cu128` - **DO NOT CHANGE** (required for B200)
- `datasets==3.6.0` - For Imagenette loading
- `wandb==0.23.0` - For experiment tracking
- `stable-pretraining` - Custom SSL framework (installed from git)
- `pytorch-lightning==2.5.6` - Training framework
- `timm==1.0.22` - Vision Transformer models

## Running Experiments

### 1. Quick Test Sweep (Recommended First)

**Purpose**: Validate setup and observe basic trends (3 alphas × 10 epochs = ~30 minutes on single GPU)

```bash
# From repository root
cd /workspace/ship3copy

# Login to WandB (one-time setup)
wandb login

# Run quick test
bash benchmarks/imagenette/quick_sweep_test.sh
```

**Configuration**:
- Alphas: -0.5, 0.0, 0.5
- Epochs: 10
- Batch Size: 64
- WandB Project: `mae-reconstruction-gap-manual-probes`

**Expected Runtime**: ~10-15 minutes per alpha value on single A4500 GPU

### 2. Learning Rate + Alpha Sweep (NEW)

**Purpose**: Explore reconstruction gap across both learning rates and alpha values (2 LRs × 3 alphas × 10 epochs = ~1 hour)

```bash
# From repository root
cd /workspace/ship3copy

# Login to WandB (if not already)
wandb login

# Run LR + alpha sweep
bash benchmarks/imagenette/lr_alpha_sweep.sh
```

**Configuration**:
- Learning Rates: 1e-3, 1e-4
- Alphas: -0.5, 0.0, 0.5
- Epochs: 10
- Batch Size: 64
- Total experiments: 6 (2 LRs × 3 alphas)
- Total data points: 60 (6 experiments × 10 epochs)
- WandB Project: `mae-reconstruction-gap-lr-sweep`

**Expected Runtime**: ~60-90 minutes on single A4500 GPU

**Visualization Encoding**:
- **Hue (color family)**: Alpha value (red=negative, white=0, blue=positive)
- **Brightness**: Composite of epoch and learning rate (darker→brighter shows training progress)
- **Point size**: Constant (all points same size)

### 3. Full Alpha Sweep (Production Run for B200)

**Purpose**: Comprehensive sweep across 26 alpha values (26 alphas × 50 epochs = ~13-15 hours on 3x B200)

```bash
# From repository root
cd /workspace/ship3copy

# Login to WandB (if not already)
wandb login

# Run full sweep
bash benchmarks/imagenette/run_alpha_sweep.sh
```

**Configuration**:
- Alphas: 26 values (see above)
- Epochs: 50
- Batch Size: 512 (optimized for 3x B200 GPUs)
- WandB Project: `mae-reconstruction-gap-full-sweep`

**Expected Runtime** (estimated for 3x B200 GPUs):
- Per alpha: ~20-30 minutes
- Total: ~13-15 hours

**B200 Optimization Notes**:
- Batch size can be increased further if GPU memory allows
- Consider using `--num_workers=32` for faster data loading
- AMP (16-bit precision) is enabled by default - provides 2-3x speedup

## Outputs & Logging

### Per-Experiment Outputs:

Each alpha value produces:
1. **Checkpoints**: `outputs/checkpoints/vit_tiny_base-8b_alpha-{alpha}_seed0/`
2. **Trajectory File**: `outputs/imagenette-mae-reconstruction-gap/{alpha}/trajectory.json`
   - Contains (MSE, Accuracy) pairs for every epoch
   - Used for reconstruction gap analysis
3. **WandB Logs**:
   - Training loss (reconstruction + supervised)
   - Validation metrics
   - Learning rate schedules
   - Real-time visualizations

### Final Analysis Outputs:

After all experiments complete, `analysis.py` generates:

1. **Tube Scatter Plot** (`tube_scatter_signed_alpha.png`):
   - Shows reconstruction gap: (MSE, Accuracy) pairs across all alphas/epochs
   - Color-coded by alpha value
   - Computes tube volume Γ (spread measure)

2. **Alpha vs Metrics** (`alpha_vs_final_metrics.png`):
   - Final reconstruction MSE vs alpha
   - Final classification accuracy vs alpha

3. **Negative vs Positive Comparison** (`negative_vs_positive_comparison.png`):
   - Compares anti-regularization (α < 0) vs regularization (α > 0)

4. **Statistics JSON** (`gap_statistics.json`):
   - Reconstruction gap quantification
   - Bin-wise statistics
   - Spearman correlations

**All plots and statistics are automatically uploaded to WandB!**

## WandB Projects

### Quick Test:
- **Project**: `mae-reconstruction-gap-manual-probes`
- **URL**: https://wandb.ai/uncletr39-brown-university/mae-reconstruction-gap-manual-probes

### Full Sweep:
- **Project**: `mae-reconstruction-gap-full-sweep`
- **URL**: https://wandb.ai/uncletr39-brown-university/mae-reconstruction-gap-full-sweep

## File Structure

```
benchmarks/imagenette/
├── mae-vit-reconstruction-gap.py              # Main training script
├── mae-vit-reconstruction-gap-FrozenOnlineProbeLars.py  # Backup with OnlineProbe (not used)
├── quick_sweep_test.sh                        # Quick 3-alpha test sweep
├── lr_alpha_sweep.sh                          # Learning rate + alpha sweep (NEW)
├── run_alpha_sweep.sh                         # Full 26-alpha sweep
├── analysis.py                                # Post-processing & visualization
├── plot_reconstruction_gap.py                 # Alternative plotting script
└── README_MAE_RECONSTRUCTION_GAP.md          # This file
```

## Important Notes

### 1. Manual Supervised Probes (Current Setup)

The experiment uses **manual supervised probes** that:
- Share the same AdamW optimizer as the MAE encoder
- Use the same learning rate (1.5e-4)
- Are controlled by the `alpha` parameter
- **Alpha != 0**: Probe trains jointly with encoder (gradients flow)
- **Alpha = 0**: Probe excluded from optimizer (pure MAE baseline)

This follows the **SupMAE** (Supervised MAE) standard for joint training.

### 2. Frozen OnlineProbe (Not Included)

We created a diagnostic-only `OnlineProbe` with LARS optimizer that:
- Trains on frozen encoder features
- Does NOT affect main model training
- Uses `add_to_loss=False` parameter

**Current Status**: Removed due to AMP compatibility issues with multiple optimizers
**Backup**: Saved in `mae-vit-reconstruction-gap-FrozenOnlineProbeLars.py`
**Future Work**: Can be re-added after fixing AMP/multi-optimizer support

### 3. PyTorch 2.8 Requirement

**CRITICAL**: Do NOT downgrade PyTorch! Version 2.8.0+cu128 is required for:
- B200 GPU compatibility
- CUDA 12.8 features
- Optimal performance

If package conflicts occur, consult `requirements_frozen_v45.txt` for compatible versions.

### 4. Ground Truth Verification

All three tasks use correct ground truth:
- ✅ MAE Reconstruction: Uses `batch["image"]` (original images)
- ✅ Manual Supervised Probe: Uses `batch["label"]` (Imagenette class labels)
- ✅ Data flow verified end-to-end

## Monitoring Progress

### During Training:

```bash
# Check running experiment status
tail -f wandb/latest-run/files/output.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check WandB online dashboard
# Navigate to the appropriate project URL above
```

### Experiment Status:

```bash
# List completed trajectory files
find outputs/imagenette-mae-reconstruction-gap -name "trajectory.json"

# Count completed experiments
find outputs/imagenette-mae-reconstruction-gap -name "trajectory.json" | wc -l

# Check most recent experiment
ls -lht outputs/checkpoints/ | head
```

## Troubleshooting

### Issue: "datasets==4.0.0" error
**Solution**: Downgrade to `datasets==3.6.0` (already in requirements_frozen_v45.txt)

### Issue: AMP/GradScaler errors with multiple optimizers
**Solution**: Use manual probes only (current setup), not OnlineProbe

### Issue: Out of memory
**Solution**: Reduce `BATCH_SIZE` in sweep scripts (currently 512 for full sweep, 64 for quick test)

### Issue: Slow data loading
**Solution**: Increase `NUM_WORKERS` in sweep scripts (currently 16, can go up to 32 for B200)

## Expected Results

### Reconstruction Gap Hypothesis:

We expect to observe a "tube" in the (MSE, Accuracy) scatter plot, indicating that:
- Multiple encoder configurations achieve similar reconstruction MSE
- But produce very different semantic representations (classification accuracy)
- This demonstrates MAE's ill-posed reconstruction objective

### Alpha Effect Hypothesis:

- **Negative alpha (α < 0)**: Anti-regularization may harm both reconstruction and accuracy
- **Zero alpha (α = 0)**: Pure MAE baseline
- **Positive alpha (α > 0)**: Regularization may improve accuracy at cost of reconstruction

### Quantitative Metrics:

- **Tube Volume (Γ)**: Sum of accuracy spread across MSE bins - measures reconstruction gap
  - Algorithm: Bin MSE into 15 equal-width bins, compute max(accuracy) - min(accuracy) per bin, sum across bins
  - High Γ = large accuracy variance at similar MSE → strong reconstruction gap
  - Low Γ = accuracies tightly coupled to MSE → no reconstruction gap
- **Spearman Correlation (ρ)**: Correlation between MSE and accuracy - negative indicates gap

## Citation

If using this code, please cite:
- **MAE**: "Masked Autoencoders Are Scalable Vision Learners" (He et al., CVPR 2022)
- **SupMAE**: "Supervised Masked Autoencoders Are Efficient Vision Learners" (CVPR 2022)
- **stable-pretraining**: Custom SSL framework (https://github.com/akshayg108/stable-pretraining)

## Contact

For questions or issues:
- Check WandB logs first: [Project URLs above]
- Review trajectory.json files for per-epoch metrics
- Verify PyTorch 2.8 compatibility

---

**Last Updated**: 2025-11-13
**Experiment Version**: MAE Reconstruction Gap - Manual Probes Only
**Hardware Target**: 3x NVIDIA B200 GPUs
