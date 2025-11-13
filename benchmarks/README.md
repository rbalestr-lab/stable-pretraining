# Stable-Pretraining Benchmarks

This directory contains benchmark scripts for various self-supervised learning methods.

## Data Storage Configuration

By default, datasets are stored in `~/.cache/stable-pretraining/data/`. This location can be customized in two ways:

### Option 1: Environment Variable (Recommended)

Set the `STABLE_PRETRAINING_DATA_DIR` environment variable to specify a custom data directory:

```bash
# Set for current session
export STABLE_PRETRAINING_DATA_DIR=/path/to/your/data

# Or set permanently in your shell configuration (~/.bashrc, ~/.zshrc, etc.)
echo 'export STABLE_PRETRAINING_DATA_DIR=/path/to/your/data' >> ~/.bashrc
```

### Option 2: Default Location

If no environment variable is set, data will be stored in:
- `~/.cache/stable-pretraining/data/` (Linux/Mac)
- `C:\Users\<username>\.cache\stable-pretraining\data\` (Windows)

Each dataset will be stored in its own subdirectory (e.g., `cifar10/`, `imagenet/`, etc.).

## Running Benchmarks

### CIFAR-10 Benchmarks

```bash
# SimCLR
python benchmarks/cifar10/simclr-resnet18.py

# BYOL
python benchmarks/cifar10/byol-resnet18.py

# VICReg
python benchmarks/cifar10/vicreg-resnet18.py

# Barlow Twins
python benchmarks/cifar10/barlow-resnet18.py

# NNCLR
python benchmarks/cifar10/nnclr-resnet18.py
```

### Imagenette Benchmarks - MAE Reconstruction Gap

The MAE reconstruction gap benchmark demonstrates the ill-posedness of MAE's reconstruction objective through signed alpha regularization. It supports **two interfaces**: direct Python scripts and Hydra configs with SLURM submission.

#### Method 1: Direct Python Scripts (Simple, Quick)

```bash
# Pure MAE (alpha=0)
python benchmarks/imagenette/mae-vit-reconstruction-gap.py --model vit_tiny --decoder_type base-8b --alpha 0.0

# Positive regularization (alpha>0, aids representation learning)
python benchmarks/imagenette/mae-vit-reconstruction-gap.py --model vit_tiny --decoder_type base-8b --alpha 1.0

# Negative regularization (alpha<0, adversarial to classification)
python benchmarks/imagenette/mae-vit-reconstruction-gap.py --model vit_tiny --decoder_type base-8b --alpha -1.0

# Run full alpha sweep (25 values from -10 to +10) via bash script
bash benchmarks/imagenette/run_alpha_sweep.sh

# Quick test (3 alphas: -1.0, 0.0, 1.0)
bash benchmarks/imagenette/quick_test.sh
```

#### Method 2: Hydra Configs (Advanced, SLURM-ready)

**Basic usage:**
```bash
# Single run with default parameters
python -m stable_pretraining.run --config-path benchmarks/imagenette --config-name mae_reconstruction_gap

# Override parameters via CLI
python -m stable_pretraining.run --config-path benchmarks/imagenette --config-name mae_reconstruction_gap \
    model_name=vit_small \
    decoder_type=tiny-4b \
    alpha=1.0 \
    trainer.max_epochs=50
```

**Multi-run sweeps (local):**
```bash
# Quick 3-point alpha sweep
python -m stable_pretraining.run --config-path benchmarks/imagenette --config-name mae_reconstruction_gap \
    --multirun alpha=-1.0,0.0,1.0

# Full 25-point alpha sweep
python -m stable_pretraining.run --config-path benchmarks/imagenette --config-name mae_reconstruction_gap \
    --multirun alpha=-10,-5,-2,-1,-0.5,-0.1,-0.05,-0.01,-0.005,-0.001,-0.0001,-0.00001,0.0,0.00001,0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1.0,2.0,5.0,10.0

# Sweep multiple dimensions (model + decoder + alpha)
python -m stable_pretraining.run --config-path benchmarks/imagenette --config-name mae_reconstruction_gap \
    --multirun model_name=vit_tiny,vit_small decoder_type=base-8b,linear alpha=-1.0,0.0,1.0
```

**SLURM submission (cluster):**
```bash
# Single SLURM job
python -m stable_pretraining.run --config-path benchmarks/imagenette --config-name mae_reconstruction_gap_slurm

# Alpha sweep submitted to SLURM (each alpha value gets its own job)
python -m stable_pretraining.run --config-path benchmarks/imagenette --config-name mae_reconstruction_gap_slurm \
    --multirun alpha=-10,-5,-2,-1,-0.5,-0.1,-0.05,-0.01,-0.005,-0.001,-0.0001,-0.00001,0.0,0.00001,0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1.0,2.0,5.0,10.0

# With custom SLURM settings
python -m stable_pretraining.run --config-path benchmarks/imagenette --config-name mae_reconstruction_gap_slurm \
    --multirun alpha=-1.0,0.0,1.0 \
    hydra.launcher.partition=gpu \
    hydra.launcher.qos=normal \
    hydra.launcher.timeout_min=360 \
    hydra.launcher.mem_gb=32
```

**Important:** Edit `benchmarks/imagenette/mae_reconstruction_gap_slurm.yaml` to set:
- `hydra.launcher.partition` - Your cluster's GPU partition
- `hydra.launcher.qos` - Your Quality of Service setting

#### Configuration Options

The MAE reconstruction gap benchmark supports:
- **Models**: `vit_tiny`, `vit_small`, `vit_base`, `vit_large`
- **Decoders**: `linear` or `<size>-<depth>b` (e.g., `base-8b`, `tiny-4b`, `small-8b`)
- **Alpha**: Any float value (negative for anti-regularization, zero for pure MAE, positive for regularization)
- **Mask ratio**: Default 0.75 (can be adjusted)
- **Training epochs**: Default 100 (can be adjusted)

#### Visualization Features

The MAE reconstruction gap benchmark includes comprehensive WandB visualization:

**Fixed Validation Samples:**
- Automatically generates 16 fixed validation samples on first run
- Ensures consistent visual comparison across all alpha values
- Samples are saved to `fixed_samples/fixed_val_samples.pt` and reused
- Set via `--fixed_samples_dir` argument (default: `fixed_samples/`)

**Reconstruction Visualizations:**
- 4-column grid logged to WandB every N epochs:
  1. **Original GT**: Ground truth images
  2. **Masked GT**: Ground truth with masked patches removed
  3. **Partial Recon**: Visible patches + reconstructed masked patches
  4. **Full Recon**: Complete reconstruction from model
- Controlled via `--viz_interval` argument (default: 10 epochs)
- Uses fixed samples for fair comparison across experiments

**Automatic Metrics Logging:**
- Reconstruction loss (MSE) tracked every epoch
- Linear probe accuracy tracked via OnlineProbe callback
- KNN probe accuracy tracked via OnlineKNN callback
- Learning rate logged every step via LearningRateMonitor
- All metrics automatically logged to WandB with interactive plots

**Example with custom visualization:**
```bash
python benchmarks/imagenette/mae-vit-reconstruction-gap.py \
    --model vit_tiny \
    --decoder_type base-8b \
    --alpha 1.0 \
    --viz_interval 5 \
    --fixed_samples_dir my_fixed_samples/
```

**WandB Dashboard:**
- View reconstruction visualizations in the "Media" tab
- Compare reconstruction quality across alpha values side-by-side
- Track (MSE, Accuracy) trajectories in real-time
- Export trajectory data from `{log_dir}/trajectory.json`

## Notes

- The data directory will be created automatically if it doesn't exist
- Downloaded datasets are cached and won't be re-downloaded unless deleted
- Make sure you have sufficient disk space in your chosen data directory
