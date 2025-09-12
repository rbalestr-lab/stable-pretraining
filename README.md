# stable-pretraining

[![Documentation](https://img.shields.io/badge/Documentation-blue.svg)](https://rbalestr-lab.github.io/stable-pretraining.github.io/dev/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-blue.svg)](https://github.com/rbalestr-lab/stable-pretraining/tree/main/benchmarks)
[![Test Status](https://github.com/rbalestr-lab/stable-pretraining/actions/workflows/testing.yml/badge.svg)](https://github.com/rbalestr-lab/stable-pretraining/actions/workflows/testing.yml)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/rbalestr-lab/stable-pretraining/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/rbalestr-lab/stable-pretraining/tree/main)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![WandB](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/site)


AI is moving beyond labels. Today's models learn through **self-supervision** and **multimodal alignment**, extracting knowledge from raw data to build general-purpose representations that work across tasks. These foundation models are then deployed at scale, often after finetuning, to solve tasks in zero or few shot.

`stable-pretraining` is a PyTorch framework built on top of Lightning for this new paradigm. What sets us apart is **real-time visibility into training quality** through extensive logging and monitoring. Our callback ecosystem (`OnlineProbe`, `OnlineKNN`, `RankMe`, and many more) provides insights into feature collapse, training dynamics, and downstream performance. Data flow as dictionaries through model components, metrics, and callbacks, making any intermediate value accessible and debuggable. With `stable-pretraining`: track everything, debug faster, iterate sooner.

Join our Discord: [https://discord.gg/8M6hT39X](https://discord.gg/adzpqWKM25)

## How?

To reach flexibility, scalability and stability, we rely on battle-tested third party libraries: `PyTorch`, `Lightning`, `HuggingFace`, `TorchMetrics` amongst a few others. Those dependencies allow us to focus on assembling everything into a powerful ML framework. ``stable-pretraining`` adopts a flexible and modular design for seamless integration of components from external libraries, including architectures, loss functions, evaluation metrics, and augmentations.

## Core Structure

`stable-pretraining` simplifies complex ML workflows into 4 intuitive components:

1. **data**: Your dataset must follow a dictionary-structured format where each sample is a dictionary with named fields (e.g., `{"image": ..., "label": ...}`). This ensures consistent behavior across all components. You have multiple options for creating datasets:

    - **HuggingFace datasets** (if available on the Hub):
    ```python
    import stable_pretraining as spt
    train_dataset = spt.data.HFDataset(
        path="frgfm/imagenette",
        name="160px",
        split="train",
        transform=train_transform,
    )
    ```

    - **From PyTorch datasets**:
    ```python
    train_dataset = spt.data.FromTorchDataset(
        torchvision_dataset,
        names=["image", "label"],  # Map tuple outputs to dictionary keys
        transform=train_transform,
    )
    ```

    - **Custom datasets**: Any dataset that returns dictionaries

    Once created, wrap your dataloaders in our `DataModule` for precise logging:
    ```python
    datamodule = spt.data.DataModule(train=train_dataloader, val=val_dataloader)
    ```
2. **module**: The key differentiator from PyTorch Lightning - **you only define the `forward` function**, not `training_step`! This unified approach computes losses and generates useful quantities that can be retrieved for monitoring and analysis:

    ```python
    def forward(self, batch, stage):
        out = {}
        out["embedding"] = self.backbone(batch["image"])
        if self.training:
            # Define your loss directly in forward
            proj = self.projector(out["embedding"])
            views = spt.data.fold_views(proj, batch["sample_idx"])
            out["loss"] = self.simclr_loss(views[0], views[1])
        return out
    ```

    **Key points:**
    - The `forward` method defines both the loss and any quantities to monitor
    - No need to override `training_step`, `validation_step`, etc.
    - Return a dictionary with a `"loss"` key for training
    - All model components are passed as kwargs to `spt.Module`:

    ```python
    # First define your model components
    backbone = spt.backbone.from_torchvision("resnet18")
    projector = torch.nn.Linear(512, 128)

    # Then create the module with all components
    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=forward,  # The forward function defined above
        simclr_loss=spt.losses.NTXEntLoss(temperature=0.5),
        optim={
            "optimizer": {"type": "Adam", "lr": 0.001},
            "scheduler": {"type": "CosineAnnealing"}
        }
    )
    ```

3. **callbacks**: Monitor and evaluate your models in real-time during training. Callbacks are key ingredients of `stable-pretraining`, providing rich insights without interrupting your training flow:

    ```python
    # Monitor SSL representations with a linear probe
    linear_probe = spt.callbacks.OnlineProbe(
        name="linear_probe",  # Useful for retrieving metrics and values in logging
        input="embedding",  # Which output from forward to monitor
        target="label",      # Ground truth from batch
        probe=torch.nn.Linear(512, 10),
        loss_fn=torch.nn.CrossEntropyLoss(),
        metrics={
            "top1": torchmetrics.classification.MulticlassAccuracy(10),
            "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
        },
    )

    # Track representation quality with KNN evaluation
    knn_probe = spt.callbacks.OnlineKNN(
        name="knn_probe",
        input="embedding",
        target="label",
        queue_length=20000,
        k=10,
    )
    ```

    Callbacks are powered by an intelligent queue management system that automatically shares memory between callbacks monitoring the same data thus eliminating redundant computations.

    **Why callbacks matter:**
    - **Real-time feedback**: Know if your SSL method is learning useful representations.
    - **Debugging made easy**: Catch issues like representation collapse early.
    - **Research insights**: Track multiple metrics simultaneously for deeper understanding.

4. **trainer**: Orchestrate everything together with PyTorch Lightning's `Trainer`:
    ```
    trainer = pl.Trainer(
            max_epochs=10,
            num_sanity_val_steps=1,
            callbacks=[linear_probe, knn_probe, rankme],  # Your monitoring callbacks
            precision="16-mixed",
            logger=False,
            enable_checkpointing=False,
        )
    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()
    ```
    Once configured, the `Manager` connects all components and handles the training loop with precise logging and monitoring (optional).

## Complete Example

<details>
<summary>SimCLR on CIFAR-10</summary>

This example demonstrates the key features of `stable-pretraining`: dictionary-structured data, unified forward function, and rich monitoring through callbacks.

```python
import lightning as pl
import torch
import torchmetrics
import torchvision
from torch import nn
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining.data import transforms

# Define augmentations for SimCLR (creates 2 views of each image)
simclr_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
        # Second view with slightly different augmentations
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
    ]
)

# Load CIFAR-10 and wrap in dictionary format
cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
cifar_val = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

train_dataset = spt.data.FromTorchDataset(
    cifar_train,
    names=["image", "label"],  # Convert tuple to dictionary
    transform=simclr_transform,
)

val_dataset = spt.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=transforms.Compose(
        transforms.RGB(),
        transforms.Resize((32, 32)),
        transforms.ToImage(**spt.data.static.CIFAR10),
    ),
)

# Create dataloaders with view sampling for contrastive learning
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=spt.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
    batch_size=256,
    num_workers=8,
    drop_last=True,
)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=256,
    num_workers=10,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

# Define the forward function (replaces training_step in PyTorch Lightning)
def forward(self, batch, stage):
    out = {}
    out["embedding"] = self.backbone(batch["image"])
    if self.training:
        # Project embeddings and compute contrastive loss
        proj = self.projector(out["embedding"])
        views = spt.data.fold_views(proj, batch["sample_idx"])
        out["loss"] = self.simclr_loss(views[0], views[1])
    return out

# Build model components
backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
backbone.fc = torch.nn.Identity()  # Remove classification head

projector = nn.Sequential(
    nn.Linear(512, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 256),
)

# Create the module with all components
module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=forward,
    simclr_loss=spt.losses.NTXEntLoss(temperature=0.5),
    optim={
        "optimizer": {"type": "LARS", "lr": 5, "weight_decay": 1e-6},
        "scheduler": {"type": "LinearWarmupCosineAnnealing"},
        "interval": "epoch",
    },
)

# Add callbacks for monitoring performance during training
linear_probe = spt.callbacks.OnlineProbe(
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(512, 10),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
        "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
    input_dim=512,
    k=10,
)

# Configure training
trainer = pl.Trainer(
    max_epochs=1000,
    callbacks=[knn_probe, linear_probe],  # Monitor SSL quality in real-time
    precision="16-mixed",
    logger=WandbLogger(project="cifar10-simclr"),
)

# Launch training
manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
```
</details>


## Installation

The library is not yet available on PyPI. You can install it from the source code, as follows.

1. <details><summary>conda (optional)</summary>

    First use your favorite environment manager and install your favorite pytorch version, we provide an example with conda
    ```
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    ```
    follow installation instructions... once completed, create your environment
    ```
    conda create -n my_env python=3.11
    ```
    with your environment name (here `my_env`) and your favorite Python version (here, `3.11`). Once completed, make sure to activate your environment (`conda activate my_env`) before proceeding to the next steps!
  </details>

2. Pytorch and our library (we recommend using `uv` for quicker package management):
    ```bash
    pip3 install uv
    uv pip install torch torchvision torchaudio
    uv pip install -e .  # Core dependencies only
    ```

    For optional features (vision models, experiment tracking, cluster support, etc.):
    ```bash
    uv pip install -e ".[vision,tracking]"  # Example: add vision models and wandb
    uv pip install -e ".[all]"  # Or install all optional dependencies
    ```
    See `pyproject.toml` for available dependency groups (`vision`, `tracking`, `cluster`, `visualization`, `datasets`, `extras`, `dev`, `doc`).

    If you do not want to use uv, simply remove it from the above commands.

3. API login (optional)
    ```
    wandb login
    huggingface-cli login
    ```
4. LATEX support in Matplotlib (optional)

    1.  <details>
        <summary>Install the LaTex font (Computer Modern)</summary>

        - we provide the ttf files [in the repo](assets/cm-unicode-0.7.0%202/) to make things simple
        - create your local folder (if not present) and copy the ttf files there
          - `mkdir -p ~/.local/share/fonts `
          - `cp assets/cm-unicode-0.7.0\ 2/*ttf ~/.local/share/fonts/`
        - refresh the font cache with `fc-cache -f -v`
        - validate that the fonts are listed in your system with `fc-list | grep cmu`
        - refresh matplotlib cache
          ```
          import shutil
          import matplotlib

          shutil.rmtree(matplotlib.get_cachedir())
          ```
        </details>


    2. <details>
        <summary>Install the Tex compiler (optional, if not available on your system)</summary>

        - install texlive locally following https://tug.org/texlive/quickinstall.html#running where you can use `-texdir your_path` to install to a local path (so you don't need sudo privileges)
        - follow the instructions at the end of the installation to edit the PATH variables, you can edit that variable for a conda environment with `conda env config vars set PATH=$PATH`
        - make sure inside the conde environment that you point to the right binaries e.g. `whereis latex` and `whereis mktexfmt`
        - If at some point there is an error that the file `latex.fmt` is not found. You can generate it with
          - `pdftex -ini   -jobname=latex -progname=latex -translate-file=cp227.tcx *latex.ini`
          - or (unsure) `fmtutil-sys --all`
        </details>

    3. <details>
        <summary>rc config (optional)</summary>

        ```
        font.family: serif
        font.serif: cmr10
        font.sans-serif: cmss10
        font.monospace: cmtt10

        text.usetex: True
        text.latex.preamble: \usepackage{amssymb} \usepackage{amsmath} \usepackage{bm}

        xtick.labelsize: 14
        ytick.labelsize: 14
        legend.fontsize: 14
        axes.labelsize: 16
        axes.titlesize: 16
        axes.formatter.use_mathtext: True
        ```
        which can be written to a file, e.g., `~/.config/matplotlib/matplotlibrc` or set via `rc` in your script directly. See here for more details.
        </details>

    4. <details>
        <summary>Example of matplotlib script to run for a quick test (optional)</summary>

        ```
        from matplotlib import rc
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('text', usetex=True)
        import numpy as np
        import matplotlib.pyplot as plt


        t = np.arange(0.0, 1.0 + 0.01, 0.01)
        s = np.cos(4 * np.pi * t) + 2

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(t, s)

        plt.xlabel(r'\textbf{time} (s)')
        plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
        plt.title(r"\TeX\ is Number "
                  r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
                  fontsize=16, color='gray')
        # Make room for the ridiculously large title.
        plt.subplots_adjust(top=0.8)

        plt.savefig('tex_demo')
        plt.show()
        ```
      </details>

## Ways You Can Contribute:

- If you'd like to contribute new features, bug fixes, or improvements to the documentation, please refer to our [contributing guide](https://rbalestr-lab.github.io/stable-pretraining.github.io/dev/contributing.html) for detailed instructions on how to get started.

- You can also contribute by adding new methods, datasets, or configurations that improve the current performance of a method in the [benchmark section](https://github.com/rbalestr-lab/stable-pretraining/tree/main/benchmarks).

## Contributors

Core contributors (in order of joining the project):
- [Randall Balestriero](https://github.com/RandallBalestriero)
- [Hugues Van Assel](https://github.com/huguesva)
- [Sami BuGhanem](https://github.com/sami-bg)
- [Lucas Maes](https://github.com/lucas-maes)
