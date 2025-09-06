"""Online latent space visualization callback with dimensionality reduction.

This callback learns a 2D projection of high-dimensional features while preserving
neighborhood structure using a contrastive loss between high-D and low-D similarities.
"""

from functools import partial
from typing import Dict, Literal, Optional, Union

import numpy as np
import torch
from hydra.utils import instantiate
from lightning.pytorch import LightningModule, Trainer
from loguru import logger as logging
from torch import Tensor

from ..utils.distance_metrics import compute_pairwise_distances_chunked

from .queue import find_or_create_queue_callback
from .utils import TrainableCallback


class LatentViz(TrainableCallback):
    """Online latent visualization callback with neighborhood-preserving dimensionality reduction.

    This callback learns a 2D projection that preserves neighborhood structure from
    high-dimensional features. It uses a contrastive loss that attracts neighbors
    and repels non-neighbors in the 2D space.

    The loss function is:
        L = -∑_{ij} P_{ij} log Q_{ij} + ∑_{i,j ∈ Neg(i)} log(1 - Q_{ij})

    where:
        - P_{ij} is the high-D neighborhood graph (based on k-NN)
        - Q_{ij} is the similarity in the learned 2D space
        - Neg(i) is the set of negative samples for point i

    Args:
        name: Unique identifier for this callback instance.
        input: Key in batch dict containing input features to visualize.
        target: Optional key in batch dict containing labels for coloring plots.
            If None, points will be plotted without color coding.
        projection: The projection module to train (maps high-D to 2D). Can be:
            - nn.Module instance
            - callable that returns a module
            - Hydra config to instantiate
        queue_length: Size of the circular buffer for features.
        k_neighbors: Number of nearest neighbors for building P matrix.
        n_negatives: Number of negative samples per positive pair.
        optimizer: Optimizer configuration. If None, uses Adam (recommended for DR tasks).
        scheduler: Learning rate scheduler configuration. If None, uses ConstantLR.
        accumulate_grad_batches: Number of batches to accumulate gradients.
        update_interval: Update projection network every N training batches (default: 10).
        warmup_epochs: Number of epochs to wait before starting projection training (default: 0).
            Allows main model to stabilize before learning 2D projections.
        distance_metric: Metric for computing distances in high-D space.
        plot_interval: Interval (in epochs) for plotting 2D visualization.
        save_dir: Optional directory to save plots. If None, saves to 'latent_viz_{name}'.
        input_dim: Expected dimensionality of input features (for queue).
    """

    def __init__(
        self,
        name: str,
        input: str,
        target: Optional[str],
        projection: torch.nn.Module,
        queue_length: int = 2048,
        k_neighbors: int = 15,
        n_negatives: int = 5,
        optimizer: Optional[Union[str, dict, partial, torch.optim.Optimizer]] = None,
        scheduler: Optional[
            Union[str, dict, partial, torch.optim.lr_scheduler.LRScheduler]
        ] = None,
        accumulate_grad_batches: int = 1,
        update_interval: int = 10,
        warmup_epochs: int = 0,
        distance_metric: Literal["euclidean", "cosine"] = "euclidean",
        plot_interval: int = 10,
        save_dir: Optional[str] = None,
        input_dim: Optional[Union[int, tuple, list]] = None,
    ):
        super().__init__(
            name=name,
            optimizer=optimizer,
            scheduler=scheduler,
            accumulate_grad_batches=accumulate_grad_batches,
        )

        self.input = input
        self.target = target
        self.queue_length = queue_length

        self.k_neighbors = k_neighbors
        self.n_negatives = n_negatives
        self.update_interval = update_interval
        self.warmup_epochs = warmup_epochs
        self.distance_metric = distance_metric
        self.plot_interval = plot_interval
        self.save_dir = save_dir

        if input_dim is not None and isinstance(input_dim, (list, tuple)):
            import numpy as np

            input_dim = int(np.prod(input_dim))
        self.input_dim = input_dim

        self._projection_config = projection

        # Will be initialized in setup
        self._input_queue = None
        self._target_queue = None

        logging.info(f"Initialized LatentViz callback: {name}")
        logging.info(f"  - Input: {input}")
        logging.info(
            f"  - Target: {target if target else 'None (no labels for coloring)'}"
        )
        logging.info(f"  - Queue length: {queue_length}")
        logging.info(f"  - K neighbors: {k_neighbors}")
        logging.info(f"  - Negative samples: {n_negatives}")
        logging.info(f"  - Update interval: {update_interval} batches")
        logging.info(f"  - Warmup epochs: {warmup_epochs}")
        logging.info(f"  - Accumulate grad batches: {accumulate_grad_batches}")

    def _initialize_module(self, pl_module: LightningModule) -> torch.nn.Module:
        """Initialize the projection module from configuration."""
        if isinstance(self._projection_config, torch.nn.Module):
            projection_module = self._projection_config
        elif callable(self._projection_config):
            projection_module = self._projection_config()
        else:
            projection_module = instantiate(self._projection_config, _convert_="object")

        return projection_module

    def setup_optimizer(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize optimizer - default to AdamW for dimensionality reduction tasks."""
        if self._optimizer_config is None:
            # Use AdamW by default for LatentViz (better weight decay handling)
            logging.info(
                f"{self.name}: Using default AdamW optimizer for dimensionality reduction"
            )
            self.optimizer = torch.optim.AdamW(
                self.module.parameters(),
                lr=1e-3,  # Good default for AdamW
                weight_decay=1e-2,  # Higher weight decay works well with AdamW
                betas=(0.9, 0.999),  # Standard Adam betas
            )
        else:
            # Use explicitly provided optimizer config
            from stable_pretraining.optim.utils import create_optimizer

            self.optimizer = create_optimizer(
                self.module.parameters(), self._optimizer_config
            )

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup module, optimizer, scheduler, and queues."""
        super().setup(trainer, pl_module, stage)

        if stage != "fit":
            return

        # Find or create queues (same as knn.py)
        self._input_queue = find_or_create_queue_callback(
            trainer,
            self.input,
            self.queue_length,
            self.input_dim,
            torch.float32 if self.input_dim is not None else None,
            gather_distributed=True,
            create_if_missing=True,
        )
        logging.info(f"{self.name}: Using queue for input '{self.input}'")

        # Only create target queue if target is specified
        if self.target is not None:
            self._target_queue = find_or_create_queue_callback(
                trainer,
                self.target,
                self.queue_length,
                None,  # No specific dimension for targets
                torch.long,
                gather_distributed=True,
                create_if_missing=True,
            )
            logging.info(f"{self.name}: Using queue for target '{self.target}'")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
    ) -> None:
        """Perform projection network training step."""
        # Skip training during warmup period
        if trainer.current_epoch < self.warmup_epochs:
            if batch_idx == 0:  # Log once per epoch
                logging.info(
                    f"{self.name}: Warmup period - skipping projection training "
                    f"(epoch {trainer.current_epoch + 1}/{self.warmup_epochs})"
                )
            return

        # Only update every N batches to reduce computational overhead
        if batch_idx % self.update_interval != 0:
            return

        # Get cached features directly from the shared queue
        # Access the raw queue from the class-level registry
        from .queue import OnlineQueue

        shared_queue = OnlineQueue._shared_queues.get(self.input)
        if shared_queue is None:
            return

        cached_features = shared_queue.get()
        if cached_features is None or len(cached_features) == 0:
            return

        self.module.train()

        with torch.enable_grad():
            # Detach features to prevent gradients flowing to main model
            x = cached_features.detach()

            proj_dtype = next(self.module.parameters()).dtype
            if x.dtype != proj_dtype:
                x = x.to(proj_dtype)

            z_2d = self.module(x)
            loss = self._compute_loss(x, z_2d)
            loss = loss / self.accumulate_grad_batches
            loss.backward()

        loss_value = loss.item() * self.accumulate_grad_batches
        pl_module.log(
            f"train/{self.name}_loss",
            loss_value,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.optimizer_step(batch_idx, trainer)

    def _compute_loss(
        self,
        x_high: Tensor,
        z_2d: Tensor,
    ) -> Tensor:
        """Compute the neighborhood-preserving loss.

        Loss = -∑_{ij} P_{ij} log Q_{ij} + ∑_{i,j ∈ Neg(i)} log(1 - Q_{ij})

        Args:
            x_high: High-dimensional features [N, D]
            z_2d: 2D projections [N, 2]
        """
        n_samples = x_high.size(0)
        device = x_high.device

        chunk_size = 256 if n_samples > 1000 else -1
        high_d_distances = compute_pairwise_distances_chunked(
            x_high, x_high, metric=self.distance_metric, chunk_size=chunk_size
        )

        k_actual = min(self.k_neighbors, n_samples - 1)  # Exclude self
        high_d_distances.fill_diagonal_(float("inf"))  # Exclude self
        _, nn_indices = high_d_distances.topk(k=k_actual, dim=1, largest=False)

        # Compute 2D similarities (Q matrix) using Student-t kernel - chunked for memory efficiency
        # Student-t kernel: q_ij = (1 + ||z_i - z_j||^2)^(-1)
        z_distances_sq = compute_pairwise_distances_chunked(
            z_2d, z_2d, metric="squared_euclidean", chunk_size=chunk_size
        )
        q_matrix = 1.0 / (1.0 + z_distances_sq)

        # Set diagonal to 0 (no self-similarity)
        mask = torch.ones_like(q_matrix).detach()
        mask.fill_diagonal_(0)
        q_matrix = q_matrix * mask

        # Normalize Q to [0, 1] range
        q_matrix = q_matrix / (q_matrix + 1)

        # Compute attraction loss for positive pairs (neighbors) - vectorized
        row_indices = (
            torch.arange(n_samples, device=device).unsqueeze(1).expand(-1, k_actual)
        )
        q_neighbors = q_matrix[row_indices, nn_indices]
        attraction_loss = -(q_neighbors + 1e-10).log().mean()

        # Compute repulsion loss for negative pairs - uniform sampling
        n_negatives_per_point = self.n_negatives * k_actual
        neg_indices = torch.randint(
            0, n_samples, (n_samples, n_negatives_per_point), device=device
        )
        row_indices_neg = (
            torch.arange(n_samples, device=device)
            .unsqueeze(1)
            .expand(-1, n_negatives_per_point)
        )
        q_negatives = q_matrix[row_indices_neg, neg_indices]
        repulsion_loss = -((1 - q_negatives).clamp(min=1e-10).log()).mean()

        total_loss = attraction_loss + repulsion_loss

        return total_loss

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Plot 2D visualization at specified intervals."""
        # Skip visualization during warmup period
        if trainer.current_epoch < self.warmup_epochs:
            logging.info(
                f"{self.name}: Warmup period - skipping visualization "
                f"(epoch {trainer.current_epoch + 1}/{self.warmup_epochs})"
            )
            return

        # Plot visualization at intervals
        if trainer.current_epoch % self.plot_interval != 0:
            return

        # Get cached features
        cached_features = self._input_queue.data
        if cached_features is None or cached_features.numel() == 0:
            return

        # Get cached labels if available
        cached_labels = None
        if self._target_queue is not None:
            cached_labels = self._target_queue.data
            if cached_labels is not None and cached_labels.numel() == 0:
                cached_labels = None

        # Project to 2D
        self.module.eval()
        with torch.no_grad():
            # Ensure correct dtype
            proj_dtype = next(self.module.parameters()).dtype
            if cached_features.dtype != proj_dtype:
                cached_features = cached_features.to(proj_dtype)

            z_2d = self.module(cached_features)

        # Create visualization
        self._plot_2d_embeddings(z_2d, cached_labels, trainer.current_epoch, trainer)

    def _plot_2d_embeddings(
        self, z_2d: Tensor, labels: Optional[Tensor], epoch: int, trainer: Trainer
    ) -> None:
        """Save 2D embeddings to file and log to experiment tracker."""
        import os

        # Save coordinates to NPZ file
        z_2d_np = z_2d.cpu().numpy()
        labels_np = labels.cpu().numpy() if labels is not None else None
        if self.save_dir is not None:
            save_dir = self.save_dir
        else:
            save_dir = f"latent_viz_{self.name}"
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"epoch_{epoch:04d}.npz")
        save_data = {"coordinates": z_2d_np}
        if labels_np is not None:
            save_data["labels"] = labels_np
        np.savez_compressed(save_path, **save_data)

        logging.info(f"{self.name}: Saved 2D coordinates to {save_path}")

        # Log to experiment tracker if available
        try:
            from lightning.pytorch.loggers import WandbLogger

            if isinstance(trainer.logger, WandbLogger):
                import wandb

                # Create WandB-specific table only (no direct scatter logging)
                if labels_np is not None:
                    data = np.column_stack([z_2d_np, labels_np.astype(int)])
                    columns = ["x", "y", "class"]
                else:
                    data = z_2d_np
                    columns = ["x", "y"]

                table = wandb.Table(columns=columns, data=data.tolist())

                # Log table - will overwrite previous epoch's table
                wandb.log(
                    {
                        f"{self.name}/2d_latent_table": table,
                        f"{self.name}/current_epoch": epoch,
                    }
                )

                logging.info(
                    f"{self.name}: Logged latent table to experiment tracker at epoch {epoch}"
                )
        except ImportError:
            logging.debug(
                f"{self.name}: WandB not installed, skipping visualization logging"
            )
        except Exception as e:
            logging.error(f"{self.name}: Failed to log visualization: {e}")

    @property
    def projection_module(self):
        """Alias for self.module for backward compatibility."""
        return self.module
