"""PMSN: Prior Matching for Siamese Networks.

Extension of MSN that replaces the implicit uniform feature prior with an
arbitrary (e.g. power-law) prior via KL-divergence regularisation, enabling
better representation learning on class-imbalanced datasets.

References:
    Assran et al. "The Hidden Uniform Cluster Prior in Self-Supervised Learning."
    ICLR 2023. https://arxiv.org/abs/2210.07277
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_pretraining.methods.msn import MSN, MSNOutput


@dataclass
class PMSNOutput(MSNOutput):
    """MSNOutput extended with a KL divergence term for logging."""

    kl_loss: Optional[torch.Tensor] = None


def power_law_prior(n_prototypes: int, tau: float, device=None) -> torch.Tensor:
    """Compute a power-law prior: [π]_k ∝ 1 / k^τ  (1-indexed, L1-normalised).

    Args:
        n_prototypes: Number of prototypes K.
        tau: Power-law exponent τ > 0.
            τ=0 approaches uniform; larger τ yields a more long-tailed distribution.
        device: Target device.

    Returns:
        (K,) tensor that sums to 1.
    """
    k = torch.arange(1, n_prototypes + 1, dtype=torch.float, device=device)
    p = 1.0 / k.pow(tau)
    return p / p.sum()


def uniform_prior(n_prototypes: int, device=None) -> torch.Tensor:
    """Uniform distribution over K prototypes."""
    return torch.full((n_prototypes,), 1.0 / n_prototypes, device=device)


class PMSN(MSN):
    """PMSN: Prior Matching for Siamese Networks.

    Identical architecture to MSN, with the loss replaced as follows:
        MSN  : CE(student, teacher) + λ · H_mean_entropy  (uniform prior)
        PMSN : CE(student, teacher) + λ · KL(p̄ ‖ π)      (arbitrary prior)

    :param prior: "power_law" | "uniform" | (K,) tensor.
        - "power_law" (default): uses pPL(τ), controlled by the tau argument.
        - "uniform": equivalent to MSN (useful for ablation).
        - Tensor: directly specify a custom prior (e.g. true class distribution of iNat18).
    :param tau: Power-law exponent (only used when prior="power_law", default 0.25).
    :param prior_weight: Weight λ for the KL term (default 1.0).
    :param encoder_name: timm ViT name (default ``"vit_small_patch16_224"``).
    :param projector_hidden_dim: Hidden dim (default 2048).
    :param projector_bottleneck_dim: Bottleneck dim (default 256).
    :param n_prototypes: Prototype count (default 1024).
    :param mask_ratio: Patch mask ratio for the *student* (default 0.6).
    :param temperature_student: Student softmax temperature (default 0.1).
    :param temperature_teacher: Teacher softmax temperature (default 0.025).
    :param ema_decay_start: Initial backbone/head EMA (default 0.996).
    :param ema_decay_end: Final EMA (default 1.0).
    :param image_size: Input size (default 224).
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        prior: Union[str, torch.Tensor] = "power_law",
        tau: float = 0.25,
        prior_weight: float = 1.0,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_hidden_dim: int = 2048,
        projector_bottleneck_dim: int = 256,
        n_prototypes: int = 1024,
        mask_ratio: float = 0.6,
        temperature_student: float = 0.1,
        temperature_teacher: float = 0.025,
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
        image_size: int = 224,
        pretrained: bool = False,
    ):
        super().__init__(
            encoder_name=encoder_name,
            projector_hidden_dim=projector_hidden_dim,
            projector_bottleneck_dim=projector_bottleneck_dim,
            n_prototypes=n_prototypes,
            mask_ratio=mask_ratio,
            temperature_student=temperature_student,
            temperature_teacher=temperature_teacher,
            me_max_weight=0.0,  # Disable MSN's me_max; replaced by PMSN's KL term
            ema_decay_start=ema_decay_start,
            ema_decay_end=ema_decay_end,
            image_size=image_size,
            pretrained=pretrained,
        )

        self.prior_type = prior if isinstance(prior, str) else "custom"
        self.tau = tau
        self.prior_weight = prior_weight
        self.n_prototypes = n_prototypes

        # Register prior as a buffer so .to(device) and state_dict are handled automatically
        if isinstance(prior, torch.Tensor):
            assert prior.shape == (n_prototypes,), (
                f"prior tensor must be shape ({n_prototypes},), got {tuple(prior.shape)}"
            )
            self.register_buffer("_prior", prior / prior.sum())
        elif prior == "uniform":
            self.register_buffer("_prior", uniform_prior(n_prototypes))
        elif prior == "power_law":
            self.register_buffer("_prior", power_law_prior(n_prototypes, tau))
        else:
            raise ValueError(
                f"Unknown prior type: {prior!r}. Use 'power_law', 'uniform', or a Tensor."
            )

    def _prior_kl(self, student_logits: torch.Tensor) -> torch.Tensor:
        """Compute KL(p̄ ‖ π).

        p̄ = mean_i softmax(s_logits_i / T_student)
        KL(p̄ ‖ π) = Σ_k p̄_k log(p̄_k / π_k)
        """
        mean_p = F.softmax(student_logits / self.temperature_student, dim=-1).mean(
            dim=0
        )
        mean_p = mean_p.clamp(min=1e-8)
        prior = self._prior.clamp(min=1e-8)
        return (mean_p * (mean_p.log() - prior.log())).sum()

    def forward(
        self,
        view1: Optional[torch.Tensor] = None,
        view2: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> PMSNOutput:
        # Eval / single-image: delegate to parent
        if images is not None or (view1 is not None and view2 is None):
            out = super().forward(view1=view1, view2=view2, images=images)
            return PMSNOutput(
                loss=out.loss,
                embedding=out.embedding,
                student_logits=out.student_logits,
                teacher_logits=out.teacher_logits,
                kl_loss=None,
            )

        # Training: parent forward gives CE-only loss (me_max_weight=0)
        parent_out = super().forward(view1=view1, view2=view2)

        # Add KL(p̄ ‖ π) regularisation term
        kl = self._prior_kl(parent_out.student_logits)
        loss = parent_out.loss + self.prior_weight * kl

        return PMSNOutput(
            loss=loss,
            embedding=parent_out.embedding,
            student_logits=parent_out.student_logits,
            teacher_logits=parent_out.teacher_logits,
            kl_loss=kl.detach(),
        )
