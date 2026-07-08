"""Unit tests for PMSN (Prior Matching for Siamese Networks)."""

from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.methods.msn import MSN
from stable_pretraining.methods.pmsn import PMSN, power_law_prior, uniform_prior

pytestmark = pytest.mark.unit

TINY_VIT = "vit_tiny_patch16_224"
B = 2
C, H, W = 3, 224, 224

SMALL_KWARGS = dict(
    encoder_name=TINY_VIT,
    n_prototypes=32,
    projector_hidden_dim=64,
    projector_bottleneck_dim=16,
    mask_ratio=0.5,
)


def _two_views() -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return (
        torch.randn(B, C, H, W, generator=g),
        torch.randn(B, C, H, W, generator=g),
    )


# --- prior helper functions -----------------------------------------------


def test_power_law_prior_sums_to_one_and_is_decreasing():
    prior = power_law_prior(16, tau=0.25)
    assert prior.shape == (16,)
    assert torch.allclose(prior.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.all(prior[:-1] >= prior[1:])


def test_power_law_prior_tau_zero_is_uniform():
    prior = power_law_prior(16, tau=0.0)
    assert torch.allclose(prior, uniform_prior(16), atol=1e-6)


def test_uniform_prior_is_flat_and_normalised():
    prior = uniform_prior(8)
    assert prior.shape == (8,)
    assert torch.allclose(prior, torch.full((8,), 1.0 / 8))
    assert torch.allclose(prior.sum(), torch.tensor(1.0))


# --- prior registration on the module --------------------------------------


def test_pmsn_default_prior_is_power_law():
    model = PMSN(**SMALL_KWARGS, tau=0.5)
    expected = power_law_prior(32, tau=0.5)
    assert torch.allclose(model._prior, expected, atol=1e-6)


def test_pmsn_uniform_prior_option():
    model = PMSN(**SMALL_KWARGS, prior="uniform")
    assert torch.allclose(model._prior, uniform_prior(32), atol=1e-6)


def test_pmsn_custom_prior_tensor_is_normalised():
    raw = torch.arange(1, 33, dtype=torch.float)
    model = PMSN(**SMALL_KWARGS, prior=raw)
    assert torch.allclose(model._prior, raw / raw.sum(), atol=1e-6)
    assert model.prior_type == "custom"


def test_pmsn_custom_prior_wrong_shape_raises():
    bad = torch.ones(10)
    with pytest.raises(AssertionError, match="prior tensor must be shape"):
        PMSN(**SMALL_KWARGS, prior=bad)


def test_pmsn_unknown_prior_string_raises():
    with pytest.raises(ValueError, match="Unknown prior type"):
        PMSN(**SMALL_KWARGS, prior="not_a_real_prior")


# --- forward / backward -----------------------------------------------------


def test_pmsn_forward_backward():
    model = PMSN(**SMALL_KWARGS)
    model.train()
    v1, v2 = _two_views()
    output = model(view1=v1, view2=v2)

    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert output.kl_loss is not None
    assert torch.isfinite(output.kl_loss)

    output.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert any(g.abs().sum() > 0 for g in grads)


def test_pmsn_kl_loss_matches_manual_computation():
    torch.manual_seed(0)
    model = PMSN(**SMALL_KWARGS)
    model.train()
    v1, v2 = _two_views()
    output = model(view1=v1, view2=v2)

    mean_p = F.softmax(output.student_logits / model.temperature_student, dim=-1).mean(
        dim=0
    )
    mean_p = mean_p.clamp(min=1e-8)
    prior = model._prior.clamp(min=1e-8)
    expected_kl = (mean_p * (mean_p.log() - prior.log())).sum()

    assert torch.allclose(output.kl_loss, expected_kl, atol=1e-6)


def test_pmsn_prior_weight_zero_matches_parent_msn_ce():
    """With prior_weight=0, PMSN's loss must reduce to MSN's CE-only loss.

    PMSN disables MSN's own me_max term (me_max_weight=0.0) and adds
    prior_weight * KL instead, so zeroing prior_weight should reproduce
    the parent's CE-only forward exactly given the same random mask.
    """
    model = PMSN(**SMALL_KWARGS, prior_weight=0.0)
    model.train()
    v1, v2 = _two_views()

    torch.manual_seed(42)
    pmsn_out = model(view1=v1, view2=v2)

    torch.manual_seed(42)
    msn_out = MSN.forward(model, view1=v1, view2=v2)

    assert torch.allclose(pmsn_out.loss, msn_out.loss, atol=1e-6)


def test_pmsn_prior_weight_scales_kl_contribution():
    model = PMSN(**SMALL_KWARGS, prior_weight=2.0)
    model.train()
    v1, v2 = _two_views()

    torch.manual_seed(7)
    out = model(view1=v1, view2=v2)

    torch.manual_seed(7)
    parent_out = MSN.forward(model, view1=v1, view2=v2)

    assert torch.allclose(out.loss, parent_out.loss + 2.0 * out.kl_loss, atol=1e-6)


def test_pmsn_eval_mode_delegates_to_parent_without_kl():
    model = PMSN(**SMALL_KWARGS)
    model.eval()
    v1, _ = _two_views()

    with torch.no_grad():
        output = model(images=v1)

    assert output.kl_loss is None
    assert torch.isfinite(output.loss)
    assert output.embedding is not None and output.embedding.shape[0] == B
