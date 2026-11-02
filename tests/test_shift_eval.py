"""Tests for test-time corruption functions in shift_eval.

Covers all seven conditions: shape, [0,1] range, determinism under fixed seed,
identity of clean, and that every corruption changes the image.
"""

import numpy as np
import pytest
import torch

from echo_seg.evaluation.shift_eval import (
    CORRUPTIONS,
    CORRUPTIONS_HELD_OUT,
    CORRUPTIONS_IN_DR_TRAINING,
)


@pytest.fixture()
def img() -> torch.Tensor:
    """Fixed (1, 1, 256, 256) float32 tensor in [0, 1]."""
    rng = torch.Generator()
    rng.manual_seed(0)
    return torch.rand(1, 1, 256, 256, generator=rng)


@pytest.mark.parametrize("name,fn", [(k, v) for k, v in CORRUPTIONS.items() if v is not None])
def test_output_shape(name, fn, img):
    np.random.seed(42)
    out = fn(img)
    assert out.shape == img.shape, f"{name}: shape changed"


@pytest.mark.parametrize("name,fn", [(k, v) for k, v in CORRUPTIONS.items() if v is not None])
def test_output_range(name, fn, img):
    np.random.seed(42)
    out = fn(img)
    assert out.min() >= 0.0, f"{name}: min below 0"
    assert out.max() <= 1.0, f"{name}: max above 1"


@pytest.mark.parametrize("name,fn", [(k, v) for k, v in CORRUPTIONS.items() if v is not None])
def test_determinism(name, fn, img):
    """Same numpy seed → same output."""
    np.random.seed(42)
    out1 = fn(img)
    np.random.seed(42)
    out2 = fn(img)
    assert torch.allclose(out1, out2), f"{name}: output differs across identical seeds"


def test_clean_is_identity(img):
    assert CORRUPTIONS["clean"] is None


def test_corruptions_differ_from_input(img):
    """Every non-trivial corruption should change the image."""
    for name, fn in CORRUPTIONS.items():
        if fn is None:
            continue
        np.random.seed(42)
        out = fn(img)
        assert not torch.allclose(out, img), f"{name}: output identical to input"


def test_speckle_corr_multiplier_stats(img):
    """speckle_corr multiplier must have mean≈1, std≈0.30, and produce a visible shift."""
    from echo_seg.evaluation.shift_eval import _speckle_corr
    # Use a uniform image so multiplier = output / input exactly
    uniform = torch.full_like(img, 0.5)
    np.random.seed(42)
    out = _speckle_corr(uniform)
    multiplier = out / uniform
    assert abs(multiplier.mean().item() - 1.0) < 0.05, "speckle_corr multiplier mean not ≈1"
    assert abs(multiplier.std().item() - 0.30) < 0.05, "speckle_corr multiplier std not ≈0.30"
    mae = (out - uniform).abs().mean().item()
    assert mae > 0.05, f"speckle_corr MAE too small ({mae:.4f}): corruption is near-no-op"


def test_speckle_multiplier_mean(img):
    """speckle multiplier must have mean≈1 (Rayleigh σ=0.798 → mean≈1.0)."""
    from echo_seg.evaluation.shift_eval import _speckle
    uniform = torch.full_like(img, 0.5)
    np.random.seed(42)
    out = _speckle(uniform)
    multiplier = out / uniform
    assert abs(multiplier.mean().item() - 1.0) < 0.05, "speckle multiplier mean not ≈1"


def test_held_out_not_in_dr_training():
    """low_res and speckle_corr must not appear in DR training augmentations."""
    assert CORRUPTIONS_HELD_OUT.isdisjoint(CORRUPTIONS_IN_DR_TRAINING)
    assert "low_res" in CORRUPTIONS_HELD_OUT
    assert "speckle_corr" in CORRUPTIONS_HELD_OUT


def test_all_conditions_covered():
    """Every key in CORRUPTIONS is classified as either DR-training or held-out (except clean)."""
    non_clean = {k for k in CORRUPTIONS if k != "clean"}
    classified = CORRUPTIONS_IN_DR_TRAINING | CORRUPTIONS_HELD_OUT
    assert non_clean == classified, f"Unclassified: {non_clean - classified}"
