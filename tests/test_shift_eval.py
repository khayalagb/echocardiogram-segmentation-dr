"""Tests for test-time corruption functions in shift_eval."""

import numpy as np
import pytest
import torch

from echo_seg.evaluation.shift_eval import CORRUPTIONS


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
