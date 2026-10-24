"""Tests for Dice and HD95 on toy binary masks."""

import numpy as np
import pytest

from echo_seg.evaluation.metrics import dice_score, hd95, bootstrap_ci


def test_dice_perfect():
    mask = np.array([[1, 1, 0], [1, 0, 0]])
    assert dice_score(mask, mask) == pytest.approx(1.0)


def test_dice_no_overlap():
    pred = np.array([[1, 0], [0, 0]])
    gt = np.array([[0, 0], [0, 1]])
    assert dice_score(pred, gt) == pytest.approx(0.0)


def test_dice_both_empty():
    # Both empty → perfect agreement by convention
    empty = np.zeros((4, 4), dtype=int)
    assert dice_score(empty, empty) == pytest.approx(1.0)


def test_hd95_identical():
    mask = np.zeros((10, 10), dtype=int)
    mask[3:7, 3:7] = 1
    assert hd95(mask, mask) == pytest.approx(0.0, abs=1e-6)


def test_hd95_empty_returns_inf():
    full = np.ones((5, 5), dtype=int)
    empty = np.zeros((5, 5), dtype=int)
    assert hd95(empty, full) == float("inf")
    assert hd95(full, empty) == float("inf")


def test_hd95_spacing():
    # Two adjacent single-pixel masks with spacing (2, 2) should give distance > 0
    pred = np.zeros((5, 5), dtype=int)
    gt = np.zeros((5, 5), dtype=int)
    pred[2, 1] = 1
    gt[2, 3] = 1
    d = hd95(pred, gt, spacing=(2.0, 2.0))
    assert d > 0


def test_bootstrap_ci_bounds():
    values = list(range(100))
    mean, lo, hi = bootstrap_ci(values, n=500, seed=0)
    assert lo < mean < hi
    assert lo >= 0
    assert hi <= 100
