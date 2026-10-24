"""Tests for augmentation pipeline — critical: no horizontal flip allowed."""

import numpy as np

from echo_seg.data.augmentation import build_train_transform, build_val_transform


def _asymmetric_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Float32 image with a distinctive left-side marker."""
    img = np.zeros((h, w), dtype=np.float32)
    img[:, : w // 4] = 1.0  # left quarter is bright
    return img


def test_no_horizontal_flip():
    """The training transform must never flip the image horizontally.

    A deterministic asymmetric image is passed 50 times; if a flip occurred,
    the right side would become bright. We check the left mean stays higher.
    """
    transform = build_train_transform(image_size=(64, 64))
    img = _asymmetric_image()
    mask = np.zeros((64, 64), dtype=np.int32)

    left_mean_total = 0.0
    right_mean_total = 0.0
    n = 50
    for _ in range(n):
        aug = transform(image=img.copy(), mask=mask.copy())
        out = aug["image"].numpy()[0]  # (1, H, W) → (H, W)
        left_mean_total += out[:, : 32].mean()
        right_mean_total += out[:, 32:].mean()

    assert left_mean_total > right_mean_total, (
        "Left mean should exceed right mean across all augmented copies "
        "when no horizontal flip is applied."
    )


def test_val_transform_shape():
    img = np.random.rand(128, 100).astype(np.float32)
    mask = np.zeros((128, 100), dtype=np.int32)
    transform = build_val_transform(image_size=(256, 256))
    aug = transform(image=img, mask=mask)
    assert aug["image"].shape == (1, 256, 256)
    assert aug["mask"].shape == (256, 256)


def test_train_transform_shape():
    img = np.random.rand(200, 180).astype(np.float32)
    mask = np.zeros((200, 180), dtype=np.int32)
    transform = build_train_transform(image_size=(256, 256))
    aug = transform(image=img, mask=mask)
    assert aug["image"].shape == (1, 256, 256)
    assert aug["mask"].shape == (256, 256)
