"""Tests for patient-level split and leak detection."""

import csv
import tempfile
from pathlib import Path

import pytest

from echo_seg.data.split import create_manifest, verify_no_leak


def _make_fake_dataset(root: Path, n_patients: int) -> None:
    """Create minimal fake CAMUS structure with stub MHD files."""
    for i in range(1, n_patients + 1):
        pid = f"patient{i:04d}"
        pdir = root / pid
        pdir.mkdir()
        for view in ("2CH", "4CH"):
            (pdir / f"Info_{view}.cfg").write_text("ImageQuality: Good\n")
            for phase in ("ED", "ES"):
                (pdir / f"{pid}_{view}_{phase}.mhd").touch()
                (pdir / f"{pid}_{view}_{phase}_gt.mhd").touch()


def test_split_counts():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "data"
        root.mkdir()
        _make_fake_dataset(root, 100)
        manifest = Path(tmpdir) / "manifest.csv"
        counts = create_manifest(str(root), str(manifest), seed=42, train_frac=0.70, val_frac=0.15)
        total = counts["train"] + counts["val"] + counts["test"]
        # 100 patients × 4 images each = 400 rows (minus any skipped)
        assert total == 400
        assert counts["train"] > counts["val"]
        assert counts["val"] > 0
        assert counts["test"] > 0


def test_no_leak():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "data"
        root.mkdir()
        _make_fake_dataset(root, 50)
        manifest = Path(tmpdir) / "manifest.csv"
        create_manifest(str(root), str(manifest), seed=0)
        verify_no_leak(str(manifest))  # must not raise


def test_leak_detected():
    """verify_no_leak raises when a patient is duplicated across splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_manifest = Path(tmpdir) / "bad.csv"
        rows = [
            {"patient_id": "p01", "view": "2CH", "phase": "ED", "quality": "good",
             "split": "train", "image": "a", "mask": "b"},
            {"patient_id": "p01", "view": "2CH", "phase": "ES", "quality": "good",
             "split": "test", "image": "c", "mask": "d"},  # same patient, different split
        ]
        with open(bad_manifest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with pytest.raises(AssertionError, match="Patient leak"):
            verify_no_leak(str(bad_manifest))
