"""Patient-level data splitting with automated leak detection."""

import csv
import random
from pathlib import Path

VIEWS = ("2CH", "4CH")
PHASES = ("ED", "ES")


def _find_file(patient_dir: Path, stem: str) -> Path | None:
    for ext in (".mhd", ".nii.gz", ".nii"):
        p = patient_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _parse_quality(info_path: Path) -> str:
    """Read ImageQuality field from Info_*CH.cfg."""
    if not info_path.exists():
        return "unknown"
    for line in info_path.read_text().splitlines():
        if line.startswith("ImageQuality:"):
            return line.split(":", 1)[1].strip().lower()
    return "unknown"


def _scan_patients(root: Path, logger=None) -> list[str]:
    """Return sorted patient IDs that have at least one GT file."""
    candidates = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name.startswith("patient")]
    patients = []
    for i, d in enumerate(candidates, 1):
        has_gt = any(
            _find_file(d, f"{d.name}_{v}_{p}_gt") is not None
            for v in VIEWS
            for p in PHASES
        )
        if has_gt:
            patients.append(d.name)
        if i % 100 == 0 or i == len(candidates):
            msg = "Scanned %d / %d patients — %d with GT so far"
            if logger:
                logger.info(msg, i, len(candidates), len(patients))
            else:
                print(msg % (i, len(candidates), len(patients)))
    return patients


def create_manifest(
    data_root: str,
    output_path: str,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    logger=None,
) -> dict[str, int]:
    """Build the split manifest CSV and return per-split counts.

    All images of one patient go to the same split.
    Mask-less entries are skipped and logged by the caller.
    """
    root = Path(data_root).resolve()
    patients = _scan_patients(root, logger=logger)
    if not patients:
        raise ValueError(f"No patients with ground truth found under {data_root!r}")

    rng = random.Random(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)
    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train : n_train + n_val])

    rows = []
    skipped = 0
    for pid in patients:
        pdir = root / pid
        split = "train" if pid in train_ids else ("val" if pid in val_ids else "test")
        for view in VIEWS:
            quality = _parse_quality(pdir / f"Info_{view}.cfg")
            for phase in PHASES:
                img = _find_file(pdir, f"{pid}_{view}_{phase}")
                gt = _find_file(pdir, f"{pid}_{view}_{phase}_gt")
                if img is None or gt is None:
                    skipped += 1
                    continue
                rows.append(
                    {
                        "patient_id": pid,
                        "view": view,
                        "phase": phase,
                        "quality": quality,
                        "split": split,
                        "image": str(img),
                        "mask": str(gt),
                    }
                )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patient_id", "view", "phase", "quality", "split", "image", "mask"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")}
    counts["skipped"] = skipped
    return counts


def verify_no_leak(manifest_path: str) -> None:
    """Raise AssertionError if any patient ID appears in more than one split."""
    split_patients: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            split_patients[row["split"]].add(row["patient_id"])

    overlaps = {}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        shared = split_patients[a] & split_patients[b]
        if shared:
            overlaps[f"{a}∩{b}"] = shared

    assert not overlaps, f"Patient leak detected: {overlaps}"
