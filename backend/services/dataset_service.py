"""Dataset services: prepare, validate and inspect datasets.

``scripts/prepare_dataset.py`` and the ``/api/training/dataset/*`` endpoints both call into
here, so the CLI and the web UI can never drift apart.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.datasets.preprocessing import (
    DatasetConfig, build_sample, prepare_metadata, validate_layout, write_sample, garment_mask_from_alpha,
)
from backend.services.sample_data import sample_dataset_available, write_sample_dataset
from backend.utils.errors import DatasetError
from backend.utils.image_utils import load_image
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatasetService:
    """High-level dataset operations used by the CLI and the API."""

    def __init__(self, datasets_dir: str | Path = "datasets") -> None:
        self.root = Path(datasets_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------------
    def list_datasets(self) -> List[Dict[str, Any]]:
        """Every dataset folder found under ``datasets/`` with its validation report."""
        out: List[Dict[str, Any]] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            report = validate_layout(child)
            metadata = report.get("metadata", {})
            out.append({
                "name": child.name,
                "path": str(child),
                "counts": {split: report["splits"].get(split, {}).get("pairs", 0) for split in ("train", "val", "test")},
                "ok": report["ok"],
                "issues": report["issues"],
                "warnings": report["warnings"][:6],
                "metadata": metadata,
                "origin": metadata.get("origin", "unknown"),
            })
        return out

    def validate(self, name: str, strict: bool = False) -> Dict[str, Any]:
        path = self.resolve(name)
        return validate_layout(path, strict=strict)

    def resolve(self, name_or_path: str) -> Path:
        """Accept an absolute path, a path relative to the project, or a dataset name."""
        candidate = Path(name_or_path)
        if candidate.exists():
            return candidate
        named = self.root / name_or_path
        if named.exists():
            return named
        raise DatasetError(f"Dataset '{name_or_path}' was not found (looked in {self.root}).")

    # ---------------------------------------------------------------------------------
    def generate_samples(self, count: Dict[str, int] | None = None, size: int = 512, seed: int = 11, overwrite: bool = True) -> Dict[str, Any]:
        """Create/refresh the built-in synthetic dataset (sample-data mode)."""
        target = self.root / "samples"
        if target.exists() and overwrite:
            shutil.rmtree(target, ignore_errors=True)
        counts = count or {"train": 24, "val": 6, "test": 6}
        summary = write_sample_dataset(target, train=counts.get("train", 24), val=counts.get("val", 6), test=counts.get("test", 6), size=size, seed=seed)
        report = validate_layout(target)
        return {"ok": True, "path": str(target), "counts": summary, "report": report}

    def sample_available(self) -> bool:
        return sample_dataset_available(self.root / "samples")

    # ---------------------------------------------------------------------------------
    def convert_viton_folder(
        self,
        source: str | Path,
        name: str = "viton_hd",
        split_source: Optional[Dict[str, str]] = None,
        config: Optional[DatasetConfig] = None,
        progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Convert an existing VITON-HD/DressCode-style folder into VestiAI's layout.

        Expects (per split) at least ``image/`` and ``cloth/``; ``cloth-mask``, ``agnostic``
        and ``pose`` are used when present and generated otherwise. Missing splits are
        created by re-splitting the training set with ``val_fraction``/``test_fraction``.
        """
        source = Path(source)
        if not source.exists():
            raise DatasetError(f"Source dataset {source} does not exist.")
        config = config or DatasetConfig(root=str(self.root / name))
        target = self.root / name
        target.mkdir(parents=True, exist_ok=True)

        split_source = split_source or {}
        counts: Dict[str, int] = {}

        # 1) direct split folders
        existing = [split for split in ("train", "val", "test") if (source / split).exists()]
        if existing:
            for split in existing:
                source_dir = source / split
                target_split = split_source.get(split, split)
                count = self._copy_or_convert_split(source_dir, target / target_split, config, progress)
                counts[target_split] = count
        else:
            # 2) flat VITON-HD (source/train style) — treat everything as train, then re-split
            count = self._copy_or_convert_split(source, target / "train", config, progress)
            counts["train"] = count

        # 3) create val/test by splitting the training pairs if needed
        if counts.get("train") and not (target / "val" / "pairs.txt").exists():
            self.split_dataset(target, config.val_fraction, config.test_fraction, config.seed)
            for split in ("train", "val", "test"):
                counts[split] = self._count_pairs(target / split)

        metadata = prepare_metadata(target, extra={
            "origin": "converted",
            "source": str(source),
            "notes": "Converted by VestiAI from a VITON-HD/DressCode-compatible folder.",
        })
        return {"ok": True, "path": str(target), "counts": counts, "metadata": metadata,
                "report": validate_layout(target, strict=False)}

    def _copy_or_convert_split(self, source_dir: Path, target_dir: Path, config: DatasetConfig, progress: Optional[Any]) -> int:
        """Copy a split, generating any missing derivative (mask/agnostic/pose)."""
        image_dir = source_dir / "image"
        cloth_dir = source_dir / "cloth"
        if not image_dir.exists() or not cloth_dir.exists():
            raise DatasetError(f"{source_dir} must contain 'image/' and 'cloth/' folders.")

        pairs_path = source_dir / "pairs.txt"
        entries: List[Tuple[str, str, str]] = []
        if pairs_path.exists():
            for line in pairs_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    entries.append((parts[0], parts[1], parts[2] if len(parts) > 2 else "upper"))
        else:
            for person in sorted(image_dir.glob("*.*")):
                cloth = cloth_dir / person.name
                entries.append((person.name, cloth.name if cloth.exists() else person.name, "upper"))

        if config.max_samples:
            entries = entries[: config.max_samples]
        if not entries:
            return 0

        for sub in ("image", "cloth", "cloth-mask", "agnostic", "pose"):
            (target_dir / sub).mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        for index, (person_name, cloth_name, category) in enumerate(entries):
            person_path = image_dir / person_name
            cloth_path = cloth_dir / cloth_name
            if not person_path.exists() or not cloth_path.exists():
                logger.debug("Skipping %s/%s (missing file)", person_name, cloth_name)
                continue

            person = load_image(person_path, mode="RGB")
            cloth = load_image(cloth_path, mode="RGB")
            mask_path = source_dir / "cloth-mask" / cloth_name
            agnostic_path = source_dir / "agnostic" / person_name
            pose_path = source_dir / "pose" / (Path(person_name).stem + ".png")

            # Fast path: the dataset already ships everything -> straight copy.
            if mask_path.exists() and agnostic_path.exists() and pose_path.exists():
                shutil.copy2(person_path, target_dir / "image" / person_name)
                shutil.copy2(cloth_path, target_dir / "cloth" / cloth_name)
                shutil.copy2(mask_path, target_dir / "cloth-mask" / cloth_name)
                shutil.copy2(agnostic_path, target_dir / "agnostic" / person_name)
                shutil.copy2(pose_path, target_dir / "pose" / (Path(person_name).stem + ".png"))
                sample_category = category
            else:
                cloth_mask = load_image(mask_path, mode="L") if mask_path.exists() else None
                sample = build_sample(
                    person_image=person, garment_image=cloth, cloth_mask=cloth_mask,
                    kind=_kind_for(category), agnostic_mode=config.agnostic_mode,
                )
                sample["category"] = sample_category = category
                write_sample(sample, target_dir, Path(person_name).stem)
            lines.append(f"{person_name} {cloth_name} {sample_category}\n")

            if progress is not None and index % 25 == 0:
                progress(index + 1, len(entries))

        (target_dir / "pairs.txt").write_text("".join(lines), encoding="utf-8")
        return len(lines)

    def split_dataset(self, name_or_path: str, val_fraction: float = 0.1, test_fraction: float = 0.1, seed: int = 42) -> Dict[str, Any]:
        """Re-split a dataset's train folder into train/val/test (file-level, keeps pairs)."""
        root = self.resolve(name_or_path)
        train_dir = root / "train"
        pairs_path = train_dir / "pairs.txt"
        if not pairs_path.exists():
            raise DatasetError(f"{root}/train/pairs.txt is required for splitting.")
        lines = [line.strip() for line in pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) < 3:
            raise DatasetError("Not enough samples to split (need at least 3).")
        rng = random.Random(seed)
        rng.shuffle(lines)
        n_test = max(1, int(len(lines) * test_fraction))
        n_val = max(1, int(len(lines) * val_fraction))
        test_lines = lines[:n_test]
        val_lines = lines[n_test:n_test + n_val]
        train_lines = lines[n_test + n_val:]

        for split, split_lines in (("train", train_lines), ("val", val_lines), ("test", test_lines)):
            split_dir = root / split
            for sub in ("image", "cloth", "cloth-mask", "agnostic", "pose"):
                (split_dir / sub).mkdir(parents=True, exist_ok=True)
            (split_dir / "pairs.txt").write_text("".join(line + "\n" for line in split_lines), encoding="utf-8")
            for line in split_lines:
                parts = line.split()
                person_name, cloth_name = parts[0], parts[1]
                self._link_or_copy(train_dir, split_dir, person_name, cloth_name)

        prepare_metadata(root, extra={"split_seed": seed})
        return {
            "ok": True,
            "counts": {"train": len(train_lines), "val": len(val_lines), "test": len(test_lines)},
            "path": str(root),
        }

    def _link_or_copy(self, source_split: Path, target_split: Path, person_name: str, cloth_name: str) -> None:
        mapping = {
            "image": person_name,
            "cloth": cloth_name,
            "cloth-mask": cloth_name,
            "agnostic": person_name,
            "pose": Path(person_name).stem + ".png",
        }
        for sub, filename in mapping.items():
            source = source_split / sub / filename
            if not source.exists():
                continue
            destination = target_split / sub / filename
            if destination.exists():
                continue
            try:  # hard link saves disk space and is instant on the same volume
                import os

                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)

    def _count_pairs(self, split_dir: Path) -> int:
        pairs = split_dir / "pairs.txt"
        if not pairs.exists():
            return 0
        return sum(1 for line in pairs.read_text(encoding="utf-8").splitlines() if line.strip())

    # ---------------------------------------------------------------------------------
    def statistics(self, name_or_path: str, resolution: int = 512, limit: int = 200) -> Dict[str, Any]:
        """Dataset statistics for the dashboard: counts, sizes, mask coverage, aspect ratios."""
        from backend.datasets.dataset import dataset_report

        root = self.resolve(name_or_path)
        report = dataset_report(root, resolution=resolution)
        samples: List[Dict[str, Any]] = []
        for split in ("train", "val", "test"):
            image_dir = root / split / "image"
            if not image_dir.exists():
                continue
            for path in list(image_dir.glob("*.*"))[:limit]:
                try:
                    image = load_image(path, mode="RGB")
                    samples.append({"split": split, "file": path.name, "width": image.shape[1], "height": image.shape[0]})
                except Exception:
                    continue
        if samples:
            widths = [item["width"] for item in samples]
            heights = [item["height"] for item in samples]
            report["image_stats"] = {
                "sampled": len(samples),
                "width_mean": round(sum(widths) / len(widths), 1),
                "height_mean": round(sum(heights) / len(heights), 1),
                "width_min": min(widths), "width_max": max(widths),
                "height_min": min(heights), "height_max": max(heights),
            }
        return report

    def delete(self, name: str) -> bool:
        path = self.resolve(name)
        if path.name in {"samples"}:
            logger.warning("Refusing to delete the built-in sample dataset via the API; use the CLI.")
        shutil.rmtree(path, ignore_errors=True)
        return not path.exists()


def _kind_for(category: str) -> str:
    if category in {"dress", "kurta", "traditional"}:
        return "dress"
    if category in {"bottom", "lower"}:
        return "lower"
    return "upper"


def describe_dataset_commands() -> Dict[str, str]:
    """Commands shown in the UI/docs so users always have the exact CLI to run."""
    return {
        "samples": "python scripts/prepare_dataset.py --use-samples",
        "convert": "python scripts/prepare_dataset.py --source /path/to/viton_hd --name viton_hd",
        "validate": "python scripts/validate_dataset.py --dataset datasets/viton_hd",
        "prepare_garments": "python scripts/preprocess_garments.py --input ./my_product_photos --output datasets/my_garments",
    }
