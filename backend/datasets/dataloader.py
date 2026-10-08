"""DataLoader construction tuned per platform.

Low worker counts matter: diffusion data pipelines decode several images per sample, so on
Windows/macOS (spawn start method) each worker re-imports torch and can cost ~1 GB of RAM.
The defaults here pick something safe, and ``pin_memory``/``persistent_workers`` are only
enabled where they actually help (CUDA).
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Any, Dict, Optional

from backend.datasets.dataset import VTONPairDataset, build_dataset, collate_fn
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LoaderConfig:
    """DataLoader settings resolved from the training config."""

    batch_size: int = 2
    num_workers: int = 2
    shuffle: bool = True
    drop_last: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    seed: int = 42

    @classmethod
    def resolve(
        cls,
        batch_size: int = 2,
        num_workers: Optional[int] = None,
        device: str = "cpu",
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ) -> "LoaderConfig":
        """Fill in platform-appropriate defaults."""
        if num_workers is None:
            cpu_count = os.cpu_count() or 2
            if platform.system() in {"Darwin", "Windows"}:
                num_workers = min(2, max(0, cpu_count // 2))
            else:
                num_workers = min(8, max(1, cpu_count - 1))
        cuda = device == "cuda"
        return cls(
            batch_size=batch_size,
            num_workers=int(num_workers),
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=cuda,
            persistent_workers=int(num_workers) > 0,
            prefetch_factor=2 if int(num_workers) > 0 else 2,
            seed=seed,
        )


def build_dataloaders(
    root: str,
    resolution: int = 512,
    mode: str = "QUICK_DEMO",
    batch_size: int = 2,
    num_workers: Optional[int] = None,
    device: str = "cpu",
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    prompt_template: Optional[str] = None,
    augment: bool = True,
    **dataset_kwargs: Any,
) -> Dict[str, Any]:
    """Create train/val/test loaders (val + test are shuffle-free)."""
    import torch
    from torch.utils.data import DataLoader

    datasets: Dict[str, VTONPairDataset] = {}
    for split in ("train", "val", "test"):
        try:
            dataset = build_dataset(
                root=root, split=split, resolution=resolution, mode=mode, augment=augment and split == "train",
                seed=seed, max_samples=max_train_samples if split == "train" else None,
                prompt_template=prompt_template, **dataset_kwargs,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not build '%s' split: %s", split, exc)
            continue
        if len(dataset) == 0:
            logger.warning("Split '%s' has no samples; skipping its loader.", split)
            continue
        datasets[split] = dataset

    if "train" not in datasets:
        raise FileNotFoundError(
            f"No usable training samples under '{root}'. Run `python scripts/prepare_dataset.py --use-samples` "
            "or point --dataset at a VITON-HD/DressCode conversion."
        )

    loaders: Dict[str, Any] = {}
    config_train = LoaderConfig.resolve(batch_size, num_workers, device, shuffle=True, drop_last=True, seed=seed)
    loaders["train"] = DataLoader(
        datasets["train"], batch_size=config_train.batch_size, shuffle=True,
        num_workers=config_train.num_workers, pin_memory=config_train.pin_memory,
        drop_last=config_train.drop_last, collate_fn=collate_fn,
        persistent_workers=config_train.persistent_workers, prefetch_factor=config_train.prefetch_factor,
        generator=torch.Generator().manual_seed(seed),
    )

    for split in ("val", "test"):
        if split not in datasets:
            continue
        config = LoaderConfig.resolve(max(1, batch_size), num_workers, device, shuffle=False, drop_last=False, seed=seed)
        loaders[split] = DataLoader(
            datasets[split], batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=False,
            collate_fn=collate_fn, persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
        )

    logger.info(
        "Loaders ready: %s", {name: len(loader.dataset) for name, loader in loaders.items()}  # type: ignore[attr-defined]
    )
    return {"loaders": loaders, "datasets": datasets}


def describe_loaders(loaders: Dict[str, Any]) -> Dict[str, Any]:
    """Serialisable summary for the training dashboard."""
    out: Dict[str, Any] = {}
    for name, loader in loaders.items():
        dataset = getattr(loader, "dataset", None)
        out[name] = {
            "samples": len(dataset) if dataset is not None else 0,
            "batches": len(loader),
            "batch_size": getattr(loader, "batch_size", None),
            "num_workers": getattr(loader, "num_workers", None),
            "summary": dataset.summary() if hasattr(dataset, "summary") else {},
        }
    return out
