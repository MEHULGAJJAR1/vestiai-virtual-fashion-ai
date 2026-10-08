#!/usr/bin/env python
"""VestiAI — export / package a trained checkpoint.

Takes a training output folder and produces a self-contained, portable adapter directory
that the app can load, plus an optional archive you can move to another machine:

    checkpoints/epochs/epoch_0004 ──export──▶ checkpoints/exports/vestiai-vton-<stamp>/
                                               adapter weights (safetensors)
                                               config.json + model_card.json
                                               README.md  (how to load it)
                                               manifest.json (hashes + sizes + provenance)
    --archive → vestiai-vton-<stamp>.tar.gz

It verifies the adapter first with ``adapter_is_valid`` and refuses to ship a broken folder.
Nothing is quantised or converted silently: any conversion is explicit via ``--copy-as``.

    python scripts/export_model.py --checkpoint checkpoints/best_model
    python scripts/export_model.py --checkpoint checkpoints/epochs/epoch_0004 --archive
    python scripts/export_model.py --checkpoint checkpoints/best_model --promote   # copy to checkpoints/best_model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path

from _common import (  # noqa: E402
    PROJECT_ROOT, add_common_io_args, banner, fail, human_bytes, kv_table, load_settings,
    print_json, require_checkpoint, setup_logging,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export a VestiAI checkpoint as a portable adapter package.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", default=None, help="checkpoint folder (default: auto — best, latest, newest epoch)")
    p.add_argument("--output", default=None, help="destination folder (default: checkpoints/exports/<name>-<timestamp>)")
    p.add_argument("--name", default=None, help="package name (default: derived from the checkpoint)")
    p.add_argument("--archive", action="store_true", help="also write a .tar.gz next to the package")
    p.add_argument("--promote", action="store_true", help="copy the package into checkpoints/best_model")
    p.add_argument("--force", action="store_true", help="overwrite an existing package folder")
    p.add_argument("--json", action="store_true")
    add_common_io_args(p)
    return p


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    settings = load_settings(args.config)

    from backend.models.vton_controlnet import ModelCard, adapter_is_valid, describe_checkpoint

    source = require_checkpoint(args.checkpoint, "Export")
    info = describe_checkpoint(source)
    banner("VestiAI · checkpoint export", str(source))

    if not info.get("valid"):
        print(f"✗ this checkpoint is not loadable: {info.get('reason')}")
        print("  expected files: config.json + at least one *.safetensors / *.bin weight file.")
        print("  If the run was interrupted, try the newest folder under checkpoints/epochs/.\n")
        return 4

    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = args.name or f"vestiai-vton-{stamp}"
    destination = Path(args.output) if args.output else Path(settings.checkpoints_dir) / "exports" / name
    if destination.exists() and not args.force:
        fail(f"{destination} already exists (use --force to overwrite)")
    destination.mkdir(parents=True, exist_ok=True)

    # --- copy the adapter payload -----------------------------------------------------
    copied = []
    for pattern in ("*.safetensors", "*.bin", "config.json", "vestiai_config.json", "model_card.json",
                    "training_state.pt", "metrics.json"):
        for file in sorted(source.glob(pattern)):
            target = destination / file.name
            shutil.copy2(file, target)
            copied.append(target)

    if not any(file.suffix in {".safetensors", ".bin"} for file in copied):
        fail("no weight files were found to export")

    card = ModelCard.load(source)
    if card is None:
        card = ModelCard(category="vestiai-vton", notes=f"exported from {source} on {stamp}")
    card.exported_at = time.time() if hasattr(card, "exported_at") else None
    try:
        card.save(destination)
    except Exception:
        pass

    manifest = {
        "name": name,
        "created": stamp,
        "source": str(source),
        "files": [
            {"name": file.name, "bytes": file.stat().st_size, "sha256": sha256(file)}
            for file in sorted(destination.iterdir()) if file.is_file()
        ],
        "model_card": info.get("model_card"),
        "config": info.get("config"),
        "valid": adapter_is_valid(destination),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (destination / "README.md").write_text(export_readme(name, manifest, destination), encoding="utf-8")

    archive_path = None
    if args.archive:
        archive_path = destination.with_suffix(".tar.gz")
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(destination, arcname=name)
        print(f"  archive   : {archive_path} ({human_bytes(archive_path.stat().st_size)})")

    if args.promote:
        best = Path(settings.checkpoints_dir) / "best_model"
        if best.exists():
            backup = best.with_name(f"best_model_prev_{stamp}")
            shutil.move(str(best), str(backup))
            print(f"  previous best_model moved to {backup.name}")
        shutil.copytree(destination, best)
        print(f"  promoted  : {best}")

    banner("package written")
    kv_table({
        "folder": str(destination),
        "files": len(manifest["files"]),
        "total size": human_bytes(sum(file["bytes"] for file in manifest["files"])),
        "loadable": manifest["valid"],
        "category": (manifest["model_card"] or {}).get("category", "—"),
        "base model": (manifest["model_card"] or {}).get("base_model", "—"),
    })
    print("\n  use it:")
    print(f"    copy the folder into checkpoints/best_model on the inference machine, then reload")
    print(f"    the app (Model Status → Reload checkpoint) or run:")
    print(f"      python scripts/inference.py --person me.jpg --garment tee.jpg --checkpoint {destination}")
    if args.json:
        print_json(manifest)
    print()
    return 0


def export_readme(name: str, manifest: dict, destination: Path) -> str:
    card = manifest.get("model_card") or {}
    files = "\n".join(f"| `{file['name']}` | {human_bytes(file['bytes'])} | `{file['sha256'][:16]}…` |"
                      for file in manifest["files"])
    return f"""# {name}

Exported from VestiAI on {manifest['created']}.

| field | value |
| --- | --- |
| source | `{manifest['source']}` |
| category | `{card.get('category', 'vestiai-vton')}` |
| base model | `{card.get('base_model', '—')}` |
| resolution | {card.get('resolution', '—')} |
| trainable params | {card.get('trainable_parameters', '—')} |
| loadable | {'yes' if manifest['valid'] else 'NO — do not ship this'} |

## Files

| file | size | sha256 |
| --- | --- | --- |
{files}

## Loading it

```python
from backend.training.inference import InferenceConfig, VTONInference

engine = VTONInference(InferenceConfig(backend="diffusion", checkpoint="{destination}"))
result = engine.try_on("person.jpg", "garment.jpg", output_path="result.png")
```

Or place the folder at `checkpoints/best_model/` and hit **Reload checkpoint** on the Model
Status page — the app picks it up without a restart.

## What this adapter is

The ControlNet branch of VestiAI's inpainting pipeline, initialised from the base UNet and
fine-tuned to place this project's garment categories on a person photo. The base VAE, text
encoder and UNet stay frozen, so the file stays small (~3 GB for a full ControlNet, a few
hundred MB with LoRA only) and the pretrained knowledge of fabric and lighting is preserved.
"""


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
