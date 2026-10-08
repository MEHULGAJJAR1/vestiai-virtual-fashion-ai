#!/usr/bin/env python
"""VestiAI — train the garment-category classifier head.

VestiAI classifies a garment in three tiers: CLIP zero-shot (if `transformers` is installed),
a small scikit-learn head trained here, and a pure-geometry heuristic that always works. This
script trains the middle tier: a 13-D silhouette feature vector (aspect ratio, fill, chest /
waist / hem widths, flare, sleeve mass, shoulder slope, symmetry, notch, solidity, saturation)
→ `LogisticRegression`.

The head it writes is a plain fitted estimator dumped with joblib, because that is exactly what
`GarmentClassifier._load_trained` expects: `joblib.load(path).predict_proba(vector)` with
`classes_` on the object. Drop the file at ``models_cache/garment_classifier.joblib`` (or point
``classifier_head:`` in configs/default.yaml at it) and the service picks it up on the next
ingest — classification then reports ``method: trained_head`` and, on the Model Status page,
``trained_head: true``.

Label sources, all real:

    1. every garment already in My Closet (its mask + its category, minus user_overrides)
    2. ``--bootstrap N`` garments per category, drawn by the procedural sample generator —
       a legitimate way to get a working head before you own any photos
    3. ``--extra NAME=FOLDER`` folders, labelled by folder name (your own labelled data)

    python scripts/train_classifier.py --bootstrap 40 --output models_cache/garment_classifier.joblib
    python scripts/train_classifier.py --from-closet
    python scripts/train_classifier.py --from-closet --extra dresses=./labelled/dresses
    python scripts/train_classifier.py --evaluate-only --output models_cache/garment_classifier.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from _common import (  # noqa: E402
    PROJECT_ROOT, add_common_io_args, banner, fail, kv_table, load_settings, print_json, setup_logging,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the 13-D silhouette → category classifier head (scikit-learn).",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output", default=None,
                   help="where to write the joblib head (default: models_cache/garment_classifier.joblib)")
    p.add_argument("--from-closet", action="store_true", help="use My Closet as the label source")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="also generate N synthetic garments per category as training data")
    p.add_argument("--extra", action="append", default=[], metavar="NAME=FOLDER",
                   help="add a labelled folder (repeatable); NAME is the category")
    p.add_argument("--categories", default=None, help="comma-separated category whitelist")
    p.add_argument("--test-size", type=float, default=0.25, help="held-out fraction")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--evaluate-only", action="store_true", help="load the existing head and re-report its accuracy")
    p.add_argument("--json", action="store_true")
    add_common_io_args(p)
    return p


def closet_samples(settings, categories: list[str]) -> tuple[list, list, list]:
    """Pull (features, labels, origins) from the closet records."""
    from backend.services.garment_classifier import feature_vector
    from backend.services.garment_service import ClosetStore
    from backend.utils.image_utils import load_image

    closet = ClosetStore(settings.garments_dir, settings.closet_db)
    features, labels, origins = [], [], []
    for record in closet.all():
        category = (record.category or "").lower()
        if not category or category in {"unknown", "custom"}:
            continue          # unlabelled: it would teach the head nothing
        if categories and category not in categories:
            continue
        # Records labelled by a human (method == "user_override") are the most reliable labels
        # in the closet, so they are used as-is; auto-detected ones count too but the report
        # prints the split so a noisy source is visible in the numbers.
        mask_path, image_path = Path(record.mask_path), Path(record.image_path)
        if not mask_path.exists() or not image_path.exists():
            continue
        alpha = np.asarray(load_image(mask_path, mode="L"))
        rgb = np.asarray(load_image(image_path))
        features.append(feature_vector(alpha, rgb))
        labels.append(category)
        origins.append(f"closet:{record.key}")
    return features, labels, origins


def bootstrap_samples(categories: list[str], per_category: int, seed: int) -> tuple[list, list, list]:
    """Generate real synthetic garments (procedural generator) and extract their features."""
    from backend.services.garment_classifier import feature_vector
    from backend.services.sample_data import generate_garment

    features, labels, origins = [], [], []
    for category in categories:
        for index in range(per_category):
            garment = generate_garment(category=category, size=640, seed=seed + index * 17 + hash(category) % 997)
            rgba = np.asarray(garment.image)
            alpha = rgba[..., 3]
            rgb = rgba[..., :3]
            features.append(feature_vector(alpha, rgb))
            labels.append(garment.category)
            origins.append(f"sample:{garment.label}")
    return features, labels, origins


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    settings = load_settings(args.config)

    from backend.services.garment_classifier import CATEGORIES

    categories = [c.strip().lower() for c in (args.categories.split(",") if args.categories else CATEGORIES) if c.strip()]
    output = Path(args.output) if args.output else settings.models_cache_dir / "garment_classifier.joblib"
    banner("VestiAI · garment classifier head", f"{len(categories)} categories → {output}")

    if args.evaluate_only:
        return evaluate_only(output, categories, args)

    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features: list = []
    labels: list[str] = []
    origins: list[str] = []

    if args.from_closet or not (args.bootstrap or args.extra):
        closet_features, closet_labels, closet_origins = closet_samples(settings, categories)
        features += closet_features
        labels += closet_labels
        origins += closet_origins
        print(f"  closet      : {len(closet_features)} labelled garment(s)")

    if args.extra:
        extra_features, extra_labels, extra_origins = labelled_folders(args.extra, categories)
        features += extra_features
        labels += extra_labels
        origins += extra_origins
        print(f"  labelled dirs: {len(extra_features)} image(s) from {len(args.extra)} folder(s)")

    if args.bootstrap:
        sample_features, sample_labels, sample_origins = bootstrap_samples(categories, args.bootstrap, args.seed)
        features += sample_features
        labels += sample_labels
        origins += sample_origins
        print(f"  bootstrap   : {len(sample_features)} synthetic garment(s) "
              f"({args.bootstrap} per category × {len(categories)})")

    if len(features) < 24:
        fail(
            f"only {len(features)} labelled sample(s) — that is not enough to train a head.\n"
            "  Get more with any of:  --bootstrap 40   ·   --from-closet (after uploading garments)\n"
            "                        ·  --extra kurtas=./labelled/kurtas"
        )

    X = np.vstack([np.asarray(f, dtype=np.float32) for f in features])
    y = np.asarray(labels, dtype=object)
    counts = Counter(labels)
    kv_table({"samples": len(y), "features": X.shape[1], "classes": dict(sorted(counts.items()))})

    # Every class needs at least 2 members to be testable.
    keep = [label for label, count in counts.items() if count >= 2]
    dropped = {label: count for label, count in counts.items() if count < 2}
    if dropped:
        print(f"  dropping classes with <2 samples: {dropped}")
    mask = np.isin(y, keep)
    X, y = X[mask], y[mask]
    if len(set(y)) < 2:
        fail("at least two categories with ≥2 samples each are required")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    report = classification_report(y_test, prediction, zero_division=0)
    matrix = confusion_matrix(y_test, prediction, labels=sorted(set(y)))

    folds = min(5, min(Counter(y).values()))
    cv = cross_val_score(model, X, y, cv=max(2, folds)) if folds >= 2 else np.asarray([accuracy])

    banner("held-out results")
    kv_table({
        "train": len(y_train),
        "test": len(y_test),
        "accuracy": round(float(accuracy), 3),
        "cv mean": round(float(cv.mean()), 3),
        "cv std": round(float(cv.std()), 3),
        "cv folds": int(len(cv)),
    })
    print("\n  per class:\n")
    for line in report.strip().splitlines():
        print(f"    {line}")
    print("\n  confusion matrix (rows = truth, cols = prediction):")
    print(f"    labels: {sorted(set(y))}")
    for row in matrix:
        print(f"    {list(int(v) for v in row)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "path": str(output),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": float(accuracy),
        "cv_mean": float(cv.mean()),
        "samples": int(len(y)),
        "classes": sorted(counts),
        "feature_names": [
            "aspect", "fill_ratio", "top_width_ratio", "chest_width_ratio", "waist_width_ratio",
            "hem_width_ratio", "hem_flare", "sleeve_mass", "shoulder_slope", "symmetry",
            "top_notch", "solidity", "color_saturation",
        ],
        "sources": {"closet": sum(o.startswith("closet:") for o in origins),
                    "bootstrap": sum(o.startswith("sample:") for o in origins),
                    "folders": sum(o.startswith("folder:") for o in origins)},
        "note": "Plain sklearn estimator (StandardScaler + LogisticRegression); "
                "GarmentClassifier loads it with joblib and calls predict_proba().",
    }, indent=2), encoding="utf-8")

    banner("head written")
    kv_table({"joblib": str(output), "sidecar": str(sidecar), "size": f"{output.stat().st_size / 1024:.1f} KB"})
    print("\n  activate it (either):")
    print(f"    · point `classifier_head:` in configs/default.yaml at {output}")
    print(f"    · or set it per run:  VESTIAI_CONFIG=... / pass classifier_head to load_settings(overrides=…))")
    print("  then check the Model Status page: the classifier panel must show `trained_head: true`,")
    print("  and new uploads should report `method: trained_head`.\n")
    if args.json:
        print_json({"output": str(output), "accuracy": float(accuracy), "cv_mean": float(cv.mean()),
                    "samples": int(len(y)), "classes": sorted(counts)})
    return 0


def labelled_folders(pairs: list[str], categories: list[str]) -> tuple[list, list, list]:
    """Read ``--extra name=folder`` labelled folders through the real garment pipeline."""
    from backend.services.garment_classifier import cloth_mask_from_product_image, feature_vector
    from backend.utils.image_utils import load_image

    features, labels, origins = [], [], []
    for item in pairs:
        if "=" not in item:
            fail(f"--extra expects NAME=FOLDER, got '{item}'")
        name, folder = item.split("=", 1)
        name = name.strip().lower()
        if name not in categories and categories:
            print(f"  · skipping '{name}' (not in --categories)")
            continue
        folder_path = Path(folder).expanduser()
        if not folder_path.exists():
            fail(f"labelled folder not found: {folder_path}")
        count = 0
        for file in sorted(folder_path.rglob("*")):
            if file.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rgba = load_image(file, mode="RGBA")
            alpha = rgba[..., 3]
            rgb = rgba[..., :3]
            if int((alpha > 20).sum()) < 0.02 * alpha.size:
                # Opaque photo: segment it first so the features mean the same thing.
                alpha = cloth_mask_from_product_image(rgb)
            features.append(feature_vector(alpha, rgb))
            labels.append(name)
            origins.append(f"folder:{file.name}")
            count += 1
        print(f"  · {name:<12} {count} image(s) from {folder_path}")
    return features, labels, origins


def evaluate_only(output: Path, categories: list[str], args) -> int:
    """Re-report a stored head on freshly gathered data (no retraining)."""
    if not output.exists():
        fail(f"no head at {output} — train one first")
    import joblib
    from sklearn.metrics import accuracy_score

    settings = load_settings(args.config)
    features, labels, _ = closet_samples(settings, categories)
    if not features:
        fail("the closet has no labelled garments to evaluate against")
    model = joblib.load(output)
    X = np.vstack([np.asarray(f, dtype=np.float32) for f in features])
    y = np.asarray(labels, dtype=object)
    try:
        prediction = model.predict(X)
    except Exception as exc:
        fail(f"the stored head could not be used: {exc}")
    banner("evaluate-only")
    kv_table({
        "head": str(output),
        "samples": len(y),
        "closet accuracy": round(float(accuracy_score(y, prediction)), 3),
        "classes on the head": list(getattr(model, "classes_", [])) or "see pipeline step",
    })
    print("\n  note: this is the head scored on closet labels — some of which it may have trained on.\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
