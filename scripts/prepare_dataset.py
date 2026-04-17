"""\
prepare_dataset.py
------------------
Builds the training/test folder structure expected by the TensorFlow pipeline
from the raw dataset layout in `datasets/skinIssues/`.

Input structure (as currently in this repo):
    datasets/skinIssues/
        acne/
        blackheades/      (typo in folder name)
        combination/
        dark spots/
        dry/
        normal/
        oily/
        pores/
        wrinkles/

Output structure (generated):
    datasets/
        train/
            Acne/
            Blackheads/
            Combination/
            Dark Spots/
            Dry/
            Normal/
            Oily/
            Pores/
            Wrinkles/
        test/
            ... same class folders ...
        class_names.json          (canonical class order used in training)
        dataset_metadata.json     (counts + mapping)

By default this script:
- Uses an 80/20 train/test split.
- Uses hardlinks (fast + no duplicated disk usage).
- Uses short sequential filenames to reduce Windows path-length issues.

Run from project root:
    python scripts/prepare_dataset.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


CANONICAL_CLASS_NAMES: list[str] = [
    "Acne",
    "Blackheads",
    "Combination",
    "Dark Spots",
    "Dry",
    "Normal",
    "Oily",
    "Pores",
    "Wrinkles",
]

FOLDER_TO_CLASS_NAME: dict[str, str] = {
    "acne": "Acne",
    "blackheades": "Blackheads",  # folder typo in dataset
    "blackheads": "Blackheads",
    "combination": "Combination",
    "dark spots": "Dark Spots",
    "dark_spots": "Dark Spots",
    "dry": "Dry",
    "normal": "Normal",
    "oily": "Oily",
    "pores": "Pores",
    "wrinkles": "Wrinkles",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Paths:
    project_root: Path
    source_root: Path
    train_root: Path
    test_root: Path
    class_names_json: Path
    dataset_metadata_json: Path


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _iter_images(class_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return files


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            # Fall back to copy if hardlinking fails (e.g., cross-device)
            shutil.copy2(src, dst)
            return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    raise ValueError(f"Unsupported mode: {mode}")


def _normalize_folder_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def build_paths(project_root: Path) -> Paths:
    datasets_dir = project_root / "datasets"
    return Paths(
        project_root=project_root,
        source_root=datasets_dir / "skinIssues",
        train_root=datasets_dir / "train",
        test_root=datasets_dir / "test",
        class_names_json=datasets_dir / "class_names.json",
        dataset_metadata_json=datasets_dir / "dataset_metadata.json",
    )


def prepare_dataset(
    paths: Paths,
    test_ratio: float,
    seed: int,
    mode: str,
    clean: bool,
) -> None:
    if not paths.source_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {paths.source_root}. "
            "Expected datasets/skinIssues/<class-folders>/"
        )

    if clean:
        _safe_rmtree(paths.train_root)
        _safe_rmtree(paths.test_root)

    paths.train_root.mkdir(parents=True, exist_ok=True)
    paths.test_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Discover and normalize folders
    source_folders = [p for p in paths.source_root.iterdir() if p.is_dir()]
    if not source_folders:
        raise RuntimeError(f"No class folders found under {paths.source_root}")

    unknown = []
    class_to_images: dict[str, list[Path]] = {name: [] for name in CANONICAL_CLASS_NAMES}

    for folder in source_folders:
        norm = _normalize_folder_name(folder.name)
        mapped = FOLDER_TO_CLASS_NAME.get(norm)
        if not mapped:
            unknown.append(folder.name)
            continue

        images = _iter_images(folder)
        class_to_images[mapped].extend(images)

    if unknown:
        raise RuntimeError(
            "Unknown class folders under datasets/skinIssues: "
            + ", ".join(sorted(unknown))
            + ". Add them to FOLDER_TO_CLASS_NAME in scripts/prepare_dataset.py."
        )

    # Ensure we have at least something to work with
    total_images = sum(len(v) for v in class_to_images.values())
    if total_images == 0:
        raise RuntimeError(
            f"No images found under {paths.source_root}. "
            f"Supported extensions: {sorted(IMAGE_EXTS)}"
        )

    # Write class_names.json so training/inference uses the same order.
    paths.class_names_json.write_text(
        json.dumps(CANONICAL_CLASS_NAMES, indent=2), encoding="utf-8"
    )

    metadata: dict[str, object] = {
        "source_root": str(paths.source_root).replace("\\\\", "/"),
        "test_ratio": test_ratio,
        "seed": seed,
        "mode": mode,
        "class_names": CANONICAL_CLASS_NAMES,
        "counts": {},
    }

    for class_name in CANONICAL_CLASS_NAMES:
        images = list(class_to_images[class_name])
        rng.shuffle(images)

        n = len(images)
        if n == 0:
            # Keep empty folders so the class still exists in the model output
            (paths.train_root / class_name).mkdir(parents=True, exist_ok=True)
            (paths.test_root / class_name).mkdir(parents=True, exist_ok=True)
            metadata["counts"][class_name] = {"train": 0, "test": 0, "total": 0}
            continue

        # Choose a test split that keeps at least 1 train sample when possible.
        if n == 1:
            test_count = 0
        else:
            test_count = int(round(n * test_ratio))
            test_count = max(1, min(test_count, n - 1))

        test_images = images[:test_count]
        train_images = images[test_count:]

        train_dir = paths.train_root / class_name
        test_dir = paths.test_root / class_name
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Use short sequential filenames to avoid long Windows paths.
        for i, src in enumerate(train_images, start=1):
            ext = src.suffix.lower() or ".jpg"
            dst = train_dir / f"{i:06d}{ext}"
            _link_or_copy(src, dst, mode)

        for i, src in enumerate(test_images, start=1):
            ext = src.suffix.lower() or ".jpg"
            dst = test_dir / f"{i:06d}{ext}"
            _link_or_copy(src, dst, mode)

        metadata["counts"][class_name] = {
            "train": len(train_images),
            "test": len(test_images),
            "total": n,
        }

    paths.dataset_metadata_json.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("✅ Dataset prepared")
    print(f"- Train: {paths.train_root}")
    print(f"- Test:  {paths.test_root}")
    print(f"- Class names: {paths.class_names_json}")
    print(f"- Metadata:    {paths.dataset_metadata_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Skin Insight dataset")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of images to allocate to test set (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic split (default: 42)",
    )
    parser.add_argument(
        "--mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="How to materialize train/test files (default: hardlink)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete existing datasets/train and datasets/test first",
    )

    args = parser.parse_args()

    if not (0.0 <= args.test_ratio <= 0.9):
        raise SystemExit("--test-ratio must be between 0.0 and 0.9")

    project_root = Path(__file__).resolve().parents[1]
    paths = build_paths(project_root)

    prepare_dataset(
        paths=paths,
        test_ratio=args.test_ratio,
        seed=args.seed,
        mode=args.mode,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()
