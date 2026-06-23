"""
src/data_download.py

Automatically downloads the Brain Tumor MRI Dataset from Kaggle into the
local data/ folder. This script must be run before training — the dataset
itself is NEVER committed to the repository (see .gitignore).

Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Classes: glioma, meningioma, notumor, pituitary
Source split: Training/ (~5712 images) and Testing/ (~1311 images)

Authentication:
    This script uses `kagglehub`, which will prompt for Kaggle credentials
    on first run (or read them from ~/.kaggle/kaggle.json if already
    configured). To set up credentials:
        1. Go to https://www.kaggle.com/settings -> "Create New Token"
        2. This downloads kaggle.json
        3. Place it at ~/.kaggle/kaggle.json (Linux/Mac) or
           C:\\Users\\<you>\\.kaggle\\kaggle.json (Windows)

Usage:
    python src/data_download.py
    python src/data_download.py --config configs/config.yaml
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_split_dirs(root: Path, train_subdir: str, test_subdir: str):
    """
    kagglehub may extract the dataset with an extra nesting level.
    Walk the downloaded tree to robustly locate Training/ and Testing/.
    """
    candidates_train = list(root.rglob(train_subdir))
    candidates_test = list(root.rglob(test_subdir))

    if not candidates_train or not candidates_test:
        raise FileNotFoundError(
            f"Could not locate '{train_subdir}' / '{test_subdir}' folders "
            f"under {root}. Downloaded contents: {list(root.iterdir())}"
        )

    # Prefer the shallowest match
    train_dir = sorted(candidates_train, key=lambda p: len(p.parts))[0]
    test_dir = sorted(candidates_test, key=lambda p: len(p.parts))[0]
    return train_dir, test_dir


def main():
    parser = argparse.ArgumentParser(description="Download Brain Tumor MRI dataset from Kaggle")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--force", action="store_true", help="Re-download even if data/ already populated")
    args = parser.parse_args()

    config = load_config(args.config)
    ds_cfg = config["dataset"]

    data_dir = Path(ds_cfg["data_dir"])
    train_target = data_dir / ds_cfg["train_subdir"]
    test_target = data_dir / ds_cfg["test_subdir"]

    if train_target.exists() and test_target.exists() and not args.force:
        print(f"[data_download] Dataset already present at '{data_dir}/'. "
              f"Use --force to re-download.")
        return

    print(f"[data_download] Downloading '{ds_cfg['kaggle_handle']}' via kagglehub ...")
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub is not installed. Run: pip install kagglehub", file=sys.stderr)
        sys.exit(1)

    try:
        download_path = kagglehub.dataset_download(ds_cfg["kaggle_handle"])
    except Exception as e:
        print(
            "ERROR: kagglehub download failed. This usually means Kaggle API "
            "credentials are missing.\n"
            "Fix: download kaggle.json from https://www.kaggle.com/settings "
            "and place it at ~/.kaggle/kaggle.json (or %USERPROFILE%\\.kaggle\\kaggle.json "
            "on Windows), then re-run this script.\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    download_root = Path(download_path)
    print(f"[data_download] Downloaded to cache: {download_root}")

    train_src, test_src = find_split_dirs(
        download_root, ds_cfg["train_subdir"], ds_cfg["test_subdir"]
    )

    data_dir.mkdir(parents=True, exist_ok=True)

    if train_target.exists():
        shutil.rmtree(train_target)
    if test_target.exists():
        shutil.rmtree(test_target)

    print(f"[data_download] Copying Training/ -> {train_target}")
    shutil.copytree(train_src, train_target)
    print(f"[data_download] Copying Testing/  -> {test_target}")
    shutil.copytree(test_src, test_target)

    # Sanity check: report image counts per class
    print("\n[data_download] Dataset ready. Class distribution:")
    for split_name, split_path in [("Training", train_target), ("Testing", test_target)]:
        print(f"  {split_name}:")
        for cls in ds_cfg["classes"]:
            cls_dir = split_path / cls
            count = len(list(cls_dir.glob("*"))) if cls_dir.exists() else 0
            print(f"    {cls:<12} {count} images")

    print(f"\n[data_download] Done. Data located at '{data_dir.resolve()}'")


if __name__ == "__main__":
    main()
