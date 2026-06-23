"""
src/evaluate.py

Loads a trained model (weights-only or full model, whichever format was
saved) and runs a detailed evaluation on the Testing/ split: confusion
matrix, per-class precision/recall/F1, and a saved classification report.
Useful for comparing the 4 backbones beyond raw accuracy.

Usage:
    python src/evaluate.py --model_name resnet50
    python src/evaluate.py --model_name vgg16 --config configs/config.yaml
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models import get_preprocess_fn, load_trained_model


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained brain tumor MRI classifier")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--model_name",
        required=True,
        choices=["resnet50", "vgg16", "densenet121", "efficientnetb0"],
        help="Which trained model to evaluate",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ds_cfg = config["dataset"]
    paths_cfg = config["paths"]
    model_cfg = config["model"]

    img_h, img_w = ds_cfg["image_size"]

    # These kwargs must match what was used in train.py's build_model()
    # call, so the rebuilt architecture's layers line up with the saved
    # weights (only used if the model was saved as weights-only).
    model_kwargs = dict(
        input_shape=(img_h, img_w, 3),
        num_classes=model_cfg["num_classes"],
        dropout_rate=model_cfg["dropout_rate"],
        dense_units=model_cfg["dense_units"],
        freeze_backbone=model_cfg["freeze_backbone"],
        fine_tune_last_n_layers=model_cfg.get("fine_tune_last_n_layers", 0),
    )

    print(f"[evaluate] Loading model: {args.model_name}")
    model = load_trained_model(paths_cfg["model_output_dir"], args.model_name, model_kwargs)

    test_dir = Path(ds_cfg["data_dir"]) / ds_cfg["test_subdir"]
    preprocess_fn = get_preprocess_fn(args.model_name)

    test_aug = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_gen = test_aug.flow_from_directory(
        test_dir,
        target_size=(img_h, img_w),
        batch_size=ds_cfg["batch_size"],
        class_mode="categorical",
        classes=ds_cfg["classes"],
        shuffle=False,
    )

    print("[evaluate] Running predictions on test set ...")
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    class_names = ds_cfg["classes"]

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=class_names)
    print("\n" + report_text)

    artifacts_dir = Path(paths_cfg["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    report_path = artifacts_dir / f"{args.model_name}_classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    report_txt_path = artifacts_dir / f"{args.model_name}_classification_report.txt"
    with open(report_txt_path, "w") as f:
        f.write(report_text)

    # ---- Confusion matrix plot ----
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {args.model_name}")
    fig.tight_layout()

    cm_path = artifacts_dir / f"{args.model_name}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    print(f"\n[evaluate] Saved classification report -> {report_path}")
    print(f"[evaluate] Saved confusion matrix plot   -> {cm_path}")


if __name__ == "__main__":
    main()
