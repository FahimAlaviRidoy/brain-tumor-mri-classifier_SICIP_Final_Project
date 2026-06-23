"""
src/predict.py

Runs inference on a single MRI image using a trained model and prints
the predicted tumor class with confidence scores for all classes.

Usage:
    python src/predict.py --input path/to/sample.jpg --model_name resnet50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tensorflow.keras.preprocessing import image as keras_image

from models import get_preprocess_fn, load_trained_model


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_class_mapping(artifacts_dir: Path) -> dict:
    """Loads {0: 'glioma', 1: 'meningioma', ...} saved during training."""
    mapping_path = artifacts_dir / "class_indices.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Class mapping not found at '{mapping_path}'. "
            f"Run training first: python src/train.py"
        )
    with open(mapping_path, "r") as f:
        raw = json.load(f)
    # JSON keys are strings; convert back to int
    return {int(k): v for k, v in raw.items()}


def preprocess_image(img_path: str, target_size, preprocess_fn):
    img = keras_image.load_img(img_path, target_size=target_size)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_fn(arr)
    return arr


def predict_single_image(model, img_array, idx_to_class: dict):
    probs = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))
    predicted_class = idx_to_class[predicted_idx]
    confidence = float(probs[predicted_idx])
    all_scores = {idx_to_class[i]: float(p) for i, p in enumerate(probs)}
    return predicted_class, confidence, all_scores


def main():
    parser = argparse.ArgumentParser(description="Predict brain tumor class from an MRI image")
    parser.add_argument("--input", required=True, help="Path to an MRI image (jpg/png)")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--model_name",
        default=None,
        choices=["resnet50", "vgg16", "densenet121", "efficientnetb0"],
        help="Overrides model.active_model from the config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = args.model_name or config["model"]["active_model"]
    img_h, img_w = config["dataset"]["image_size"]
    paths_cfg = config["paths"]
    model_cfg = config["model"]

    # Must match build_model() kwargs used during training, so a
    # weights-only file can be loaded back into an identical architecture.
    model_kwargs = dict(
        input_shape=(img_h, img_w, 3),
        num_classes=model_cfg["num_classes"],
        dropout_rate=model_cfg["dropout_rate"],
        dense_units=model_cfg["dense_units"],
        freeze_backbone=model_cfg["freeze_backbone"],
        fine_tune_last_n_layers=model_cfg.get("fine_tune_last_n_layers", 0),
    )

    idx_to_class = load_class_mapping(Path(paths_cfg["artifacts_dir"]))
    preprocess_fn = get_preprocess_fn(model_name)

    print(f"[predict] Loading model: {model_name}")
    model = load_trained_model(paths_cfg["model_output_dir"], model_name, model_kwargs)

    img_array = preprocess_image(args.input, (img_h, img_w), preprocess_fn)
    predicted_class, confidence, all_scores = predict_single_image(model, img_array, idx_to_class)

    print(f"\nInput image:      {args.input}")
    print(f"Model used:       {model_name}")
    print(f"Predicted class:  {predicted_class}")
    print(f"Confidence:       {confidence:.2%}")
    print("\nAll class scores:")
    for cls, score in sorted(all_scores.items(), key=lambda x: -x[1]):
        print(f"  {cls:<12} {score:.2%}")


if __name__ == "__main__":
    main()
