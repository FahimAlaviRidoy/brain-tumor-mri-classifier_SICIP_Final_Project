"""
src/train.py

Trains a transfer-learning brain tumor classifier and logs the full run
to MLflow (params, metrics per epoch, training curve plots, the trained
model, and the class-index mapping).

Run this script once per backbone (resnet50, vgg16, densenet121,
efficientnetb0) to produce the 4 MLflow runs required for comparison.
Override `model.active_model` via --model_name without editing the YAML.

Usage:
    python src/train.py --config configs/config.yaml --model_name resnet50
    python src/train.py --config configs/config.yaml --model_name vgg16
    python src/train.py --config configs/config.yaml --model_name densenet121
    python src/train.py --config configs/config.yaml --model_name efficientnetb0
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless plotting (no display needed)
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import yaml
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models import build_model, get_preprocess_fn


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return Adam(learning_rate=lr)
    if name == "sgd":
        return SGD(learning_rate=lr, momentum=0.9)
    if name == "rmsprop":
        return RMSprop(learning_rate=lr)
    raise ValueError(f"Unknown optimizer '{name}'")


def build_data_generators(config: dict, model_name: str):
    ds_cfg = config["dataset"]
    img_h, img_w = ds_cfg["image_size"]
    batch_size = ds_cfg["batch_size"]

    data_dir = Path(ds_cfg["data_dir"])
    train_dir = data_dir / ds_cfg["train_subdir"]
    test_dir = data_dir / ds_cfg["test_subdir"]

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training data not found at '{train_dir}'. "
            f"Run `python src/data_download.py` first."
        )

    preprocess_fn = get_preprocess_fn(model_name)

    # Train/val split comes out of Training/, with light augmentation.
    # Augmentation choices are conservative — MRI scans are fairly
    # standardized in orientation/contrast, so we avoid aggressive
    # color/rotation jitter that could create unrealistic anatomy.
    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=ds_cfg["val_split"],
    )

    test_aug = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_aug.flow_from_directory(
        train_dir,
        target_size=(img_h, img_w),
        batch_size=batch_size,
        class_mode="categorical",
        classes=ds_cfg["classes"],
        subset="training",
        seed=ds_cfg["seed"],
        shuffle=True,
    )

    val_gen = train_aug.flow_from_directory(
        train_dir,
        target_size=(img_h, img_w),
        batch_size=batch_size,
        class_mode="categorical",
        classes=ds_cfg["classes"],
        subset="validation",
        seed=ds_cfg["seed"],
        shuffle=False,
    )

    test_gen = test_aug.flow_from_directory(
        test_dir,
        target_size=(img_h, img_w),
        batch_size=batch_size,
        class_mode="categorical",
        classes=ds_cfg["classes"],
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def plot_training_curves(history, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train a brain tumor MRI classifier")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--model_name",
        default=None,
        choices=["resnet50", "vgg16", "densenet121", "efficientnetb0"],
        help="Overrides model.active_model from the config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    train_cfg = config["training"]
    mlflow_cfg = config["mlflow"]
    paths_cfg = config["paths"]

    model_name = args.model_name or model_cfg["active_model"]
    img_h, img_w = config["dataset"]["image_size"]
    input_shape = (img_h, img_w, 3)

    print(f"[train] Building data generators for '{model_name}' preprocessing ...")
    train_gen, val_gen, test_gen = build_data_generators(config, model_name)
    class_indices = train_gen.class_indices  # e.g. {'glioma':0, 'meningioma':1, ...}
    print(f"[train] Class index mapping: {class_indices}")

    print(f"[train] Building model: {model_name}")
    model = build_model(
        model_name=model_name,
        input_shape=input_shape,
        num_classes=model_cfg["num_classes"],
        dropout_rate=model_cfg["dropout_rate"],
        dense_units=model_cfg["dense_units"],
        freeze_backbone=model_cfg["freeze_backbone"],
        fine_tune_last_n_layers=model_cfg.get("fine_tune_last_n_layers", 0),
    )

    optimizer = get_optimizer(train_cfg["optimizer"], train_cfg["learning_rate"])
    model.compile(optimizer=optimizer, loss=train_cfg["loss"], metrics=["accuracy"])
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=train_cfg["reduce_lr_patience"],
            min_lr=1e-7,
        ),
    ]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    model_dir = Path(paths_cfg["model_output_dir"])
    artifacts_dir = Path(paths_cfg["artifacts_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{model_name}_transfer_learning"
    with mlflow.start_run(run_name=run_name) as run:
        # ---- Log hyperparameters ----
        mlflow.log_params({
            "model_name": model_name,
            "image_height": img_h,
            "image_width": img_w,
            "batch_size": config["dataset"]["batch_size"],
            "val_split": config["dataset"]["val_split"],
            "epochs": train_cfg["epochs"],
            "learning_rate": train_cfg["learning_rate"],
            "optimizer": train_cfg["optimizer"],
            "dropout_rate": model_cfg["dropout_rate"],
            "dense_units": model_cfg["dense_units"],
            "freeze_backbone": model_cfg["freeze_backbone"],
            "fine_tune_last_n_layers": model_cfg.get("fine_tune_last_n_layers", 0),
            "early_stopping_patience": train_cfg["early_stopping_patience"],
        })

        print(f"[train] Starting training for {train_cfg['epochs']} epochs ...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=train_cfg["epochs"],
            callbacks=callbacks,
            verbose=2,
        )

        # ---- Log per-epoch metrics ----
        for epoch in range(len(history.history["loss"])):
            mlflow.log_metrics({
                "train_loss": history.history["loss"][epoch],
                "train_accuracy": history.history["accuracy"][epoch],
                "val_loss": history.history["val_loss"][epoch],
                "val_accuracy": history.history["val_accuracy"][epoch],
            }, step=epoch)

        # ---- Evaluate on held-out Testing/ set ----
        print("[train] Evaluating on test set ...")
        test_loss, test_acc = model.evaluate(test_gen, verbose=2)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})
        print(f"[train] Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

        # ---- Save training curve plot and log as artifact ----
        plot_path = artifacts_dir / f"{model_name}_training_curves.png"
        plot_training_curves(history, plot_path)
        mlflow.log_artifact(str(plot_path))

        # ---- Save class index mapping (needed by predict.py / app) ----
        class_map_path = artifacts_dir / "class_indices.json"
        idx_to_class = {v: k for k, v in class_indices.items()}
        with open(class_map_path, "w") as f:
            json.dump(idx_to_class, f, indent=2)
        mlflow.log_artifact(str(class_map_path))

        # ---- Save and log the trained model weights ----
        # NOTE: TensorFlow/Keras 2.10 (pinned for Windows DirectML
        # compatibility, even though DirectML itself isn't used - see
        # README) does not have the newer Keras v3 ".keras" format,
        # which only became default starting Keras 2.13. On 2.10, ANY
        # extension passed to model.save() routes through the same
        # legacy HDF5 saver, which serializes the full architecture
        # config to JSON. EfficientNetB0's config embeds raw TensorFlow
        # tensors (normalization stats) that crash that JSON step.
        #
        # Fix: save weights only (model.save_weights), which skips
        # config serialization entirely and only writes the numeric
        # weight arrays. This works identically across all 4 backbones.
        # Reconstructing the architecture at load time is handled by
        # build_model() in models.py using the same config.
        weights_path = model_dir / f"{model_name}.weights.h5"
        model.save_weights(weights_path)
        mlflow.log_artifact(str(weights_path))

        # Also try logging via mlflow.keras for the MLflow Model Registry /
        # serving format. NOTE: this internally calls model.save() (full
        # config + weights), which can hit the same JSON serialization
        # crash described above for backbones like EfficientNetB0 whose
        # config embeds raw TensorFlow tensors. The weights file above is
        # already saved and logged as an artifact regardless, so this is
        # treated as a best-effort bonus, not a hard requirement.
        try:
            mlflow.keras.log_model(model, artifact_path="keras_model")
        except Exception as e:
            print(
                f"[train] WARNING: mlflow.keras.log_model failed for "
                f"'{model_name}' (non-fatal, weights were already saved "
                f"separately above): {e}"
            )

        print(f"[train] Run complete. MLflow run_id: {run.info.run_id}")
        print(f"[train] Model weights saved to: {weights_path}")


if __name__ == "__main__":
    main()
