"""
src/models.py

Model factory for transfer-learning based brain tumor classification.
Supports four backbones, all using ImageNet pretrained weights with a
custom classification head:

    - resnet50
    - vgg16
    - densenet121
    - efficientnetb0

Each backbone's native Keras preprocessing function is used so inputs
are normalized correctly for that specific architecture.
"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50,
    VGG16,
    DenseNet121,
    EfficientNetB0,
)
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess


BACKBONE_REGISTRY = {
    "resnet50": {
        "constructor": ResNet50,
        "preprocess": resnet50_preprocess,
    },
    "vgg16": {
        "constructor": VGG16,
        "preprocess": vgg16_preprocess,
    },
    "densenet121": {
        "constructor": DenseNet121,
        "preprocess": densenet_preprocess,
    },
    "efficientnetb0": {
        "constructor": EfficientNetB0,
        "preprocess": efficientnet_preprocess,
    },
}


def get_preprocess_fn(model_name: str):
    """Return the correct Keras preprocessing function for a backbone name."""
    model_name = model_name.lower()
    if model_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Valid options: {list(BACKBONE_REGISTRY.keys())}"
        )
    return BACKBONE_REGISTRY[model_name]["preprocess"]


def build_model(
    model_name: str,
    input_shape=(224, 224, 3),
    num_classes: int = 4,
    dropout_rate: float = 0.3,
    dense_units: int = 256,
    freeze_backbone: bool = True,
    fine_tune_last_n_layers: int = 0,
    pretrained_weights: str = "imagenet",
):
    """
    Build a transfer-learning classification model.

    Args:
        model_name: one of 'resnet50', 'vgg16', 'densenet121', 'efficientnetb0'
        input_shape: input image shape (H, W, C)
        num_classes: number of output classes
        dropout_rate: dropout applied before the final dense layer
        dense_units: size of the intermediate dense layer
        freeze_backbone: if True, freezes all backbone layers (pure transfer learning)
        fine_tune_last_n_layers: if > 0, unfreezes the last N layers of the
            backbone for fine-tuning (only applied when freeze_backbone=True)
        pretrained_weights: passed through to the Keras Applications
            constructor's `weights` argument. Use "imagenet" (default) when
            training from scratch. Use None when about to immediately
            overwrite all weights via load_weights() (e.g. in
            load_trained_model() below) - this skips an unnecessary
            ImageNet weights download/load step.

    Returns:
        A compiled-ready (uncompiled) tf.keras.Model. Compile separately so
        the caller controls optimizer/loss/metrics (done in train.py).
    """
    model_name = model_name.lower()
    if model_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Valid options: {list(BACKBONE_REGISTRY.keys())}"
        )

    constructor = BACKBONE_REGISTRY[model_name]["constructor"]

    base_model = constructor(
        include_top=False,
        weights=pretrained_weights,
        input_shape=input_shape,
    )
    base_model.trainable = not freeze_backbone

    if freeze_backbone and fine_tune_last_n_layers > 0:
        for layer in base_model.layers[-fine_tune_last_n_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=f"{model_name}_brain_tumor_classifier")
    return model


def find_model_path(model_dir, model_name: str):
    """
    Locate a trained model artifact for `model_name` inside `model_dir`.
    Checks, in order of preference:
      1. {model_name}.weights.h5  - weights-only (current save format)
      2. {model_name}.h5          - full model, legacy format (older runs)
      3. {model_name}.keras       - full model, Keras v3 format (if ever used)

    Args:
        model_dir: a Path (or path-like) to the directory containing
            saved model files (typically the configured models/ dir).
        model_name: backbone name, e.g. 'resnet50'.

    Returns:
        A tuple (path: Path, kind: str) where kind is one of
        "weights", "full_h5", "full_keras".

    Raises:
        FileNotFoundError if no recognized file exists.
    """
    from pathlib import Path

    model_dir = Path(model_dir)
    weights_path = model_dir / f"{model_name}.weights.h5"
    h5_path = model_dir / f"{model_name}.h5"
    keras_path = model_dir / f"{model_name}.keras"

    if weights_path.exists():
        return weights_path, "weights"
    if h5_path.exists():
        return h5_path, "full_h5"
    if keras_path.exists():
        return keras_path, "full_keras"

    raise FileNotFoundError(
        f"No trained model found for '{model_name}' in '{model_dir}' "
        f"(looked for .weights.h5, .h5, and .keras). "
        f"Run: python src/train.py --model_name {model_name}"
    )


def load_trained_model(model_dir, model_name: str, model_kwargs: dict):
    """
    Load a fully-built, weight-restored model for `model_name`, regardless
    of which save format was used (weights-only or full model file).

    Args:
        model_dir: directory containing saved model files.
        model_name: backbone name, e.g. 'resnet50'.
        model_kwargs: kwargs to pass to build_model() when the architecture
            needs to be reconstructed from code (weights-only case). Should
            match exactly what was used during training (input_shape,
            num_classes, dropout_rate, dense_units, freeze_backbone, etc.)
            so the rebuilt architecture's layer order matches the saved
            weights.

    Returns:
        A Keras model with trained weights loaded, ready for inference.
    """
    from tensorflow.keras.models import load_model as _load_model

    path, kind = find_model_path(model_dir, model_name)

    if kind == "weights":
        # pretrained_weights=None: skip downloading/loading ImageNet
        # weights since load_weights() below immediately overwrites
        # everything with the actually-trained weights anyway.
        model = build_model(model_name=model_name, pretrained_weights=None, **model_kwargs)
        model.load_weights(path)
        return model

    # Legacy full-model formats (older runs saved before the weights-only fix)
    return _load_model(path)


def list_available_models():
    return list(BACKBONE_REGISTRY.keys())
