"""
app/app.py

Streamlit prediction app for brain tumor MRI classification.
Lets a user upload an MRI image, pick which trained backbone to use,
and view the predicted tumor class with confidence scores.

Run locally:
    streamlit run app/app.py

Run via Docker (see Dockerfile / README):
    docker build -t brain-tumor-mri-app:1.0 .
    docker run -p 8501:8501 brain-tumor-mri-app:1.0
"""

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import yaml
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

# Allow importing src/models.py regardless of working directory
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from models import get_preprocess_fn, load_trained_model as _load_trained_model_impl  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"
MODELS_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

MODEL_DISPLAY_NAMES = {
    "resnet50": "ResNet50",
    "vgg16": "VGG16",
    "densenet121": "DenseNet121",
    "efficientnetb0": "EfficientNetB0",
}


@st.cache_resource(show_spinner=False)
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner=False)
def load_class_mapping():
    mapping_path = ARTIFACTS_DIR / "class_indices.json"
    if not mapping_path.exists():
        return None
    with open(mapping_path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


@st.cache_resource(show_spinner=True)
def load_trained_model(model_name: str, model_kwargs_tuple):
    """
    Streamlit-cached wrapper around models.load_trained_model().
    model_kwargs_tuple is a tuple of (key, value) pairs instead of a dict,
    since st.cache_resource requires hashable arguments.
    """
    model_kwargs = dict(model_kwargs_tuple)
    try:
        return _load_trained_model_impl(MODELS_DIR, model_name, model_kwargs)
    except FileNotFoundError:
        return None


def get_available_models():
    """Only show models in the dropdown that have actually been trained
    (checks weights-only .weights.h5, legacy .h5, and .keras formats)."""
    available = []
    for key in MODEL_DISPLAY_NAMES:
        if (
            (MODELS_DIR / f"{key}.weights.h5").exists()
            or (MODELS_DIR / f"{key}.h5").exists()
            or (MODELS_DIR / f"{key}.keras").exists()
        ):
            available.append(key)
    return available


def preprocess_pil_image(pil_img: Image.Image, target_size, preprocess_fn):
    pil_img = pil_img.convert("RGB").resize((target_size[1], target_size[0]))
    arr = keras_image.img_to_array(pil_img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_fn(arr)
    return arr


def main():
    st.set_page_config(
        page_title="Brain Tumor MRI Classifier",
        page_icon="🧠",
        layout="centered",
    )

    st.title("🧠 Brain Tumor MRI Classifier")
    st.caption(
        "Multi-class brain tumor classification using transfer learning "
        "(ResNet50 / VGG16 / DenseNet121 / EfficientNetB0) — SICIP Final Project, BRAC University"
    )

    config = load_config()
    class_mapping = load_class_mapping()
    available_models = get_available_models()

    if not available_models:
        st.error(
            "No trained models found in `models/`. Train at least one model first:\n\n"
            "```\npython src/train.py --model_name resnet50\n```"
        )
        st.stop()

    if class_mapping is None:
        st.error(
            "Class index mapping not found in `artifacts/class_indices.json`. "
            "This file is created automatically by `src/train.py`."
        )
        st.stop()

    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox(
            "Select model architecture",
            options=available_models,
            format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, x),
        )
        st.markdown("---")
        st.markdown(
            "**Classes detected:**\n"
            + "\n".join(f"- {v}" for v in class_mapping.values())
        )
        st.markdown("---")
        st.caption(
            "⚠️ This tool is built for an academic ML course project. "
            "It is **not** a certified medical diagnostic device and must "
            "not be used for real clinical decisions."
        )

    uploaded_file = st.file_uploader(
        "Upload an MRI scan (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(pil_img, caption="Uploaded MRI", use_column_width=True) #use_container_width=True

        with st.spinner(f"Running inference with {MODEL_DISPLAY_NAMES[model_name]} ..."):
            img_h, img_w = config["dataset"]["image_size"]
            model_cfg = config["model"]
            model_kwargs = {
                "input_shape": (img_h, img_w, 3),
                "num_classes": model_cfg["num_classes"],
                "dropout_rate": model_cfg["dropout_rate"],
                "dense_units": model_cfg["dense_units"],
                "freeze_backbone": model_cfg["freeze_backbone"],
                "fine_tune_last_n_layers": model_cfg.get("fine_tune_last_n_layers", 0),
            }
            model = load_trained_model(model_name, tuple(model_kwargs.items()))
            if model is None:
                st.error(
                    f"Failed to load the '{model_name}' model. The saved "
                    f"file may be missing or corrupted — try retraining: "
                    f"`python src/train.py --model_name {model_name}`"
                )
                st.stop()
            preprocess_fn = get_preprocess_fn(model_name)

            img_array = preprocess_pil_image(pil_img, (img_h, img_w), preprocess_fn)
            probs = model.predict(img_array, verbose=0)[0]

            predicted_idx = int(np.argmax(probs))
            predicted_class = class_mapping[predicted_idx]
            confidence = float(probs[predicted_idx])

        with col2:
            st.subheader("Prediction")
            st.metric(label="Predicted Class", value=predicted_class.capitalize())
            st.metric(label="Confidence", value=f"{confidence:.1%}")

            st.markdown("**All class probabilities:**")
            sorted_scores = sorted(
                ((class_mapping[i], float(p)) for i, p in enumerate(probs)),
                key=lambda x: -x[1],
            )
            for cls_name, score in sorted_scores:
                st.progress(score, text=f"{cls_name.capitalize()}: {score:.1%}")

        st.markdown("---")
        st.caption(
            "Note: 'notumor' indicates no tumor was detected in the scan. "
            "Other classes (glioma, meningioma, pituitary) indicate the "
            "predicted tumor type."
        )
    else:
        st.info("👆 Upload an MRI image to get a prediction.")


if __name__ == "__main__":
    main()
