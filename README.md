# Multi-Class Brain Tumor Classification Using Transfer Learning on MRI Images

**SICIP Final Certification Project — BRAC University**

A reproducible deep learning pipeline that classifies brain MRI scans into
four categories — **glioma**, **meningioma**, **pituitary tumor**, or **no
tumor** — using transfer learning across four CNN backbones (**ResNet50,
VGG16, DenseNet121, EfficientNetB0**), with full experiment tracking via
**MLflow** and a **Dockerized Streamlit app** for interactive prediction.

> ⚠️ **Disclaimer:** This is an academic course project, not a certified
> medical device. Predictions must never be used for real clinical
> decision-making.

---

## 1. Project Overview

- **Domain:** Medical image analysis with deep learning
- **Problem statement:** Manual MRI interpretation is time-consuming and
  subject to inter-observer variability. This project builds an automated
  classifier that takes a brain MRI image as input and predicts one of four
  classes, comparing four different pretrained CNN architectures to
  determine which transfers best to this task.
- **Expected output:** Given an uploaded MRI image, the system returns the
  predicted tumor class plus a confidence score for all four classes.

---

## 2. Dataset

- **Source:** [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
  (`masoudnickparvar/brain-tumor-mri-dataset`)
- **Description:** 7,023 brain MRI images combining the figshare, SARTAJ,
  and Br35H datasets, pre-split into `Training/` and `Testing/` folders,
  each containing four class subfolders: `glioma`, `meningioma`,
  `notumor`, `pituitary`.
- **Why this dataset fits the task:** It is the standard public benchmark
  for multi-class brain tumor MRI classification, has a clean class
  structure ready for `ImageDataGenerator.flow_from_directory`, and is
  large enough to fine-tune ImageNet-pretrained backbones without severe
  overfitting.
- **Download method:** Fully automatic — see Section 4. The dataset is
  **never committed to this repository** (see `.gitignore`); it is fetched
  fresh into `data/` via `src/data_download.py`.

---

## 3. Project Structure

```
final-project/
|-- README.md
|-- final_report.md
|-- requirements.txt
|-- .gitignore
|-- Dockerfile
|-- src/
|   |-- data_download.py     # downloads dataset from Kaggle into data/
|   |-- models.py            # model factory for all 4 backbones
|   |-- train.py             # trains one backbone + logs to MLflow
|   |-- evaluate.py          # confusion matrix + classification report
|   |-- predict.py           # single-image CLI inference
|   `-- compare_models.py    # aggregates MLflow runs into a comparison table
|-- app/
|   `-- app.py               # Streamlit prediction UI (Dockerized)
|-- notebooks/
|   `-- exploration.ipynb    # EDA: class balance, sample images, sizes
|-- configs/
|   `-- config.yaml          # single source of truth for all hyperparameters
|-- mlruns/                  # MLflow experiment tracking data (committed)
|-- screenshots/             # README evidence screenshots
|-- models/                  # trained weights (.weights.h5, gitignored, regenerated via train.py)
|-- data/                    # downloaded dataset (gitignored)
`-- artifacts/                # plots, reports, class mapping (gitignored except via MLflow)
```

---

## 4. Setup

**Python version:** 3.10 (recommended; matches `tensorflow==2.10.0` pin)

```bash
# 1. Clone the repo
git clone https://github.com/FahimAlaviRidoy/brain-tumor-mri-classifier_SICIP_Final_Project.git
cd final-project

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

### Hardware / training notes

This project trains on **CPU**. GPU acceleration was evaluated using
TensorFlow-DirectML (a community plugin enabling DirectX 12-based GPU
acceleration for AMD/Intel/NVIDIA GPUs on Windows, used here with an AMD
RX 5500 XT), but the plugin has an unresolved bug — a duplicate OpKernel
registration for `StatelessRandomGetKeyCounter` — that crashes during
model construction with this architecture. Microsoft has also paused
active development of the plugin, so this is a known, currently
unresolved limitation of the tool rather than an issue with this
project's code.

```bash
pip install tensorflow==2.10.0
```

Training remains practical on CPU because **the backbone is frozen**
(`freeze_backbone: true` in `configs/config.yaml`) — only the small
classification head (a few hundred thousand parameters) is actually
trained each epoch; the pretrained backbone only does forward passes.

---

## 5. Data Fetching

Dataset download is fully automatic — no manual Kaggle download needed:

```bash
python src/data_download.py
```

This uses `kagglehub` to pull the dataset into `data/Training/` and
`data/Testing/`. On first run it may prompt for Kaggle API credentials:

1. Go to <https://www.kaggle.com/settings> → **Create New Token**
2. open terminal in your pc and run this with your kaggle API token.
```bash
mkdir $env:USERPROFILE\.kaggle -Force
   "<your-new-token>" | Out-File -FilePath $env:USERPROFILE\.kaggle\access_token -Encoding ascii -NoNewline

```

`data/` is excluded from version control via `.gitignore` — every reviewer
must run this script after cloning.

---

## 6. Training

Train each backbone independently (one MLflow run per command):

```bash
python src/train.py --config configs/config.yaml --model_name resnet50
python src/train.py --config configs/config.yaml --model_name vgg16
python src/train.py --config configs/config.yaml --model_name densenet121
python src/train.py --config configs/config.yaml --model_name efficientnetb0
```

### Key hyperparameters (`configs/config.yaml`)

| Hyperparameter | Value | Notes |
|---|---|---|
| `image_size` | 224×224 | Standard input size for all 4 backbones |
| `batch_size` | 32 | |
| `epochs` | 15 | With early stopping (`patience=4` on val_loss) |
| `learning_rate` | 1e-4 | Adam optimizer |
| `freeze_backbone` | true | Pure transfer learning — only the classification head trains |
| `dropout_rate` | 0.3 | Regularization before final dense layer |
| `dense_units` | 256 | Size of the intermediate dense layer |
| `val_split` | 0.15 | Carved out of `Training/` (separate from the held-out `Testing/` set) |

Each backbone uses its **own native Keras preprocessing function**
(`resnet50.preprocess_input`, `vgg16.preprocess_input`, etc.) so inputs are
normalized exactly as each architecture expects.

---

## 7. MLflow Experiment Tracking

Start the MLflow UI to inspect all runs:

```bash
mlflow ui --port 5000
```

Then open `http://localhost:5000` in a browser.

### What is logged per run

- **Parameters:** model name, image size, batch size, epochs, learning
  rate, optimizer, dropout rate, dense units, early stopping patience
- **Metrics (per epoch):** train/val loss, train/val accuracy
- **Final metrics:** test loss, test accuracy (on the held-out `Testing/` set)
- **Artifacts:** training curve plot (`*_training_curves.png`), the trained
  weights file, and the `class_indices.json` mapping

> **Note on model format:** models are saved as **weights-only**
> (`{model_name}.weights.h5` via `model.save_weights()`), and reloaded by
> rebuilding the architecture from code (`build_model()` in
> `src/models.py`) before calling `model.load_weights()`. This project
> pins TensorFlow 2.10 (for Windows DirectML compatibility, even though
> DirectML acceleration itself wasn't usable — see the hardware note
> below), and on that version, saving a *full* model
> (`model.save(...)`, regardless of file extension) routes through a
> legacy saver that serializes the entire architecture config to JSON.
> EfficientNetB0's config embeds raw TensorFlow tensors (normalization
> statistics) that this JSON step cannot handle, crashing at save time.
> Weights-only saving skips config serialization entirely, sidestepping
> the bug, and is used uniformly across all 4 backbones for consistency.
> `src/models.py`'s `load_trained_model()` helper also transparently
> supports loading older full-model `.h5`/`.keras` files if those exist
> from earlier runs, so no retraining was needed for backbones that had
> already saved successfully before this fix.

### Screenshots

> 📸 *Add screenshots here after running training:*
> - `screenshots/mlflow_runs.png` — the MLflow experiment list showing all 4 runs
> - `screenshots/training_result.png` — a run's metrics/parameters detail page

At least **4 runs** are produced (one per backbone), satisfying the "≥2
experiments" requirement with room for direct architecture comparison.

After training all 4 backbones, generate a side-by-side comparison:

```bash
python src/compare_models.py
```

This produces `artifacts/model_comparison.csv` and
`artifacts/model_comparison_chart.png`.

---

## 8. Course Techniques Used

| Technique | Where it appears |
|---|---|
| **Transfer learning** | `src/models.py` — all 4 backbones use frozen ImageNet-pretrained convolutional bases (`include_top=False, weights="imagenet"`) |
| **Data preprocessing & augmentation** | `src/train.py` — `ImageDataGenerator` with rotation/shift/zoom/flip augmentation + architecture-specific preprocessing functions |
| **CNN architectures (ResNet50, VGG16, DenseNet121, EfficientNetB0)** | `src/models.py` |
| **MLflow experiment tracking** | `src/train.py` — params, metrics, artifacts, model logging |
| **Model evaluation** | `src/evaluate.py` — confusion matrix, precision/recall/F1, classification report |
| **Callbacks (early stopping, LR scheduling)** | `src/train.py` — `EarlyStopping`, `ReduceLROnPlateau` |
| **Reproducible project structure / config-driven pipeline** | `configs/config.yaml` is the single source of truth consumed by every script |
| **API/UI serving** | `app/app.py` — Streamlit upload-and-predict interface |
| **Dockerization** | `Dockerfile` — containerized prediction app |

---

## 9. Evaluation

Run detailed evaluation (confusion matrix + per-class metrics) for any
trained model:

```bash
python src/evaluate.py --model_name resnet50
python src/evaluate.py --model_name vgg16
python src/evaluate.py --model_name densenet121
python src/evaluate.py --model_name efficientnetb0
```

Outputs (saved to `artifacts/`):
- `{model_name}_confusion_matrix.png`
- `{model_name}_classification_report.json` / `.txt`

> 📸 *Add a screenshot of a confusion matrix here, e.g.
> `screenshots/training_result.png` or a dedicated `evaluation_result.png`.*

**Interpretation:** Compare test accuracy and per-class F1 scores across
the four backbones using `artifacts/model_comparison.csv`. Pay particular
attention to the `glioma` vs `meningioma` confusion pair, which is the most
common failure mode in this dataset due to visual similarity between the
two tumor types in certain slice orientations.

---

## 10. Single-Image CLI Prediction

```bash
python src/predict.py --input path/to/sample.jpg --model_name resnet50
```

Example output:

```
Input image:      path/to/sample.jpg
Model used:       resnet50
Predicted class:  glioma
Confidence:       94.32%

All class scores:
  glioma       94.32%
  meningioma   3.81%
  pituitary    1.42%
  notumor      0.45%
```

---

## 11. Dockerized Prediction App

The Streamlit app (`app/app.py`) lets a user **upload an MRI image**,
choose which trained backbone to run, and see the predicted class with a
confidence breakdown for all classes.

### Build

```bash
docker build -t brain-tumor-mri-app:1.0 .
```

### Run

```bash
docker run -p 8501:8501 brain-tumor-mri-app:1.0
```

Then open `http://localhost:8501` in a browser, upload an MRI image, and
view the prediction.

> ⚠️ **Important:** Train at least one model (`python src/train.py
> --model_name resnet50`) **before** building the Docker image — the
> `models/` and `artifacts/` directories are copied into the image and
> must contain a trained `.weights.h5` file and `class_indices.json`.

> 📸 *Add a screenshot here:* `screenshots/docker_app_running.png`
> showing the running container and a successful prediction, plus
> `screenshots/demo_output.png` showing the upload + result UI.

---

## 12. Limitations and Future Work

**Limitations:**
- Only 2D axial-style slice images are used; no 3D volumetric context is
  modeled, which may limit performance on borderline cases.
- The `notumor` class is sourced from a different dataset (Br35H) than the
  tumor classes (figshare/SARTAJ), introducing a potential domain shift the
  model could exploit as a shortcut feature rather than learning genuine
  tumor-absence patterns.
- All 4 backbones use frozen pretrained weights with only the
  classification head trained; fine-tuning deeper layers
  (`fine_tune_last_n_layers` in the config) was not exhaustively explored
  due to training time constraints.
- Evaluated only on the dataset's own held-out test split — no external
  validation on MRI scans from a different hospital/scanner was performed,
  so real-world generalization is unverified.

**Future work:**
- Fine-tune the last N layers of each backbone (config option already
  supports this) and compare against the frozen-backbone baseline.
- Add Grad-CAM visualizations to the Streamlit app so predictions are
  interpretable (showing which image regions drove the classification).
- Run a proper hyperparameter search (learning rate, dropout, dense units)
  with MLflow tracking each trial.
- Expand the Docker image to also serve a REST API endpoint (FastAPI)
  alongside the Streamlit UI for programmatic access.

---

## 13. Final Project Report

The completed final report is included in this repository as `final_report.md` and submitted separately as a PDF.

---

