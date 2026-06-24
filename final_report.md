# SICIP Final Project Report

**Title:** Multi-Class Brain Tumor Classification Using Transfer Learning on MRI Images

**Student Name:** Fahim Alavi Ridoy
**Student ID:** 
**Program:** Skills for Industry Competitiveness and Innovation Program (SICIP), BRAC University
**Course:** Machine Learning and Deep Learning Certification
**Submission Date:** _____________________

> ⚠️ Before submitting: copy this content into the official
> `SICIP_TEMPLATE_final_report` template (https://www.rb.gy/lie1pp),
> match its required formatting/sections exactly, export to PDF, and
> name the file `SICIP_<YOUR_NAME>_<YOUR_ID>.pdf`.

---

## 1. Introduction

Brain tumors are among the most life-threatening neurological conditions,
and early, accurate diagnosis significantly improves treatment outcomes.
Magnetic Resonance Imaging (MRI) is the standard non-invasive modality for
visualizing brain tumors, but manual interpretation by radiologists is
time-consuming and subject to inter-observer variability, especially in
high-patient-volume settings.

This project applies deep learning — specifically **transfer learning**
with pretrained convolutional neural networks — to automatically classify
brain MRI scans into one of four categories: **glioma**, **meningioma**,
**pituitary tumor**, or **no tumor**. Four widely used CNN architectures
(ResNet50, VGG16, DenseNet121, EfficientNetB0) are trained, tracked, and
compared to identify which transfers best to this medical imaging task.

## 2. Problem Statement

Given a brain MRI image, the goal is to build a multi-class image
classification system that:

1. Accepts an MRI scan (image) as input.
2. Predicts which of four classes the scan belongs to: glioma, meningioma,
   pituitary tumor, or no tumor.
3. Returns a confidence score for the prediction.
4. Is deployable through a Docker-served interactive application that a
   user can upload a sample image to and immediately see results.

This is framed as a supervised multi-class image classification problem.

## 3. Dataset

- **Source:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
  by Masoud Nickparvar on Kaggle.
- **Composition:** 7,023 MRI images combining three source datasets
  (figshare, SARTAJ, Br35H), pre-split into:
  - `Training/`: glioma, meningioma, pituitary, notumor subfolders
  - `Testing/`: same four classes, held out for final evaluation
- **Class balance:** Roughly balanced across the four classes within the
  training split.
- **Preprocessing:**
  - All images resized to 224×224 to match the input requirements of all
    four backbones.
  - Architecture-specific normalization applied via each model's native
    Keras `preprocess_input` function.
  - Light augmentation (rotation ±10°, width/height shift 5%, zoom 10%,
    horizontal flip) applied only to the training subset to improve
    generalization without distorting anatomical structure.
  - 15% of the `Training/` split held out as a validation set; the
    `Testing/` split is never seen during training and is used only for
    final evaluation.
- **Why this dataset fits the task:** It is the de facto standard public
  benchmark for multi-class brain tumor MRI classification research, with
  a clean directory structure and sufficient volume to fine-tune
  ImageNet-pretrained CNNs.

## 4. Methodology

### 4.1 Transfer Learning Approach

Four pretrained CNN backbones (ImageNet weights) are used as fixed feature
extractors:

| Backbone | Why included |
|---|---|
| **ResNet50** | Residual connections handle deeper networks well; strong general-purpose baseline |
| **VGG16** | Simple, well-understood architecture; useful as a classical baseline |
| **DenseNet121** | Dense connectivity improves gradient flow and feature reuse; parameter-efficient |
| **EfficientNetB0** | Compound-scaled architecture designed for efficiency; strong accuracy-per-parameter |

For each backbone:
1. The convolutional base is loaded with `include_top=False,
   weights="imagenet"` and **frozen** (`trainable=False`).
2. A custom classification head is appended:
   `GlobalAveragePooling2D → Dense(256, relu) → Dropout(0.3) →
   Dense(4, softmax)`.
3. Only the new head is trained, making this a pure transfer-learning
   setup (the config supports optional fine-tuning of the last N backbone
   layers via `fine_tune_last_n_layers`, not used in the baseline runs).

### 4.2 Training Configuration

| Setting | Value |
|---|---|
| Image size | 224 × 224 × 3 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Loss function | Categorical cross-entropy |
| Epochs | 15 (with early stopping, patience=4 on val_loss) |
| LR scheduling | ReduceLROnPlateau (factor=0.5, patience=2) |

**Hardware:** Training was run on CPU (AMD Ryzen 5 3600). GPU acceleration
via TensorFlow-DirectML (targeting the system's AMD RX 5500 XT) was
evaluated but encountered an unresolved bug in the plugin — a duplicate
OpKernel registration for `StatelessRandomGetKeyCounter` that crashes
during model construction — and Microsoft has paused active development
of the plugin. CPU training remained practical because the backbone is
frozen (Section 4.1): only the lightweight classification head is
actually trained each epoch, while the pretrained backbone performs fast
forward passes only.

### 4.3 Experiment Tracking

All four training runs are tracked with **MLflow**, logging:
- Hyperparameters (model name, image size, batch size, learning rate,
  dropout rate, dense units, epochs, early stopping patience)
- Per-epoch training and validation loss/accuracy
- Final test loss/accuracy on the held-out `Testing/` split
- Artifacts: training curve plots, the trained model file, and the
  class-index mapping

## 5. MLflow Experiments

Four independent MLflow runs were executed — one per backbone
architecture — under the experiment `brain-tumor-mri-classification`.

> 📸 *Insert screenshot:* MLflow experiment list page showing all 4 runs
> with their test accuracy column (`screenshots/mlflow_runs.png`).

> 📸 *Insert screenshot:* Detail page for the best-performing run showing
> logged parameters and the metrics chart over epochs
> (`screenshots/training_result.png`).

### Summary table (fill in after running `src/compare_models.py`)

| Model | Test Accuracy | Test Loss | Val Accuracy | Epochs Run |
|---|---|---|---|---|
| ResNet50 | _____ | _____ | _____ | _____ |
| VGG16 | _____ | _____ | _____ | _____ |
| DenseNet121 | _____ | _____ | _____ | _____ |
| EfficientNetB0 | _____ | _____ | _____ | _____ |

*(Generate this table automatically: `python src/compare_models.py`
produces `artifacts/model_comparison.csv` and a bar chart.)*

## 6. Results

> 📸 *Insert confusion matrix screenshots for each model from
> `artifacts/{model_name}_confusion_matrix.png`.*

**Discussion points to fill in once training completes:**
- Which backbone achieved the highest test accuracy, and by how much?
- Which class pairs were most frequently confused (commonly glioma vs.
  meningioma in this dataset)?
- Did deeper/larger models (ResNet50) outperform lighter ones
  (EfficientNetB0), or did parameter efficiency win out?
- How did validation accuracy track test accuracy — any signs of
  overfitting on a particular backbone?

## 7. Prediction Pipeline

The trained models are served two ways:

1. **CLI inference** (`src/predict.py`) — for quick single-image testing:
   ```
   python src/predict.py --input path/to/sample.jpg --model_name resnet50
   ```
2. **Interactive Streamlit app** (`app/app.py`) — a user uploads an MRI
   image, selects a backbone from a dropdown, and receives the predicted
   class plus a per-class confidence breakdown.

Both paths share the same preprocessing logic (`src/models.py`) to
guarantee consistent results between CLI and UI predictions.

## 8. Docker Setup

The Streamlit app is containerized for reproducible deployment:

```bash
docker build -t brain-tumor-mri-app:1.0 .
docker run -p 8501:8501 brain-tumor-mri-app:1.0
```

The container bundles the application code, configuration, trained model
weights, and the class-index mapping, then exposes the Streamlit UI on
port 8501.

> 📸 *Insert screenshot:* the running Docker container terminal output
> (`screenshots/docker_app_running.png`).

> 📸 *Insert screenshot:* the Streamlit UI in a browser showing an
> uploaded MRI image and its prediction (`screenshots/demo_output.png`).

## 9. Limitations

- Only 2D slice-level classification is performed; no 3D volumetric
  context is used.
- The `notumor` class originates from a different source dataset (Br35H)
  than the tumor classes, introducing a possible domain-shift shortcut the
  model could exploit instead of learning genuine tumor-absence features.
- All backbones were trained with frozen weights only; deeper fine-tuning
  was not exhaustively explored due to CPU-only training time constraints
  (GPU acceleration via TensorFlow-DirectML was evaluated but hit an
  unresolved bug in the plugin itself; see Section 4 of this report).
- Evaluation is limited to the dataset's own held-out test split; no
  external validation on scans from a different hospital or scanner was
  performed, so real-world generalization is unverified.
- This system is a research/educational prototype, not a validated
  clinical decision-support tool.

## 10. Future Work

- Fine-tune the last N convolutional layers of each backbone (already
  supported via `fine_tune_last_n_layers` in `configs/config.yaml`) and
  compare against the frozen-backbone baseline.
- Add Grad-CAM or saliency-map visualizations to make predictions
  interpretable for clinical review.
- Conduct a systematic hyperparameter search (learning rate, dropout,
  dense layer size) with MLflow tracking each trial.
- Extend the Docker image to expose a REST API (e.g., FastAPI) alongside
  the Streamlit UI for programmatic integration.
- Validate on an external, independently sourced MRI dataset to assess
  true generalization.

## 11. Conclusion

This project demonstrates a complete, reproducible deep learning pipeline
for multi-class brain tumor classification using transfer learning,
comparing four CNN architectures under identical training conditions with
full MLflow experiment tracking. The system is deployed as a Dockerized,
user-facing prediction application, satisfying the end-to-end requirements
of the SICIP final certification project.

## 12. References

1. Nickparvar, M. *Brain Tumor MRI Dataset*. Kaggle.
   https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
2. He, K. et al. (2016). *Deep Residual Learning for Image Recognition*
   (ResNet).
3. Simonyan, K. & Zisserman, A. (2014). *Very Deep Convolutional Networks
   for Large-Scale Image Recognition* (VGG16).
4. Huang, G. et al. (2017). *Densely Connected Convolutional Networks*
   (DenseNet).
5. Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for
   Convolutional Neural Networks*.
