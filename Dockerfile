# ============================================================
# Dockerfile - Brain Tumor MRI Classifier (Streamlit prediction app)
#
# NOTE: This Docker image runs the PREDICTION app only. It uses
# plain tensorflow-cpu (DirectML is a Windows-host-only acceleration
# layer and is not applicable inside a Linux container). Models are
# already trained on the host machine — this image just serves them.
# ============================================================

FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV / Pillow image handling + curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching).
# Use plain tensorflow-cpu inside the container regardless of host OS.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tensorflow-cpu==2.10.0 && \
    grep -v -E "tensorflow|directml" requirements.txt > requirements_container.txt && \
    pip install --no-cache-dir -r requirements_container.txt

# Copy application code, config, trained models, and class mapping.
# (models/ and artifacts/ must exist locally — run train.py before
# building this image; see README for the full workflow.)
COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
COPY models/ ./models/
COPY artifacts/ ./artifacts/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
