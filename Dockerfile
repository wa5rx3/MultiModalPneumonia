FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# System dependencies for OpenCV and libgomp (XGBoost/OpenMP)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy dependency file first for layer caching
COPY requirements.txt .

# Install Python dependencies.
# torch/torchvision/torchaudio are already in the base image — skip them to avoid
# duplicate installation or index URL conflicts with the +cu124 build tag.
RUN pip install --no-cache-dir \
    $(grep -vE "^(torch|torchvision|torchaudio)==" requirements.txt | grep -v "^#" | tr '\n' ' ')

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY pyproject.toml .

# Install the project package in editable mode (no extra deps — already installed above)
RUN pip install -e . --no-deps

# MIMIC data cannot be redistributed — mount at runtime.
# Required mounts:
#   -v /path/to/mimic_cxr_jpg:/workspace/mimic_cxr_jpg:ro
#   -v /path/to/mimic_iv_ed:/workspace/mimic_iv_ed:ro
#   -v /path/to/artifacts:/workspace/artifacts
#
# PhysioNet credentialing required: https://physionet.org/content/mimic-cxr-jpg/
# and https://physionet.org/content/mimic-iv-ed/

ENV PYTHONPATH=/workspace
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Example: run training help
# docker run --gpus all -v /data/artifacts:/workspace/artifacts \
#   multimodal-pneumonia \
#   python -m src.training.train_multimodal_pneumonia --help
