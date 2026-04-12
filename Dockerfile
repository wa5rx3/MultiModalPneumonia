FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
# Python 3.11 (ships with this base image)

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

# torch/torchvision are already in the base image — exclude to avoid duplicate installs.
RUN pip install --no-cache-dir \
    $(grep -vE "^(torch|torchvision|torchaudio)==|^--extra-index" requirements.txt | grep -v "^#" | tr '\n' ' ')

COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY pyproject.toml .

RUN pip install -e . --no-deps

# MIMIC data cannot be redistributed — mount at runtime.
# Required mounts:
#   -v /path/to/mimic_cxr_jpg:/workspace/mimic_cxr_jpg:ro
#   -v /path/to/mimic_iv_ed:/workspace/mimic_iv_ed:ro
#   -v /path/to/artifacts:/workspace/artifacts      (preprocessed manifests + model outputs)
#
# PhysioNet credentials required:
#   https://physionet.org/content/mimic-cxr-jpg/
#   https://physionet.org/content/mimic-iv-ed/

ENV PYTHONPATH=/workspace
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.training.train_multimodal_pneumonia", "--help"]
