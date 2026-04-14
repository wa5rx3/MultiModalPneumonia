FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements_dev.txt .
RUN pip install --no-cache-dir -r requirements_dev.txt

COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY pyproject.toml .

RUN pip install -e . --no-deps

ENV PYTHONPATH=/workspace
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.training.train_multimodal_pneumonia", "--help"]
