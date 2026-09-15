FROM docker.io/pytorch/pytorch:2.13.0-cuda12.6-cudnn9-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        vim \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages \
        timm \
        datasets \
        huggingface_hub \
        pillow \
        numpy \
        pandas \
        scikit-learn \
        scipy \
        matplotlib \
        tqdm \
        betacal \
        codecarbon \
        einops

RUN pip install --no-cache-dir --break-system-packages \
        lightning \
        wandb \
        transformers==4.48.2 \
        sentencepiece \
        torchmetrics[detection] \
        protobuf \
        accelerate \
        qwen-vl-utils \
        nltk

# ── Project source ────────────────────────────────────────────────────
COPY . /workspace/

CMD ["bash"]