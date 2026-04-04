# ── Base: CUDA 12.1 + cuDNN 8 + Python 3.11 ───────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCHDYNAMO_DISABLE=1

# ── Hệ thống ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        git wget curl build-essential \
        libssl-dev libffi-dev \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ── Workdir ────────────────────────────────────────────────────────────────
WORKDIR /app

# ── Bước 1: PyTorch + CUDA (cài riêng trước để tận dụng Docker layer cache)
RUN pip install --upgrade pip && \
    pip install \
        torch==2.8.0 \
        torchvision \
        triton \
        --index-url https://download.pytorch.org/whl/cu121

# ── Bước 2: bitsandbytes + xformers (phụ thuộc CUDA)
RUN pip install \
        bitsandbytes>=0.45.0 \
        xformers==0.0.32.post2

# ── Bước 3: unsloth + unsloth_zoo (cần build từ git)
RUN pip install \
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
        "unsloth[base] @ git+https://github.com/unslothai/unsloth" && \
    pip install --upgrade --no-deps tokenizers trl==0.22.2 unsloth unsloth_zoo

# ── Bước 4: transformers pinned version
RUN pip install transformers==5.2.0

# ── Bước 5: flash-linear-attention + causal_conv1d (torch==2.8.0 only)
RUN pip install --no-build-isolation flash-linear-attention causal_conv1d==1.6.0

# ── Bước 6: app dependencies (copy requirements trước để cache)
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Bước 7: NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# ── Bước 8: Copy source code
COPY . .

# ── Port
EXPOSE 5000

# ── Entrypoint
CMD ["python", "main.py"]
