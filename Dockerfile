FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCHDYNAMO_DISABLE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stack đã verified hoạt động với unsloth (từ issue #2775)
RUN pip install --no-cache-dir \
        "transformers==4.52.4" \
        "tokenizers==0.21.1" \
        "accelerate==1.7.0" \
        "bitsandbytes==0.46.0" \
        "peft>=0.15.0" \
        "trl>=0.18.0" \
        "xformers"

# Cài unsloth sau khi stack đã fixed
RUN pip install --no-cache-dir --no-deps \
        "unsloth[cu128-torch270]" \
        unsloth_zoo

RUN pip install --no-cache-dir --force-reinstall \
        "torchvision>=0.26.0" \
        --extra-index-url https://download.pytorch.org/whl/cu128

# App dependencies
RUN pip install --no-cache-dir \
        flask \
        werkzeug \
        pyngrok \
        "nvidia-ml-py>=12.0.0" \
        anthropic \
        duckduckgo-search \
        datasets \
        rouge-score \
        nltk \
        scikit-learn \
        sentence-transformers \
        python-dotenv \
        "huggingface-hub>=0.20.0"\
        hf_transfer

RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

COPY . .

EXPOSE 5000

CMD ["python", "main.py"]