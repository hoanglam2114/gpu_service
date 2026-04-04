#!/usr/bin/env bash
# start.sh — One-command launcher cho gpu_service
#
# Cách dùng:
#   bash start.sh          → chạy qua Docker (mặc định, khuyến nghị)
#   bash start.sh --local  → chạy thẳng Python (không dùng Docker)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-}"

# ── Kiểm tra .env ────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    echo "📋 .env không tồn tại — copy từ .env.example"
    cp .env.example .env
    echo ""
    echo "⚠️  Hãy điền các giá trị vào .env rồi chạy lại:"
    echo "     NGROK_TOKEN=..."
    echo "     ANTHROPIC_API_KEY=..."
    echo ""
    exit 1
fi

# ── Chế độ Docker (mặc định) ────────────────────────────────────────────────
if [ "$MODE" != "--local" ]; then
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker chưa được cài."
        echo "   Windows/Mac: https://www.docker.com/products/docker-desktop"
        echo "   Linux:       https://docs.docker.com/engine/install/"
        exit 1
    fi

    # Kiểm tra GPU có khả dụng không (không bắt buộc — chỉ cảnh báo)
    if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        echo "⚠️  Không detect được GPU trong Docker."
        echo "   Kiểm tra lại:"
        echo "   • Docker Desktop → Settings → Resources → WSL Integration: bật ON"
        echo "   • Đã cài NVIDIA driver trên Windows (không phải trong WSL)"
        echo "   • Windows: winget install NVIDIA.CUDA"
        echo ""
        read -rp "Tiếp tục mà không có GPU? (y/N): " confirm
        [[ "$confirm" =~ ^[yY]$ ]] || exit 1
    fi

    echo "🐳 Khởi động gpu_service qua Docker..."
    docker compose up --build
    exit 0
fi

# ── Chế độ local (--local) ──────────────────────────────────────────────────
echo "🐍 Khởi động gpu_service trực tiếp (không Docker)..."

echo "📦 Cài requirements..."
pip install -r requirements.txt -q

python - <<'EOF'
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
EOF

echo "🚀 Chạy main.py..."
python main.py
