"""
app/config.py — Cấu hình toàn cục đọc từ biến môi trường.
Thay thế google.colab.userdata bằng python-dotenv.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Ngrok
    NGROK_TOKEN: str = os.environ.get("NGROK_TOKEN", "")

    # Anthropic
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

    # Backend Node.js
    BACKEND_URL: str = os.environ.get("BACKEND_URL", "http://localhost:3000")

    # Flask
    FLASK_PORT: int = int(os.environ.get("FLASK_PORT", 5000))

    # GPU / Eval slots
    GPU_EVAL_SLOTS: int = int(os.environ.get("GPU_EVAL_SLOTS", 3))

    # Clustering
    NUM_CLUSTERS: int = int(os.environ.get("NUM_CLUSTERS", 7))
    DBSCAN_EPS: float = float(os.environ.get("DBSCAN_EPS", 0.05))
    DBSCAN_MIN_SAMPLES: int = int(os.environ.get("DBSCAN_MIN_SAMPLES", 3))

    # Storage paths
    UPLOAD_FOLDER: str = os.environ.get("UPLOAD_FOLDER", "./dataset_uploads")
    LOCAL_CHECKPOINT_BASE: str = os.environ.get("LOCAL_CHECKPOINT_BASE", "/tmp/checkpoints")

    # Training defaults
    MAX_CONCURRENT_JOBS: int = 1


config = Config()
