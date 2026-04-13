"""
main.py — Flask entrypoint.
Đăng ký tất cả blueprints, khởi động ngrok và job manager thread.
"""
import os
import threading

from flask import Flask
from pyngrok import ngrok

from app.config import config
from app.routers.train import train_bp
from app.routers.eval import eval_bp
from app.routers.infer import infer_bp
from app.routers.system import system_bp
from app.routers.worker import worker_bp
from app.services.training import job_manager_thread

# ── Môi trường ─────────────────────────────────────────────────────────────
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.LOCAL_CHECKPOINT_BASE, exist_ok=True)

# ── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.register_blueprint(train_bp)
app.register_blueprint(eval_bp)
app.register_blueprint(infer_bp)
app.register_blueprint(system_bp)
app.register_blueprint(worker_bp)


def _start_ngrok() -> str | None:
    """Khởi động ngrok tunnel, trả về public URL hoặc None nếu lỗi.

    Tắt hoàn toàn khi:
      - DISABLE_NGROK=true  (deploy lên server có IP public / H100)
      - NGROK_TOKEN chưa set
    """
    if config.DISABLE_NGROK:
        print("ℹ️  DISABLE_NGROK=true — bỏ qua ngrok, dùng IP server trực tiếp.")
        return None
    if not config.NGROK_TOKEN:
        print("⚠️  NGROK_TOKEN chưa set — bỏ qua ngrok, chỉ chạy local.")
        return None
    try:
        ngrok.set_auth_token(config.NGROK_TOKEN)
        # Đóng tất cả tunnel cũ nếu còn sót
        for t in ngrok.get_tunnels():
            ngrok.disconnect(t.public_url)
        public_url = ngrok.connect(config.FLASK_PORT).public_url
        print("=" * 50)
        print(f"🚀 NGROK URL: {public_url}")
        print("=" * 50)
        return public_url
    except Exception as e:
        print(f"❌ Lỗi khởi động ngrok: {e}")
        return None


if __name__ == "__main__":

    # Job manager chạy trong daemon thread — tự tắt khi main process thoát
    manager = threading.Thread(target=job_manager_thread, daemon=True, name="job-manager")
    manager.start()

    _start_ngrok()

    app.run(
        host="0.0.0.0",
        port=config.FLASK_PORT,
        debug=False,
        use_reloader=False,
    )