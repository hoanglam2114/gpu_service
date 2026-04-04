"""
app/state.py — Global in-memory state.
Tập trung tất cả shared state vào một chỗ thay vì rải rác trong notebook cells.
"""
import threading
import collections
import time

from app.config import config

# ── Training jobs ─────────────────────────────────────────────────────────────
jobs_db: dict = {}
job_queue: collections.deque = collections.deque()
active_training_jobs: set = set()

# ── Eval jobs ─────────────────────────────────────────────────────────────────
eval_jobs_db: dict = {}

# ── Inference logs ────────────────────────────────────────────────────────────
inference_logs_db: dict = {}

# ── Inference model cache ─────────────────────────────────────────────────────
_current_infer_model = None
_current_infer_tokenizer = None
_current_infer_model_id: str | None = None

# ── Watchdog heartbeat ────────────────────────────────────────────────────────
last_heartbeat: float = time.time()

# ── Eval slot semaphore ───────────────────────────────────────────────────────
_eval_semaphore = threading.Semaphore(config.GPU_EVAL_SLOTS)
_active_eval_count: int = 0
_active_eval_lock = threading.Lock()


def eval_slot_acquire() -> bool:
    """Thử lấy slot eval. Trả về True nếu thành công."""
    global _active_eval_count
    acquired = _eval_semaphore.acquire(blocking=False)
    if acquired:
        with _active_eval_lock:
            _active_eval_count += 1
    return acquired


def eval_slot_release() -> None:
    """Giải phóng slot sau khi eval xong."""
    global _active_eval_count
    with _active_eval_lock:
        _active_eval_count = max(0, _active_eval_count - 1)
    _eval_semaphore.release()


def get_active_eval_count() -> int:
    with _active_eval_lock:
        return _active_eval_count
