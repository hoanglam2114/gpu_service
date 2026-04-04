"""
app/callbacks.py — HuggingFace Trainer callbacks.
"""
import time
from transformers import TrainerCallback

import app.state as state


class FlaskProgressCallback(TrainerCallback):
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.start_time: float | None = None

    def on_step_begin(self, args, state_obj, control, **kwargs):
        if self.start_time is None:
            self.start_time = time.time()

    def on_log(self, args, state_obj, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss_val = round(logs["loss"], 4)
            epoch_val = round(state_obj.epoch or 0, 2)

            avg_step_time = 0.0
            if self.start_time and state_obj.global_step > 0:
                avg_step_time = (time.time() - self.start_time) / state_obj.global_step

            state.jobs_db[self.job_id].update({
                "loss": loss_val,
                "epoch": epoch_val,
                "progress": round(
                    (state_obj.global_step / state_obj.max_steps) * 100, 2
                ) if state_obj.max_steps > 0 else 0,
                "avg_step_time": round(avg_step_time, 2),
                "total_steps_per_epoch": (
                    state_obj.max_steps // max(1, int(state_obj.epoch))
                    if state_obj.epoch and state_obj.epoch > 0
                    else state_obj.max_steps
                ),
            })

            log_line = f"Step {state_obj.global_step} | Epoch {epoch_val} | Loss: {loss_val:.4f}"
            state.jobs_db[self.job_id].setdefault("logs", []).append(log_line)

    def on_save(self, args, state_obj, control, **kwargs):
        msg = f"💾 Checkpoint saved locally at step {state_obj.global_step}."
        state.jobs_db[self.job_id].setdefault("logs", []).append(msg)


class EnhancedWatchdogCallback(TrainerCallback):
    """Dừng training nếu mất heartbeat hoặc nhận stop signal."""

    def __init__(self, job_id: str):
        self.job_id = job_id

    def on_step_end(self, args, state_obj, control, **kwargs):
        if time.time() - state.last_heartbeat > 30:
            print("⚠️ Watchdog: No heartbeat for 30s. Stopping...")
            control.should_training_stop = True

        if state.jobs_db.get(self.job_id, {}).get("status") == "STOPPED":
            print(f"🛑 Stop signal for {self.job_id}. Halting...")
            control.should_training_stop = True
