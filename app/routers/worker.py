"""
routers/worker.py — GET /api/worker/status
"""
from flask import Blueprint, jsonify

import app.state as state
from app.config import config

worker_bp = Blueprint("worker", __name__, url_prefix="/api/worker")


@worker_bp.route("/status", methods=["GET"])
def worker_status():
    active_evals = state.get_active_eval_count()
    active_trains = len(state.active_training_jobs)
    queued = len(state.job_queue)

    return jsonify({
        "active_training_jobs": active_trains,
        "queued_training_jobs": queued,
        "active_eval_jobs":     active_evals,
        "max_eval_slots":       config.GPU_EVAL_SLOTS,
        "max_train_slots":      config.MAX_CONCURRENT_JOBS,
        "gpu_free":             active_trains == 0 and active_evals == 0,
    }), 200
