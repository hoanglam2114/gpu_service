"""
routers/eval.py — POST /api/eval/start, GET /api/eval/status/:id, GET /api/eval/result/:id
"""
import json
import os
import threading

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

import app.state as state
from app.config import config
from app.services.evaluation import background_eval_task

eval_bp = Blueprint("eval", __name__, url_prefix="/api/eval")


@eval_bp.route("/start", methods=["POST"])
def start_eval():
    config_str = request.form.get("config")
    if not config_str:
        return jsonify({"error": "Missing 'config' in form data"}), 400

    try:
        cfg = json.loads(config_str)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid config JSON"}), 400

    eval_job_id   = cfg.get("eval_job_id")
    job_id        = cfg.get("job_id")
    hf_repo_id    = cfg.get("hf_repo_id")
    hf_token      = cfg.get("hf_token", "")
    model_max_len = int(cfg.get("model_max_length", 2048))
    judge_model   = cfg.get("judge_model", "claude-haiku-4-5-20251001")

    if not eval_job_id:
        return jsonify({"error": "Missing 'eval_job_id' in config"}), 400
    if not job_id:
        return jsonify({"error": "Missing 'job_id' in config"}), 400
    if not hf_repo_id:
        return jsonify({"error": "Missing 'hf_repo_id'"}), 400

    # Guard: chặn eval nếu đang train (OOM trên single GPU)
    active_train = any(
        v.get("status") == "TRAINING"
        for k, v in state.jobs_db.items()
        if not k.startswith("__eval_tmp_")
    )
    if active_train:
        return jsonify({"error": "train_active", "message": "Đang có train job đang chạy. Vui lòng chờ."}), 409

    if "eval_file" not in request.files or not request.files["eval_file"].filename:
        return jsonify({"error": "Missing 'eval_file'"}), 400

    efile = request.files["eval_file"]
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    eval_file_path = os.path.join(config.UPLOAD_FOLDER, f"{eval_job_id}_{secure_filename(efile.filename)}")
    efile.save(eval_file_path)

    if not state.eval_slot_acquire():
        return jsonify({
            "error": "worker_busy",
            "message": f"Worker đang chạy {state.get_active_eval_count()}/{config.GPU_EVAL_SLOTS} eval job.",
            "active_slots": state.get_active_eval_count(),
            "max_slots": config.GPU_EVAL_SLOTS,
        }), 409

    if eval_job_id in state.eval_jobs_db and state.eval_jobs_db[eval_job_id].get("status") in ("RUNNING", "EVALUATING"):
        state.eval_slot_release()
        return jsonify({"error": f"Eval job {eval_job_id} đang chạy"}), 409

    state.eval_jobs_db[eval_job_id] = {"status": "PENDING", "progress": 0, "logs": [], "job_id": job_id}

    threading.Thread(
        target=background_eval_task,
        args=(eval_job_id, job_id, hf_repo_id, hf_token, eval_file_path, model_max_len, judge_model),
        daemon=True,
    ).start()

    return jsonify({"message": "Eval job started", "eval_job_id": eval_job_id}), 201


@eval_bp.route("/status/<eval_job_id>", methods=["GET"])
def get_eval_status(eval_job_id):
    entry = state.eval_jobs_db.get(eval_job_id)
    if not entry:
        return jsonify({"status": "NOT_FOUND"}), 404

    response = {
        "status":        entry.get("status", "UNKNOWN"),
        "progress":      entry.get("progress", 0),
        "stage":         entry.get("stage", ""),
        "stage_label":   entry.get("stage_label", ""),
        "stage_detail":  entry.get("stage_detail", ""),
        "stage_current": entry.get("stage_current", 0),
        "stage_total":   entry.get("stage_total", 0),
        "logs":          entry.get("logs", []),
    }
    if "current_sample" in entry:
        response["current_sample"] = entry["current_sample"]
    if "error" in entry:
        response["error"] = entry["error"]

    return jsonify(response), 200


@eval_bp.route("/result/<eval_job_id>", methods=["GET"])
def get_eval_result(eval_job_id):
    entry = state.eval_jobs_db.get(eval_job_id)
    if not entry:
        return jsonify({"error": "Eval job không tồn tại"}), 404

    status = entry.get("status")
    if status in ("PENDING", "RUNNING", "EVALUATING"):
        return jsonify({"status": status}), 202
    if status == "FAILED":
        return jsonify({"status": "FAILED", "error": entry.get("error", "")}), 200

    result = entry.get("result")
    if not result:
        return jsonify({"status": "PENDING"}), 202

    return jsonify(result), 200
