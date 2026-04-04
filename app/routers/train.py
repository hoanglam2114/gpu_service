"""
routers/train.py — POST /api/train/start, GET /api/train/status/:id, POST /api/train/stop/:id
"""
import json
import os

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

import app.state as state
from app.config import config

train_bp = Blueprint("train", __name__, url_prefix="/api/train")


@train_bp.route("/start", methods=["POST"])
def start_training():
    config_str = request.form.get("config")
    if not config_str:
        return jsonify({"error": "Missing 'config' in form data"}), 400

    try:
        parsed_config = json.loads(config_str)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid config JSON format"}), 400

    job_id = parsed_config.get("job_id")
    if not job_id:
        return jsonify({"error": "Missing 'job_id' in config"}), 400

    train_config = {
        "model_name":                  parsed_config.get("model_name", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"),
        "epochs":                      int(parsed_config.get("epochs", 1)),
        "batchSize":                   int(parsed_config.get("batchSize", 2)),
        "learningRate":                float(parsed_config.get("learningRate", 2e-4)),
        "modelMaxLength":              int(parsed_config.get("modelMaxLength", 2048)),
        "r":                           int(parsed_config.get("r", 16)),
        "lora_alpha":                  int(parsed_config.get("lora_alpha", 16)),
        "lora_dropout":                float(parsed_config.get("lora_dropout", 0)),
        "random_state":                int(parsed_config.get("random_state", 3407)),
        "gradient_accumulation_steps": int(parsed_config.get("gradient_accumulation_steps", 4)),
        "warmup_steps":                int(parsed_config.get("warmup_steps", 5)),
        "dataset_hf_id":               parsed_config.get("dataset_hf_id"),
        "push_to_hub":                 parsed_config.get("push_to_hub", False),
        "hf_repo_id":                  parsed_config.get("hf_repo_id"),
        "optim":                       parsed_config.get("optim", "adamw_8bit"),
        "weight_decay":                float(parsed_config.get("weight_decay", 0.01)),
        "lr_scheduler_type":           parsed_config.get("lr_scheduler_type", "linear"),
        "seed":                        int(parsed_config.get("seed", 3407)),
    }
    hf_token = parsed_config.get("hf_token")

    file_path = None
    if "file" in request.files:
        file = request.files["file"]
        if file.filename:
            os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
            file_path = os.path.join(config.UPLOAD_FOLDER, f"{job_id}_{secure_filename(file.filename)}")
            file.save(file_path)

    if not file_path and not train_config.get("dataset_hf_id"):
        return jsonify({"error": "Either a dataset file must be uploaded or 'dataset_hf_id' must be provided."}), 400

    state.job_queue.append((job_id, train_config, file_path, hf_token))
    state.jobs_db[job_id] = {"status": "QUEUED", "progress": 0, "logs": [f"Job {job_id} is in queue."]}

    return jsonify({"message": "Job queued successfully", "job_id": job_id}), 202


@train_bp.route("/status/<job_id>", methods=["GET"])
def get_status(job_id):
    state.last_heartbeat = __import__("time").time()
    return jsonify(state.jobs_db.get(job_id, {"status": "NOT_FOUND"})), 200


@train_bp.route("/stop/<job_id>", methods=["POST"])
def stop_training(job_id):
    if job_id in state.jobs_db:
        state.jobs_db[job_id]["status"] = "STOPPED"
        state.jobs_db[job_id].setdefault("logs", []).append("🛑 Stop request received. Training will halt at the next step.")
        return jsonify({"message": "Stop signal sent"}), 200
    return jsonify({"error": "Job not found"}), 404
