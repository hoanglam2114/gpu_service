"""
routers/system.py — GET /api/system/resources, POST /api/cluster, DELETE /api/cluster/cache, POST /api/cluster/filter
"""
import pynvml
from flask import Blueprint, request, jsonify

import app.state as state
from app.config import config
from app.services.clustering import ClusteringService

system_bp = Blueprint("system", __name__)

pynvml.nvmlInit()
_gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Singleton clustering service (load model once at startup)
_clustering_service = ClusteringService()


@system_bp.route("/api/system/resources", methods=["GET"])
def get_resources():
    info = pynvml.nvmlDeviceGetMemoryInfo(_gpu_handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(_gpu_handle)
    vram_used_mb  = info.used  // 1024 ** 2
    vram_total_mb = info.total // 1024 ** 2
    vram_free_mb  = info.free  // 1024 ** 2

    active = state.get_active_eval_count()
    VRAM_MIN_FREE_MB = 5120

    return jsonify({
        "vram_used_mb":     vram_used_mb,
        "vram_total_mb":    vram_total_mb,
        "vram_free_mb":     vram_free_mb,
        "gpu_util":         util.gpu,
        "active_evals":     active,
        "max_evals":        config.GPU_EVAL_SLOTS,
        "can_create_eval":  (active < config.GPU_EVAL_SLOTS) and (vram_free_mb >= VRAM_MIN_FREE_MB),
        "vram_min_free_mb": VRAM_MIN_FREE_MB,
    }), 200


@system_bp.route("/api/cluster", methods=["POST"])
def cluster():
    body = request.get_json(force=True, silent=True)
    if not body or not isinstance(body.get("data"), list):
        return jsonify({"error": "Request body phải có trường 'data' là array"}), 400

    data = body["data"]
    if not data:
        return jsonify({"data": [], "assignments": [], "groups": []}), 200

    for i, item in enumerate(data):
        if not isinstance(item, dict) or "messages" not in item:
            return jsonify({"error": f"Item index {i} thiếu trường 'messages'"}), 400

    result = _clustering_service.cluster(data)
    return jsonify(result), 200


@system_bp.route("/api/cluster/cache", methods=["DELETE"])
def clear_cluster_cache():
    _clustering_service.clear_cache()
    return jsonify({"message": "Cache đã được xoá."}), 200


@system_bp.route("/api/cluster/filter", methods=["POST"])
def cluster_filter():
    body = request.get_json(force=True, silent=True) or {}
    threshold = float(body.get("threshold", 0.9))

    if not (0.0 < threshold <= 1.0):
        return jsonify({"error": "threshold phải nằm trong khoảng (0, 1]"}), 400

    try:
        result = _clustering_service.filter_by_centroid(threshold=threshold)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result), 200
