"""
routers/infer.py — POST /api/model/load, POST /api/infer/stream, GET /api/infer/logs
"""
import json
import traceback
import uuid
import datetime

from flask import Blueprint, request, jsonify, Response
from threading import Thread

import app.state as state
from app.services.gpu import release_gpu_memory
from app.services.inference import DEFAULT_SYSTEM_PROMPT, format_inference_prompt, stream_without_thinking
from app.services.rag import build_rag_system_prompt

infer_bp = Blueprint("infer", __name__)


@infer_bp.route("/api/model/load", methods=["POST"])
def load_model():
    from unsloth import FastLanguageModel
    from transformers import TextIteratorStreamer  # noqa: ensure import

    data = request.json or {}
    hf_model_id = data.get("hf_model_id")
    if not hf_model_id:
        return jsonify({"error": "Missing hf_model_id"}), 400

    try:
        if (
            state._current_infer_model is None
            or state._current_infer_tokenizer is None
            or state._current_infer_model_id != hf_model_id
        ):
            release_gpu_memory()
            print(f"Loading model {hf_model_id} into GPU...")
            state._current_infer_model, state._current_infer_tokenizer = FastLanguageModel.from_pretrained(
                model_name=hf_model_id, max_seq_length=2048, load_in_4bit=True,
            )
            if getattr(state._current_infer_tokenizer, "pad_token", None) is None:
                state._current_infer_tokenizer.pad_token = state._current_infer_tokenizer.eos_token
            state._current_infer_tokenizer.padding_side = "right"
            state._current_infer_model_id = hf_model_id
            return jsonify({"status": "success", "message": f"Model {hf_model_id} loaded successfully."}), 200
        else:
            return jsonify({"status": "success", "message": f"Model {hf_model_id} is already loaded."}), 200

    except Exception as e:
        release_gpu_memory()
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@infer_bp.route("/api/infer/stream", methods=["POST"])
def infer_model_stream():
    from transformers import TextIteratorStreamer

    if state._current_infer_model is None or state._current_infer_tokenizer is None:
        return jsonify({"error": "No model loaded. Please call /api/model/load first."}), 400

    data = request.json or {}
    text_input = data.get("text_input")
    system_prompt = data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    hf_model_id = data.get("hf_model_id")

    if hf_model_id and hf_model_id != state._current_infer_model_id:
        return jsonify({"error": f"Model mismatch. Requested {hf_model_id} but {state._current_infer_model_id} is loaded."}), 400
    if not text_input:
        return jsonify({"error": "Missing text_input"}), 400

    # RAG augmentation
    system_prompt, _searched = build_rag_system_prompt(system_prompt, text_input)
    if _searched:
        print("[RAG] Đã augment system prompt với web search context.")

    gen_max_new_tokens     = data.get("max_new_tokens", 512)
    gen_temperature        = data.get("temperature", 0.3)
    gen_top_k              = data.get("top_k", 40)
    gen_top_p              = data.get("top_p", 0.85)
    gen_repetition_penalty = data.get("repetition_penalty", 1.15)
    gen_do_sample          = data.get("do_sample", True)

    inference_id = str(uuid.uuid4())
    state.inference_logs_db[inference_id] = {
        "inference_id": inference_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "model_id": state._current_infer_model_id,
        "input_parameters": {
            "text_input": text_input, "system_prompt": system_prompt,
            "max_new_tokens": gen_max_new_tokens, "temperature": gen_temperature,
            "top_k": gen_top_k, "top_p": gen_top_p,
            "repetition_penalty": gen_repetition_penalty, "do_sample": gen_do_sample,
        },
        "generated_text": "",
        "status": "started",
    }

    try:
        instruction_text = format_inference_prompt(state._current_infer_tokenizer, system_prompt, text_input)
        inputs = state._current_infer_tokenizer(text=instruction_text, return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(
            state._current_infer_tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        generation_kwargs = dict(
            **inputs, streamer=streamer,
            max_new_tokens=gen_max_new_tokens, do_sample=gen_do_sample,
            temperature=gen_temperature, top_k=gen_top_k, top_p=gen_top_p,
            repetition_penalty=gen_repetition_penalty,
            eos_token_id=state._current_infer_tokenizer.eos_token_id,
        )

        Thread(target=state._current_infer_model.generate, kwargs=generation_kwargs, daemon=True).start()

        def generate_stream():
            full_response = []
            for text in stream_without_thinking(streamer):
                full_response.append(text)
                yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
            state.inference_logs_db[inference_id]["generated_text"] = "".join(full_response)
            state.inference_logs_db[inference_id]["status"] = "completed"
            yield "data: [DONE]\n\n"

        return Response(generate_stream(), mimetype="text/event-stream")

    except Exception as e:
        traceback.print_exc()
        state.inference_logs_db[inference_id]["status"] = "error"
        state.inference_logs_db[inference_id]["error_message"] = str(e)
        return jsonify({"error": str(e)}), 500


@infer_bp.route("/api/infer/logs", methods=["GET"])
@infer_bp.route("/api/infer/logs/<inference_id>", methods=["GET"])
def get_inference_logs(inference_id=None):
    if inference_id:
        log = state.inference_logs_db.get(inference_id)
        if log:
            return jsonify(log), 200
        return jsonify({"error": "Inference log not found"}), 404
    return jsonify(list(state.inference_logs_db.values())), 200
