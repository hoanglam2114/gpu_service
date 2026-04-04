"""
services/evaluation.py — Format prompt, BLEU/ROUGE metrics, Claude LLM judge.
"""
import contextlib
import json
import re
import time

import anthropic
import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from unsloth import FastLanguageModel

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import app.state as state
from app.config import config

# ── Hằng số ───────────────────────────────────────────────────────────────────
SOCRATIC_SYSTEM_PROMPT = (
    "Bạn là một trợ lý giáo dục chuyên nghiệp. Nhiệm vụ của bạn là hỗ trợ học sinh "
    "theo phương pháp Socratic: không đưa ra câu trả lời trực tiếp mà sử dụng "
    "các câu hỏi gợi mở để học sinh tự tìm ra đáp án trong mô hình Lớp học đảo ngược."
)

WEIGHT_QUALITY = 0.50
WEIGHT_HALLUCINATION = 0.35
WEIGHT_SPEED = 0.15

EVAL_STAGES = {
    "warmup":      {"pct": 5,  "label": "Khởi động GPU"},
    "infer_ft":    {"pct": 35, "label": "Inference FT model"},
    "infer_base":  {"pct": 60, "label": "Inference Base model"},
    "rubric":      {"pct": 85, "label": "Chấm điểm rubric"},
    "hallucinate": {"pct": 90, "label": "Đánh giá hallucination"},
    "finalize":    {"pct": 98, "label": "Tổng hợp kết quả"},
}

_HALLUCINATION_ADMIT_KEYWORDS = [
    "không biết", "không có thông tin", "không đủ thông tin", "không chắc",
    "không thể xác định", "không có trong", "ngoài phạm vi", "không rõ",
    "tôi không", "chưa có", "không tìm thấy",
]

RUBRIC_SYSTEM_PROMPT = """Bạn là giáo viên chấm bài. Chấm điểm theo 4 tiêu chí sau (mỗi tiêu chí /5):

CORRECTNESS:
- 5đ: Đáp án hoàn toàn đúng, công thức/phương trình đúng
- 3đ: Hướng đúng nhưng tính toán sai nhỏ hoặc thiếu đơn vị
- 1đ: Sai hướng giải hoặc áp dụng sai công thức
- 0đ: Không trả lời được hoặc hoàn toàn sai

EXPLANATION:
- 5đ: Trình bày đầy đủ các bước, logic rõ ràng, dễ theo dõi
- 3đ: Có trình bày nhưng bỏ qua một số bước trung gian
- 1đ: Chỉ đưa ra kết quả, không giải thích
- 0đ: Không có giải thích hoặc giải thích sai hoàn toàn

PEDAGOGY:
- 5đ: Ngôn ngữ phù hợp THPT, có thể dùng ví dụ/gợi ý giúp học sinh hiểu
- 3đ: Hiểu được nhưng còn dùng thuật ngữ khó hoặc quá ngắn gọn
- 1đ: Khó hiểu với học sinh THPT, quá hàn lâm hoặc thiếu context
- 0đ: Không phù hợp hoàn toàn

LANGUAGE_QUALITY:
- 5đ: Ngữ pháp tiếng Việt chuẩn, dấu câu đúng, văn phong tự nhiên, mạch lạc
- 3đ: Một vài lỗi nhỏ về dấu câu hoặc câu hơi cứng nhắc nhưng vẫn đọc được
- 1đ: Nhiều lỗi ngữ pháp/chính tả hoặc câu tối nghĩa
- 0đ: Không thể đọc được hoặc không phải tiếng Việt

Trả về JSON Array, mỗi phần tử ứng với 1 câu theo thứ tự:
[{
  "base_score": <trung bình 4 tiêu chí>,
  "ft_score": <trung bình 4 tiêu chí>,
  "criteria_detail": [
    {"name": "Correctness", "base_score": .., "ft_score": .., "base_reason": "<1 câu>", "ft_reason": "<1 câu>"},
    {"name": "Explanation", "base_score": .., "ft_score": .., "base_reason": "<1 câu>", "ft_reason": "<1 câu>"},
    {"name": "Pedagogy", "base_score": .., "ft_score": .., "base_reason": "<1 câu>", "ft_reason": "<1 câu>"},
    {"name": "Language_Quality", "base_score": .., "ft_score": .., "base_reason": "<1 câu>", "ft_reason": "<1 câu>"}
  ]
}]
CHỈ trả về JSON array, không có text khác."""

HALLUCINATION_SYSTEM_PROMPT = """Bạn là chuyên gia đánh giá hallucination của mô hình ngôn ngữ. Chấm điểm theo 3 tiêu chí (mỗi tiêu chí /5):

REFUSAL_ACCURACY — Mô hình có biết khi nào nên từ chối, khi nào nên trả lời?
- 5đ: expected="không biết/ngoài phạm vi" VÀ model từ chối rõ ràng HOẶC expected có nội dung thật VÀ model KHÔNG từ chối vô lý
- 3đ: Model từ chối nhưng lý do mơ hồ, hoặc trả lời nhưng thiếu tự tin không cần thiết
- 1đ: expected="không biết" nhưng model bịa câu trả lời HOẶC expected có nội dung nhưng model từ chối toàn bộ
- 0đ: Phản hồi hoàn toàn không liên quan đến câu hỏi

FACTUAL_FAITHFULNESS — Mô hình có bịa thêm thông tin không có trong đáp án chuẩn?
- 5đ: Không có thông tin bịa đặt; nếu trả lời thì sát với expected
- 3đ: Thêm thông tin nhỏ không gây hiểu sai nghiêm trọng
- 1đ: Bịa số liệu, tên, điều khoản cụ thể không có trong expected
- 0đ: Hoàn toàn bịa đặt, không liên quan đến expected

SPECIFICITY_ACCURACY — Số liệu, tên riêng, công thức cụ thể có đúng không?
- 5đ: Tất cả số liệu/tên/công thức cụ thể đều khớp với expected
- 3đ: Một vài chi tiết nhỏ sai nhưng không ảnh hưởng đến ý nghĩa tổng thể
- 1đ: Sai số liệu, tên, hoặc công thức quan trọng
- 0đ: Toàn bộ chi tiết cụ thể đều sai hoặc bịa đặt

Lưu ý: Nếu expected chứa "không biết", "ngoài phạm vi", "không có thông tin" → out-of-scope, model NÊN từ chối.

Trả về JSON Array:
[{
  "base_score": <trung bình 3 tiêu chí>,
  "ft_score": <trung bình 3 tiêu chí>,
  "criteria_detail": [
    {"name": "Refusal_Accuracy", "base_score": .., "ft_score": .., "base_reason": "<1 câu>", "ft_reason": "<1 câu>"},
    {"name": "Factual_Faithfulness", "base_score": .., "ft_score": .., "base_reason": "<1 câu>", "ft_reason": "<1 câu>"},
    {"name": "Specificity_Accuracy", "base_score": .., "ft_score": .., "base_reason": "<1 câu>", "ft_reason": "<1 câu>"}
  ]
}]
CHỈ trả về JSON array, không có text khác."""


# ── Format prompt ─────────────────────────────────────────────────────────────

def formatting_prompts_func(examples: dict) -> dict:
    """Hỗ trợ cả OpenAI Chat format và Alpaca format."""
    texts = []
    has_messages = "messages" in examples
    batch_size = len(examples["messages"]) if has_messages else len(examples.get("instruction", []))

    for i in range(batch_size):
        instruction = ""
        output = ""

        if has_messages:
            for m in examples["messages"][i]:
                if m["role"] == "user":
                    instruction = m["content"]
                elif m["role"] == "assistant":
                    output = m["content"]
        else:
            instruction = examples.get("instruction", [""])[i]
            output = examples.get("output", [""])[i]

        text = (
            f"### System:\n{SOCRATIC_SYSTEM_PROMPT}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output} <|endoftext|>"
        )
        texts.append(text)

    return {"text": texts}


# ── N-gram metrics ────────────────────────────────────────────────────────────

def compute_ngram_metrics(expected: str, answer: str) -> dict:
    if not expected or not answer:
        return {"bleu": 0.0, "rouge_l": 0.0}
    try:
        ref_tokens = nltk.word_tokenize(expected.lower())
        hyp_tokens = nltk.word_tokenize(answer.lower())
        bleu = round(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=SmoothingFunction().method1), 4)
    except Exception:
        bleu = 0.0
    try:
        rouge_l = round(rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False).score(expected, answer)["rougeL"].fmeasure, 4)
    except Exception:
        rouge_l = 0.0
    return {"bleu": bleu, "rouge_l": rouge_l}


# ── Adapter disable context manager ──────────────────────────────────────────

@contextlib.contextmanager
def _adapter_disabled(model):
    disabled = False
    try:
        if hasattr(model, "disable_adapter") and callable(model.disable_adapter):
            try:
                cm = model.disable_adapter()
                if hasattr(cm, "__enter__"):
                    cm.__enter__()
                    disabled = ("cm", cm)
                    yield
                    return
            except TypeError:
                pass
        if hasattr(model, "disable_adapters"):
            model.disable_adapters()
            disabled = "disable_adapters"
        elif hasattr(model, "base_model") and hasattr(model.base_model, "disable_adapter_layers"):
            model.base_model.disable_adapter_layers()
            disabled = "disable_adapter_layers"
        else:
            _scales = {}
            for name, module in model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "scaling"):
                    _scales[name] = module.scaling.copy() if isinstance(module.scaling, dict) else float(module.scaling)
                    if isinstance(module.scaling, dict):
                        for k in module.scaling:
                            module.scaling[k] = 0.0
                    else:
                        module.scaling = 0.0
            disabled = ("manual_scale", _scales)
        yield
    finally:
        if disabled == "disable_adapters":
            if hasattr(model, "enable_adapters"):
                model.enable_adapters()
        elif disabled == "disable_adapter_layers":
            model.base_model.enable_adapter_layers()
        elif isinstance(disabled, tuple) and disabled[0] == "cm":
            disabled[1].__exit__(None, None, None)
        elif isinstance(disabled, tuple) and disabled[0] == "manual_scale":
            for name, module in model.named_modules():
                if name in disabled[1] and hasattr(module, "scaling"):
                    module.scaling = disabled[1][name]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    return config.ANTHROPIC_API_KEY


def _eval_log(job_id: str, msg: str) -> None:
    print(msg)
    if job_id in state.jobs_db:
        state.jobs_db[job_id].setdefault("logs", []).append(msg)


def _eval_progress(eval_job_id: str, stage: str, detail: str = "", current: int = 0, total: int = 0, sample: dict = None) -> None:
    if eval_job_id not in state.eval_jobs_db:
        return
    stage_info = EVAL_STAGES.get(stage, {"pct": 0, "label": stage})
    pct = stage_info["pct"]
    if total > 0 and stage in ("infer_ft", "infer_base", "rubric"):
        stages = list(EVAL_STAGES.keys())
        idx = stages.index(stage)
        start_pct = EVAL_STAGES[stages[idx - 1]]["pct"] if idx > 0 else 0
        pct = int(start_pct + (stage_info["pct"] - start_pct) * current / total)

    update = {
        "progress": pct,
        "stage": stage,
        "stage_label": stage_info["label"],
        "stage_detail": detail,
        "stage_current": current,
        "stage_total": total,
    }
    if sample is not None:
        update["current_sample"] = sample
    state.eval_jobs_db[eval_job_id].update(update)


def _is_out_of_scope(expected_str: str) -> bool:
    return any(kw in expected_str.lower().strip() for kw in _HALLUCINATION_ADMIT_KEYWORDS)


# ── Rubric eval ───────────────────────────────────────────────────────────────

def claude_rubric_eval(samples, base_answers, ft_answers, subject="unknown", judge_model="claude-sonnet-4-5-20251001"):
    results = []
    to_api = []
    vectorizer = TfidfVectorizer()

    for i, (sample, b_ans, f_ans) in enumerate(zip(samples, base_answers, ft_answers)):
        try:
            tfidf = vectorizer.fit_transform([b_ans, f_ans])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        except Exception:
            sim = 0.0
        if sim > 0.92:
            results.append({
                "subject": subject, "instruction": sample.get("instruction", ""),
                "expected": str(sample.get("output", sample.get("hints", ""))),
                "base_answer": b_ans, "ft_answer": f_ans,
                "base_score": 3.0, "ft_score": 3.0, "delta": 0,
                "eval_method": "skipped_similar", "criteria_detail": [],
            })
        else:
            to_api.append((i, sample, b_ans, f_ans))

    api_key = _get_api_key()
    if not api_key:
        for (i, sample, b_ans, f_ans) in to_api:
            results.append({
                "subject": subject, "instruction": sample.get("instruction", ""),
                "expected": str(sample.get("output", "")),
                "base_answer": b_ans, "ft_answer": f_ans,
                "base_score": 0.0, "ft_score": 0.0, "delta": 0,
                "eval_method": "api_failed_missing_key", "criteria_detail": [],
            })
        return results
    if not to_api:
        return results

    try:
        client = anthropic.Anthropic(api_key=api_key)
        user_parts = []
        for (i, sample, b_ans, f_ans) in to_api:
            expected = str(sample.get("output", sample.get("hints", "")))
            user_parts.append(
                f"\n--- CÂU {i+1} (môn: {subject}) ---"
                f"\nCâu hỏi: {sample.get('instruction', '')}"
                f"\nĐáp án chuẩn: {expected}"
                f"\nBASE ANS: {b_ans}"
                f"\nFT ANS: {f_ans}"
            )
        msg = client.messages.create(
            model=judge_model,
            max_tokens=4000, temperature=0,
            system=RUBRIC_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "\n".join(user_parts) + "\n\nONLY output valid JSON array."}],
        )
        reply = msg.content[0].text
        json_str = reply[reply.find("["):reply.rfind("]") + 1]
        parsed = json.loads(json_str)
        for idx, (original_i, sample, b_ans, f_ans) in enumerate(to_api):
            if idx < len(parsed):
                p = parsed[idx]
                b_score = round(float(p.get("base_score", 0)), 2)
                f_score = round(float(p.get("ft_score", 0)), 2)
                results.append({
                    "subject": subject, "instruction": sample.get("instruction", ""),
                    "expected": str(sample.get("output", sample.get("hints", ""))),
                    "base_answer": b_ans, "ft_answer": f_ans,
                    "base_score": b_score, "ft_score": f_score,
                    "delta": round(f_score - b_score, 2),
                    "eval_method": "claude_rubric",
                    "criteria_detail": p.get("criteria_detail", []),
                })
    except Exception as e:
        print(f"[!] Claude API error ({subject}): {e}")
        for (i, sample, b_ans, f_ans) in to_api:
            results.append({
                "subject": subject, "instruction": sample.get("instruction", ""),
                "expected": str(sample.get("output", "")),
                "base_answer": b_ans, "ft_answer": f_ans,
                "base_score": 0.0, "ft_score": 0.0, "delta": 0,
                "eval_method": "api_error", "criteria_detail": [],
            })
    return results


# ── Hallucination eval ────────────────────────────────────────────────────────

def _keyword_score_hallucination_fallback(answer, expected):
    if not isinstance(answer, str) or not answer.strip():
        return 0.0
    ans_lower = answer.lower()
    ans_admits = any(kw in ans_lower for kw in _HALLUCINATION_ADMIT_KEYWORDS)
    if _is_out_of_scope(str(expected)):
        return 5.0 if ans_admits else 1.0
    else:
        if ans_admits:
            return 2.0
        has_numbers = bool(re.search(r"\d", answer))
        has_specific = len(answer.split()) > 15
        if has_numbers and has_specific:
            return 4.0
        if has_specific:
            return 3.5
        return 3.0


def eval_hallucination_samples(h_samples, base_answers, ft_answers, judge_model="claude-sonnet-4-5-20251001"):
    if not h_samples:
        return 0.0, 0.0, []

    api_key = _get_api_key()

    if api_key:
        results = []
        try:
            client = anthropic.Anthropic(api_key=api_key)
            user_parts = []
            for i, s in enumerate(h_samples):
                expected = str(s.get("output", "Không biết"))
                scope_hint = "OUT-OF-SCOPE (model nên từ chối)" if _is_out_of_scope(expected) else "IN-SCOPE (model nên trả lời đúng)"
                user_parts.append(
                    f"\n--- CÂU {i+1} [{scope_hint}] ---"
                    f"\nCâu hỏi: {s.get('instruction', '')}"
                    f"\nĐáp án chuẩn: {expected}"
                    f"\nBASE ANS: {base_answers[i]}"
                    f"\nFT ANS: {ft_answers[i]}"
                )
            msg = client.messages.create(
                model=judge_model, max_tokens=3000, temperature=0,
                system=HALLUCINATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": "\n".join(user_parts) + "\n\nONLY output valid JSON array."}],
            )
            reply = msg.content[0].text
            json_str = reply[reply.find("["):reply.rfind("]") + 1]
            parsed = json.loads(json_str)
            for i, s in enumerate(h_samples):
                expected = str(s.get("output", "Không biết"))
                if i < len(parsed):
                    p = parsed[i]
                    b_score = round(float(p.get("base_score", 0)), 2)
                    f_score = round(float(p.get("ft_score", 0)), 2)
                    criteria = p.get("criteria_detail", [])
                    eval_method = "claude_hallucination"
                else:
                    b_score = f_score = 0.0
                    criteria = []
                    eval_method = "claude_hallucination_parse_error"
                results.append({
                    "subject": "hallucination", "instruction": s.get("instruction", ""),
                    "expected": expected, "base_answer": base_answers[i], "ft_answer": ft_answers[i],
                    "base_score": b_score, "ft_score": f_score,
                    "delta": round(f_score - b_score, 2),
                    "eval_method": eval_method, "criteria_detail": criteria,
                    "hallucination_type": "out_of_scope" if _is_out_of_scope(expected) else "in_scope",
                })
        except Exception as e:
            print(f"[!] Claude API error (hallucination): {e} — fallback to keyword")
            results = _hallucination_keyword_fallback(h_samples, base_answers, ft_answers, "keyword_fallback_api_error")
    else:
        results = _hallucination_keyword_fallback(h_samples, base_answers, ft_answers, "keyword_fallback_no_key")

    base_avg = round(sum(r["base_score"] for r in results) / len(results), 2)
    ft_avg = round(sum(r["ft_score"] for r in results) / len(results), 2)
    return base_avg, ft_avg, results


def _hallucination_keyword_fallback(h_samples, base_answers, ft_answers, eval_method):
    results = []
    for i, s in enumerate(h_samples):
        expected = str(s.get("output", "Không biết"))
        b_score = _keyword_score_hallucination_fallback(base_answers[i], expected)
        f_score = _keyword_score_hallucination_fallback(ft_answers[i], expected)
        results.append({
            "subject": "hallucination", "instruction": s.get("instruction", ""),
            "expected": expected, "base_answer": base_answers[i], "ft_answer": ft_answers[i],
            "base_score": b_score, "ft_score": f_score, "delta": round(f_score - b_score, 2),
            "eval_method": eval_method, "criteria_detail": [],
            "hallucination_type": "out_of_scope" if _is_out_of_scope(expected) else "in_scope",
        })
    return results


# ── Speed scoring ─────────────────────────────────────────────────────────────

def score_speed(base_avg_ms: float, ft_avg_ms: float) -> tuple[float, float]:
    if base_avg_ms <= 0:
        return 3.0, 3.0
    ratio = ft_avg_ms / base_avg_ms
    if ratio <= 0.80:   ft_score = 5.0
    elif ratio <= 0.90: ft_score = 4.0
    elif ratio <= 1.10: ft_score = 3.0
    elif ratio <= 1.20: ft_score = 2.0
    else:               ft_score = 1.0
    return 3.0, ft_score


def compute_overall_score(q: float, h: float, s: float) -> float:
    return round(q * WEIGHT_QUALITY + h * WEIGHT_HALLUCINATION + s * WEIGHT_SPEED, 2)


# ── Main evaluation runner ────────────────────────────────────────────────────

def run_auto_evaluation(job_id, eval_job_id, eval_file_path, active_model, tokenizer, max_seq, judge_model="claude-haiku-4-5-20251001"):
    from collections import defaultdict

    samples = []
    with open(eval_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                samples.append(json.loads(line))
            except Exception:
                pass
    if not samples:
        _eval_log(job_id, "[Eval] Không có sample hợp lệ, bỏ qua.")
        return

    subject_samples = [s for s in samples if s.get("subject") in ["toan", "ly", "hoa", "sinh", "van"]]
    hallucination_samples = [s for s in samples if s.get("subject") == "hallucination"]
    all_inference_samples = subject_samples + hallucination_samples
    total = len(all_inference_samples)
    _eval_log(job_id, f"[📊] Bắt đầu đánh giá: {len(subject_samples)} mẫu môn học | {len(hallucination_samples)} mẫu hallucination")

    active_model.eval()
    FastLanguageModel.for_inference(active_model)
    infer_prompt = "### Instruction:\n{}\n\n### Response:"

    # Warmup
    _eval_log(job_id, "[⚙️] Warming up GPU...")
    try:
        _d = tokenizer(["Xin chào"], return_tensors="pt").to("cuda")
        active_model.generate(**_d, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
        with _adapter_disabled(active_model):
            active_model.generate(**_d, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    except Exception:
        pass
    _eval_progress(eval_job_id, "warmup", "GPU ready")

    # Inference FT
    ft_answers, ft_times = [], []
    _eval_log(job_id, f"[⚙️] Inference FT model ({total} mẫu)...")
    for idx, s in enumerate(all_inference_samples):
        try:
            inputs = tokenizer([infer_prompt.format(s.get("instruction", ""))], return_tensors="pt").to("cuda")
            t0 = time.time()
            outputs = active_model.generate(**inputs, max_new_tokens=256, use_cache=True, pad_token_id=tokenizer.eos_token_id)
            ft_times.append((time.time() - t0) * 1000)
            resp = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("### Response:")[-1].strip()
        except Exception as e:
            resp = f"[ERROR: {e}]"
            ft_times.append(0.0)
        ft_answers.append(resp)
        _eval_progress(eval_job_id, "infer_ft", s.get("subject", ""), current=idx + 1, total=total,
                       sample={"index": idx, "instruction": s.get("instruction", ""), "ft_answer": resp, "base_answer": None})
        _eval_log(job_id, f"  [FT {idx+1}/{total}] ...")

    # Inference Base
    base_answers, base_times = [], []
    _eval_log(job_id, f"[⚙️] Inference Base model ({total} mẫu)...")
    with _adapter_disabled(active_model):
        for idx, s in enumerate(all_inference_samples):
            try:
                inputs = tokenizer([infer_prompt.format(s.get("instruction", ""))], return_tensors="pt").to("cuda")
                t0 = time.time()
                outputs = active_model.generate(**inputs, max_new_tokens=256, use_cache=True, pad_token_id=tokenizer.eos_token_id)
                base_times.append((time.time() - t0) * 1000)
                resp = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("### Response:")[-1].strip()
            except Exception as e:
                resp = f"[ERROR: {e}]"
                base_times.append(0.0)
            base_answers.append(resp)
            _eval_progress(eval_job_id, "infer_base", s.get("subject", ""), current=idx + 1, total=total,
                           sample={"index": idx, "instruction": s.get("instruction", ""), "ft_answer": ft_answers[idx], "base_answer": resp})
            _eval_log(job_id, f"  [Base {idx+1}/{total}] ...")

    n_subj = len(subject_samples)
    subj_base_ans = base_answers[:n_subj]
    subj_ft_ans = ft_answers[:n_subj]
    hall_base_ans = base_answers[n_subj:]
    hall_ft_ans = ft_answers[n_subj:]

    # N-gram metrics
    ngram_results = []
    for i, s in enumerate(all_inference_samples):
        expected = str(s.get("output", s.get("hints", "")))
        ngram_results.append({
            "base": compute_ngram_metrics(expected, base_answers[i]),
            "ft": compute_ngram_metrics(expected, ft_answers[i]),
        })
    n = len(ngram_results)
    ngram_summary = {
        "bleu":    {"base": round(sum(r["base"]["bleu"]    for r in ngram_results) / max(1, n), 4),
                    "ft":   round(sum(r["ft"]["bleu"]      for r in ngram_results) / max(1, n), 4)},
        "rouge_l": {"base": round(sum(r["base"]["rouge_l"] for r in ngram_results) / max(1, n), 4),
                    "ft":   round(sum(r["ft"]["rouge_l"]   for r in ngram_results) / max(1, n), 4)},
    }
    _eval_log(job_id, f"[📏] BLEU: Base={ngram_summary['bleu']['base']:.3f} | FT={ngram_summary['bleu']['ft']:.3f}")
    _eval_log(job_id, f"[📏] ROUGE-L: Base={ngram_summary['rouge_l']['base']:.3f} | FT={ngram_summary['rouge_l']['ft']:.3f}")

    # Rubric per subject
    _eval_log(job_id, "[📐] Chấm điểm theo rubric (Claude API)...")
    subj_groups = defaultdict(lambda: {"samples": [], "base": [], "ft": []})
    for i, s in enumerate(subject_samples):
        subj_groups[s["subject"]]["samples"].append(s)
        subj_groups[s["subject"]]["base"].append(subj_base_ans[i])
        subj_groups[s["subject"]]["ft"].append(subj_ft_ans[i])

    quality_results = []
    for subj, group in subj_groups.items():
        _eval_log(job_id, f"  [📝] Chấm môn {subj.upper()} ({len(group['samples'])} câu)...")
        _eval_progress(eval_job_id, "rubric", "Gọi Claude judge...")
        quality_results.extend(claude_rubric_eval(group["samples"], group["base"], group["ft"], subject=subj, judge_model=judge_model))

    # Hallucination
    hall_base_score, hall_ft_score, hall_results = eval_hallucination_samples(hallucination_samples, hall_base_ans, hall_ft_ans, judge_model=judge_model)
    if hallucination_samples:
        _eval_progress(eval_job_id, "hallucinate", f"{len(hallucination_samples)} mẫu")
        _eval_log(job_id, f"[🧠] Hallucination: Base={hall_base_score:.2f} | FT={hall_ft_score:.2f}")
    else:
        _eval_log(job_id, "[🧠] Không có câu hallucination")
        hall_base_score = hall_ft_score = sum(r["base_score"] for r in quality_results) / max(1, len(quality_results))

    # Speed
    base_avg_ms = sum(base_times) / len(base_times) if base_times else 0
    ft_avg_ms = sum(ft_times) / len(ft_times) if ft_times else 0
    speed_base_score, speed_ft_score = score_speed(base_avg_ms, ft_avg_ms)
    _eval_log(job_id, f"[⚡] Tốc độ: Base={base_avg_ms:.0f}ms | FT={ft_avg_ms:.0f}ms → score FT={speed_ft_score:.1f}/5")

    # Overall
    _eval_progress(eval_job_id, "finalize", "")
    quality_base = sum(x["base_score"] for x in quality_results) / len(quality_results) if quality_results else 0.0
    quality_ft = sum(x["ft_score"] for x in quality_results) / len(quality_results) if quality_results else 0.0
    overall_base = compute_overall_score(quality_base, hall_base_score, speed_base_score)
    overall_ft = compute_overall_score(quality_ft, hall_ft_score, speed_ft_score)
    overall_imp = round((overall_ft - overall_base) / max(0.01, overall_base) * 100, 1)
    _eval_log(job_id, f"[✅] ĐIỂM TỔNG: Base={overall_base:.2f} | FT={overall_ft:.2f} | Δ={overall_imp:+.1f}%")

    all_results = quality_results + hall_results
    subj_list = defaultdict(list)
    for r in all_results:
        subj_list[r["subject"]].append(r)
    by_subject = {
        k: {
            "base_avg": round(sum(x["base_score"] for x in v) / len(v), 2),
            "ft_avg": round(sum(x["ft_score"] for x in v) / len(v), 2),
            "improvement_pct": round(
                (sum(x["ft_score"] for x in v) / len(v) - sum(x["base_score"] for x in v) / len(v))
                / max(0.01, sum(x["base_score"] for x in v) / len(v)) * 100, 1
            ),
        }
        for k, v in subj_list.items()
    }

    eval_result = {
        "modelEvalId": eval_job_id, "jobId": job_id, "status": "COMPLETED", "totalSamples": total,
        "subjectBreakdown": {k: len(v) for k, v in subj_list.items()},
        "skippedBySimilarity": sum(1 for r in quality_results if r.get("eval_method") == "skipped_similar"),
        "results": all_results,
        "summary": {
            "overall": {"base_avg": overall_base, "ft_avg": overall_ft, "delta": round(overall_ft - overall_base, 2), "improvement_pct": overall_imp},
            "quality": {"base_avg": round(quality_base, 2), "ft_avg": round(quality_ft, 2), "weight": WEIGHT_QUALITY},
            "hallucination": {"base_avg": hall_base_score, "ft_avg": hall_ft_score, "weight": WEIGHT_HALLUCINATION, "sample_count": len(hallucination_samples)},
            "speed": {"base_avg_ms": round(base_avg_ms, 1), "ft_avg_ms": round(ft_avg_ms, 1), "base_score": speed_base_score, "ft_score": speed_ft_score, "weight": WEIGHT_SPEED},
            "by_subject": by_subject, "max_possible": 5, "reference_metrics": ngram_summary,
        },
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "completedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    state.jobs_db[job_id]["eval_result"] = eval_result
    _eval_log(job_id, "[💾] Kết quả eval đã lưu vào jobs_db.")


# ── Background eval task ──────────────────────────────────────────────────────

def background_eval_task(eval_job_id, job_id, hf_repo_id, hf_token, eval_file_path, model_max_length, judge_model="claude-sonnet-4-5-20251001"):
    from unsloth import FastLanguageModel
    from app.services.gpu import release_gpu_memory

    state.eval_jobs_db[eval_job_id]["status"] = "RUNNING"

    def _log(msg):
        print(msg)
        state.eval_jobs_db.get(eval_job_id, {}).setdefault("logs", []).append(msg)

    model = None
    tokenizer = None
    try:
        release_gpu_memory()
        _log(f"[🔄] Loading model từ HF Hub: {hf_repo_id}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=hf_repo_id, max_seq_length=model_max_length, load_in_4bit=True, token=hf_token or None,
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        _log("[✅] Model loaded.")

        state.eval_jobs_db[eval_job_id]["status"] = "EVALUATING"
        _run_evaluation_for_eval_job(eval_job_id, job_id, eval_file_path, model, tokenizer, model_max_length, _log, judge_model)

        _log("[✅] Hoàn thành Eval.")
        state.eval_jobs_db[eval_job_id]["status"] = "COMPLETED"

    except Exception as e:
        _log(f"[❌] LỖI EVAL: {e}")
        state.eval_jobs_db[eval_job_id].update({"status": "FAILED", "error": str(e)})

    finally:
        del model
        del tokenizer
        release_gpu_memory()
        if eval_file_path and __import__("os").path.exists(eval_file_path):
            __import__("os").remove(eval_file_path)
        state.eval_slot_release()


def _run_evaluation_for_eval_job(eval_job_id, job_id, eval_file_path, model, tokenizer, model_max_length, log_fn, judge_model):
    _tmp_key = f"__eval_tmp_{eval_job_id}"
    state.jobs_db[_tmp_key] = {"status": "EVALUATING", "logs": []}
    try:
        run_auto_evaluation(
            job_id=_tmp_key, eval_job_id=eval_job_id, eval_file_path=eval_file_path,
            active_model=model, tokenizer=tokenizer, max_seq=model_max_length, judge_model=judge_model,
        )
        raw_result = state.jobs_db[_tmp_key].get("eval_result")
        if raw_result:
            raw_result["jobId"] = job_id
            raw_result["modelEvalId"] = eval_job_id
            state.eval_jobs_db.setdefault(eval_job_id, {})["result"] = raw_result
        else:
            raise RuntimeError("run_auto_evaluation hoàn thành nhưng không có eval_result")
    finally:
        state.jobs_db.pop(_tmp_key, None)
