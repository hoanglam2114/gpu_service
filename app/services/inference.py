"""
services/inference.py — SSE streaming inference + think-tag filter.
"""


DEFAULT_SYSTEM_PROMPT = """Bạn là một gia sư thông minh, hỗ trợ học sinh THCS và THPT Việt Nam học tập theo phương pháp lớp học đảo ngược (Flipped Classroom).

VAI TRÒ CỦA BẠN:
- Không giảng lại lý thuyết từ đầu — học sinh đã tự học trước ở nhà.
- Khi học sinh hỏi, hãy ưu tiên đặt câu hỏi gợi mở để kiểm tra mức độ hiểu và kích thích tư duy trước khi giải thích.
- Hướng dẫn từng bước nhỏ, không đưa đáp án ngay — giúp học sinh tự tìm ra.
- Nếu học sinh thực sự bí hoặc đã thử nhiều lần, mới giải thích chi tiết hơn.
- Khen ngợi đúng lúc khi học sinh suy nghĩ đúng hướng.

CÁCH GIAO TIẾP:
- Tiếng Việt hoàn toàn.
- Thân thiện như bạn bè nhưng đáng tin cậy — không cợt nhả, không quá nghiêm túc.
- Câu ngắn gọn, rõ ý. Tránh giải thích dài dòng khi chưa cần thiết.
- Dùng ví dụ gần gũi với cuộc sống học sinh Việt Nam khi cần minh hoạ."""


def format_inference_prompt(tokenizer, system_prompt: str, user_input: str) -> str:
    """Format prompt theo chat template của từng model. Tắt <think> với Qwen3."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def stream_without_thinking(streamer):
    """
    Generator lọc bỏ <think>...</think> khi stream token.
    - Chưa gặp <think>  → yield bình thường
    - Đang trong <think> → bỏ, giữ 9 ký tự cuối phòng tag bị cắt
    - Sau </think>       → yield tất cả như bình thường
    """
    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    buffer = ""
    in_think = False
    thinking_done = False

    for chunk in streamer:
        buffer += chunk

        if not in_think and not thinking_done:
            if OPEN_TAG in buffer:
                in_think = True
                buffer = buffer.split(OPEN_TAG, 1)[1]
            else:
                if len(buffer) > 10:
                    yield buffer[:-10]
                    buffer = buffer[-10:]
                continue

        if in_think:
            if CLOSE_TAG in buffer:
                in_think = False
                thinking_done = True
                buffer = buffer.split(CLOSE_TAG, 1)[1]
            else:
                buffer = buffer[-9:] if len(buffer) > 9 else buffer
                continue

        if thinking_done and buffer:
            yield buffer
            buffer = ""

    if buffer and not in_think:
        yield buffer
