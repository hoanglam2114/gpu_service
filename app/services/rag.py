"""
services/rag.py — DuckDuckGo web search + RAG augment cho system prompt.
"""
import datetime

from duckduckgo_search import DDGS

_TEMPORAL_KEYWORDS = [
    "hôm nay", "hiện tại", "hiện nay", "mới nhất", "gần đây",
    "tuần này", "tháng này", "năm nay", "vừa", "ngày mai",
    "giá", "tỷ giá", "thời tiết", "tin tức", "kết quả", "lịch",
    "bao nhiêu tiền", "học phí", "tuyển sinh", "điểm chuẩn",
    "nghị định", "thông tư", "quy định mới", "lãi suất",
    "today", "current", "latest", "now",
]

_VN_HINTS = ["xăng", "điện", "học phí", "tuyển sinh", "điểm chuẩn", "lương", "thuế", "nghị định", "thông tư", "lãi suất"]
_REMOVE_PHRASES = ["thế nào", "ra sao", "như thế nào", "bao nhiêu", "là gì", "ở đâu", "khi nào", "?", "nhỉ", "nhé", "ạ"]
_TIME_REPLACE = ["hôm nay", "hiện tại", "hiện nay", "ngày hôm nay", "bây giờ", "lúc này", "thời điểm này"]


def needs_web_search(text: str) -> bool:
    return any(kw in text.lower() for kw in _TEMPORAL_KEYWORDS)


def rewrite_query(user_text: str) -> str:
    """Chuyển câu hỏi tự nhiên → search query ngắn gọn có năm."""
    year = datetime.date.today().year
    text = user_text.lower().strip()
    for phrase in _TIME_REPLACE:
        text = text.replace(phrase, str(year))
    for phrase in _REMOVE_PHRASES:
        text = text.replace(phrase, "")
    if any(h in text for h in _VN_HINTS) and "việt nam" not in text:
        text = text + " Việt Nam"
    return text.strip()


def web_search(query: str, max_results: int = 3) -> str:
    """Tìm DuckDuckGo, trả về context string. Trả về '' nếu lỗi."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results, region="vn-vi"))
        if not results:
            return ""
        return "\n".join(f"- {r.get('title', '')}: {r.get('body', '')}" for r in results)
    except Exception as e:
        print(f"[Web Search Error] {e}")
        return ""


def build_rag_system_prompt(base_system_prompt: str, query: str) -> tuple[str, bool]:
    """
    Augment system prompt với web search context nếu cần.
    Trả về (system_prompt_mới, đã_search).
    """
    if not needs_web_search(query):
        return base_system_prompt, False

    search_query = rewrite_query(query)
    today_str = datetime.date.today().strftime("%d/%m/%Y")

    print(f"[🔍 RAG] Query gốc: {query}")
    print(f"[🔍 RAG] Search query: {search_query}")

    context = web_search(search_query)
    if not context:
        print("[🔍 RAG] Không tìm được kết quả.")
        return base_system_prompt, False

    print(f"[🔍 RAG] OK — {len(context)} ký tự context")

    augmented = (
        f"{base_system_prompt}\n\n"
        f"[Thông tin tra cứu ngày {today_str}]\n"
        f"{context}\n"
        f"Dùng thông tin trên để trả lời nếu liên quan. Nếu không chắc, nói rõ với học sinh."
    )
    return augmented, True
