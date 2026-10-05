from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL


def _get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def _clean_llm_response(text: str) -> str:
    """Clean thinking tokens and reasoning prefixes from models like Nemotron/Deepseek."""
    if not text:
        return ""
    import re
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    if "Here's a thinking process:" in cleaned:
        parts = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        cleaned = parts[-1] if parts else cleaned
    return cleaned.strip()


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if not text.strip():
        return ""
    if OPENAI_API_KEY:
        try:
            client = _get_openai_client()
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. KHÔNG viết suy nghĩ, chỉ trả lời bản tóm tắt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            cleaned = _clean_llm_response(resp.choices[0].message.content)
            if cleaned and len(cleaned) <= len(text) * 2:
                return cleaned
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")

    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if not text.strip():
        return []
    if OPENAI_API_KEY:
        try:
            client = _get_openai_client()
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng có dấu chấm hỏi (?). KHÔNG viết suy nghĩ."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            cleaned = _clean_llm_response(resp.choices[0].message.content)
            questions = cleaned.split("\n")
            res = [q.strip().lstrip("0123456789.-) ") for q in questions if "?" in q][:n_questions]
            if res:
                return res
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")

    import re
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    questions = [f"{s.rstrip('.')} như thế nào?" for s in sentences[:n_questions]]
    return questions if questions else [f"Thông tin về {text[:30]} là gì?"]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if not text.strip():
        return ""
    if OPENAI_API_KEY:
        try:
            client = _get_openai_client()
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
            )
            context = resp.choices[0].message.content.strip()
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")

    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if not text.strip():
        return {}
    if OPENAI_API_KEY:
        try:
            import json as _json
            client = _get_openai_client()
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.strip("`").lstrip("json").strip()
            return _json.loads(content)
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")

    lower_text = text.lower()
    category = "policy"
    if any(w in lower_text for w in ["mật khẩu", "vpn", "cntt", "malware", "it"]):
        category = "it"
    elif any(w in lower_text for w in ["nghỉ phép", "thử việc", "lương", "bảo hiểm", "phụ cấp"]):
        category = "hr"
    elif any(w in lower_text for w in ["chi phí", "tạm ứng", "hóa đơn", "ngân sách"]):
        category = "finance"

    return {
        "topic": text.split("\n")[0][:40],
        "entities": [],
        "category": category,
        "language": "vi"
    }


_circuit_breaker_triggered = False


def _offline_fallback_enrich(text: str, source: str) -> dict:
    """Pure offline fallback rule for enrichment."""
    import re
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    summary = ". ".join(sentences[:2]) + "." if sentences else text
    sentences_for_q = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    questions = [f"{s.rstrip('.')} như thế nào?" for s in sentences_for_q[:3]] or [f"Thông tin về {text[:30]} là gì?"]
    context_prefix = f"Trích từ tài liệu {source}." if source else ""

    lower_text = text.lower()
    category = "policy"
    if any(w in lower_text for w in ["mật khẩu", "vpn", "cntt", "malware", "it"]):
        category = "it"
    elif any(w in lower_text for w in ["nghỉ phép", "thử việc", "lương", "bảo hiểm", "phụ cấp"]):
        category = "hr"
    elif any(w in lower_text for w in ["chi phí", "tạm ứng", "hóa đơn", "ngân sách"]):
        category = "finance"

    return {
        "summary": summary,
        "questions": questions,
        "context": context_prefix,
        "metadata": {
            "topic": text.split("\n")[0][:40],
            "entities": [],
            "category": category,
            "language": "vi"
        }
    }


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    global _circuit_breaker_triggered
    if not text.strip():
        return {}
    if OPENAI_API_KEY and not _circuit_breaker_triggered:
        try:
            import json as _json
            client = _get_openai_client()
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": """Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=400,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.strip("`").lstrip("json").strip()
            return _json.loads(content)
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}. Switching to offline fallback.")
            _circuit_breaker_triggered = True

    return _offline_fallback_enrich(text, source)


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
