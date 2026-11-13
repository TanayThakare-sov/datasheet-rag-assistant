"""Turn retrieved passages into an answer with sources, or say "I don't know"."""
import os

from .config import ABSTAIN_THRESHOLD, LLM_MODEL, TOP_K

ABSTAIN_TEXT = "I don't know based on the provided datasheets."

SYSTEM = (
    "You answer questions about hard drive datasheets using ONLY the numbered passages provided. "
    "Cite sources as [document, page N] after each fact. For comparisons, give each drive's value. "
    f"If the passages do not contain the answer, reply exactly: {ABSTAIN_TEXT}"
)


def build_prompt(question, hits):
    ctx = "\n\n".join(
        f"[{i}] ({h.chunk.doc}, page {h.chunk.page})\n{h.chunk.text}" for i, h in enumerate(hits, 1)
    )
    return f"Passages:\n{ctx}\n\nQuestion: {question}"


def is_abstention(text):
    return "i don't know" in text.lower().replace("\u2019", "'")


def extractive_answer(hits, n=3):
    """No-LLM fallback: show the best passages with their sources."""
    lines = [f"- [{h.chunk.doc}, page {h.chunk.page}] {h.chunk.text[:300]}" for h in hits[:n]]
    return "Most relevant passages (no LLM key set):\n" + "\n".join(lines)


def _call_llm(question, hits):
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=500,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
    )
    return msg.content[0].text.strip()


def answer_question(question, retriever, k=TOP_K, threshold=ABSTAIN_THRESHOLD, use_llm=None):
    result = retriever.search(question, k)
    sources = [{"doc": h.chunk.doc, "page": h.chunk.page, "text": h.chunk.text} for h in result.hits]
    if result.top_lexical < threshold:
        return {"answer": ABSTAIN_TEXT, "abstained": True, "sources": [], "result": result}
    if use_llm is None:
        use_llm = bool(os.getenv("ANTHROPIC_API_KEY"))
    if use_llm:
        try:
            text = _call_llm(question, result.hits)
        except Exception as e:  # network error, bad key ...
            text = extractive_answer(result.hits) + f"\n(LLM unavailable: {type(e).__name__})"
    else:
        text = extractive_answer(result.hits)
    return {"answer": text, "abstained": is_abstention(text), "sources": sources, "result": result}
