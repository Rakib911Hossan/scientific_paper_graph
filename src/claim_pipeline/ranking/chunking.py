def chunk_sentences(sentences: list[dict], chunk_size: int) -> list[list[dict]]:
    """Splits the sentence pool into contiguous chunks of at most chunk_size
    sentences each, preserving order -- used to keep any single LLM call's
    prompt within a small local model's usable context (see CLAUDE.md
    "Local backend"), at the cost of losing whole-document context within
    any one chunk call (mitigated by the doc_summary passed alongside each
    chunk, and for thesis claims, a final merge call over the combined
    candidate pool)."""
    return [sentences[i : i + chunk_size] for i in range(0, len(sentences), chunk_size)]
