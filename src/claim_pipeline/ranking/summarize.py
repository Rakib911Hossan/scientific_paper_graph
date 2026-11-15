from google.genai import types

from claim_pipeline.llm.client_pool import KeyPool
from claim_pipeline.llm.router import generate_with_retry
from claim_pipeline.ranking.chunking import chunk_sentences


def _summarize_chunk(sentences: list[dict], pool: KeyPool, model: str) -> str:
    full_text = " ".join(s["text"] for s in sentences)

    prompt = f"""Summarize the following document in 3-5 sentences, capturing its main argument, findings, and conclusions. Be concise and neutral.

Document:
{full_text}

Return ONLY the summary text, no preamble."""

    return generate_with_retry(
        pool,
        model,
        prompt,
        generation_config=types.GenerateContentConfig(temperature=0.2),
    )


def generate_document_summary(
    sentences: list[dict], pool: KeyPool, model: str, chunk_size: int | None = None
) -> str:
    """Stage 1a: short document-level summary to give the ranking call
    document-aware context (mimics BertSum's document encoding).

    chunk_size: if given and the sentence pool exceeds it, splits into
    contiguous chunks (chunk_sentences()), summarizes each independently,
    then merges the partial summaries into one final summary with a second
    call -- keeps any single call's prompt within a small local model's
    usable context (see CLAUDE.md "Local backend"). None (default, and
    what Gemini's huge context window never needs) preserves the original
    single-call behavior.
    """
    if chunk_size is None or len(sentences) <= chunk_size:
        return _summarize_chunk(sentences, pool, model)

    chunks = chunk_sentences(sentences, chunk_size)
    partial_summaries = [_summarize_chunk(chunk, pool, model) for chunk in chunks]
    combined = "\n\n".join(f"Part {i}: {s}" for i, s in enumerate(partial_summaries, start=1))

    merge_prompt = f"""Below are summaries of different sections of the same paper, in order. Combine them into a single coherent 3-5 sentence summary of the whole paper, capturing its main argument, findings, and conclusions.

{combined}

Return ONLY the combined summary text, no preamble."""

    return generate_with_retry(
        pool,
        model,
        merge_prompt,
        generation_config=types.GenerateContentConfig(temperature=0.2),
    )
