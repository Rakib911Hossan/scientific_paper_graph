import json

from google.genai import types

from claim_pipeline.llm.client_pool import KeyPool
from claim_pipeline.llm.router import generate_with_retry
from claim_pipeline.ranking.chunking import chunk_sentences
from claim_pipeline.ranking.prompts import THESIS_GENERATION_TEMPLATE, THESIS_MERGE_TEMPLATE


def _generate_claims_for_sentences(
    sentences: list[dict],
    doc_summary: str,
    pool: KeyPool,
    model: str,
    max_claims: int,
    max_retries: int,
) -> list[str]:
    sentence_block = "\n".join(f"[{s['id']}] {s['text']}" for s in sentences)
    prompt = THESIS_GENERATION_TEMPLATE.format(
        doc_summary=doc_summary,
        max_claims=max_claims,
        sentence_block=sentence_block,
    )

    for attempt in range(max_retries):
        temperature = 0.2 if attempt == 0 else 0.5
        response_text = generate_with_retry(
            pool,
            model,
            prompt,
            generation_config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=temperature,
            ),
        )
        try:
            return json.loads(response_text)["claims"]
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                raise
            continue


def _merge_claims(
    candidates: list[str],
    doc_summary: str,
    pool: KeyPool,
    model: str,
    max_claims: int,
    max_retries: int,
) -> list[str]:
    numbered_candidates = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates, start=1))
    prompt = THESIS_MERGE_TEMPLATE.format(
        doc_summary=doc_summary,
        max_claims=max_claims,
        numbered_candidates=numbered_candidates,
    )

    for attempt in range(max_retries):
        temperature = 0.2 if attempt == 0 else 0.5
        response_text = generate_with_retry(
            pool,
            model,
            prompt,
            generation_config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=temperature,
            ),
        )
        try:
            return json.loads(response_text)["claims"]
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                raise
            continue


def generate_thesis_claims(
    sentences: list[dict],
    doc_summary: str,
    pool: KeyPool,
    model: str,
    max_claims: int = 12,
    max_retries: int = 2,
    chunk_size: int | None = None,
) -> list[str]:
    """Synthesizes up to max_claims distinct, high-level findings the paper
    supports, grounded in the full sentence pool and document summary.
    Replaces the old rank/dedupe stages -- instead of selecting existing
    sentences by salience, this generates new standalone thesis statements
    directly, with de-duplication baked into the single generation call
    rather than a separate pass.

    chunk_size: if given and the sentence pool exceeds it, splits into
    contiguous chunks (chunk_sentences()), generates up to max_claims
    candidate claims per chunk independently, then merges the combined
    candidate pool down to max_claims distinct claims with a second call
    (_merge_claims()) -- keeps any single call's prompt within a small
    local model's usable context (see CLAUDE.md "Local backend"). This
    trades completeness for feasibility: the single-call version's own
    "no two claims should restate the same point" instruction only sees
    one chunk at a time here, so cross-chunk duplicates are only caught by
    the final merge, not eliminated at generation time. None (default, and
    what Gemini's huge context window never needs) preserves the original
    single-call behavior.
    """
    if chunk_size is None or len(sentences) <= chunk_size:
        return _generate_claims_for_sentences(sentences, doc_summary, pool, model, max_claims, max_retries)

    chunks = chunk_sentences(sentences, chunk_size)
    candidates: list[str] = []
    for chunk in chunks:
        candidates.extend(_generate_claims_for_sentences(chunk, doc_summary, pool, model, max_claims, max_retries))

    if len(candidates) <= max_claims:
        return candidates
    return _merge_claims(candidates, doc_summary, pool, model, max_claims, max_retries)
