import json

from google.genai import types

from claim_pipeline.llm.client_pool import KeyPool
from claim_pipeline.llm.router import generate_with_retry
from claim_pipeline.resolution.prompts import EVIDENCE_QUESTIONS_TEMPLATE
from claim_pipeline.resolution.retrieval import BM25Retriever


def generate_evidence_questions(
    claim: str,
    doc_summary: str,
    bm25_retriever: BM25Retriever,
    pool: KeyPool,
    model: str,
    n: int = 5,
    candidate_top_k: int = 8,
    max_retries: int = 2,
) -> list[str]:
    """Generates up to n distinct evidence questions for a claim (at least
    1, fewer than n if the claim doesn't genuinely warrant that many
    distinct evidence points -- the model decides, same "up to N" pattern
    as generate_thesis_claims), grounded in BM25-retrieved sentences
    relevant to the claim's own text -- not just the global document
    summary. The summary only covers ~4 dominant themes and often says
    nothing about a longer-tail claim's specific topic; without
    claim-specific grounding, question generation defaults to whatever the
    summary happens to cover, producing questions that are off-topic for
    the claim they're supposedly evidencing (verified concretely: a claim
    about combining fine-tuning with RAG/MoE/model compression generated
    questions about overfitting and data curation instead -- both were
    themes prominent in the doc summary, neither relevant to this claim --
    until grounded in BM25-retrieved candidates about the claim itself)."""
    candidates = bm25_retriever.retrieve(claim, top_k=candidate_top_k)
    candidate_block = "\n".join(f"- {c['text']}" for c in candidates)

    prompt = EVIDENCE_QUESTIONS_TEMPLATE.format(
        doc_summary=doc_summary,
        candidate_block=candidate_block,
        claim=claim,
        n=n,
    )

    for attempt in range(max_retries):
        # Vary temperature on retry -- see resolution/atomic_claims.py for a
        # concretely verified case where retrying at a fixed temperature
        # reproduced the exact same truncated/invalid JSON every time.
        temperature = 0.2 if attempt == 0 else 0.5
        response_text = generate_with_retry(
            pool,
            model,
            prompt,
            generation_config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=temperature,
            ),
            stage="evidence_questions",
        )
        try:
            return json.loads(response_text)["questions"]
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                raise
            continue
