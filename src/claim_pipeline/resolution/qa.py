from google.genai import types

from claim_pipeline.llm.client_pool import KeyPool
from claim_pipeline.llm.router import generate_with_retry
from claim_pipeline.resolution.retrieval import BM25Retriever


def _is_empty_answer(answer: str) -> bool:
    """Substring check, not exact match, on the NOT_IN_CONTEXT sentinel --
    Gemini reliably outputs the bare token, but a weaker/smaller model
    (e.g. this pipeline's local backend) can instead echo the whole
    instruction sentence containing it ("If the context does NOT contain
    the answer, respond with exactly: NOT_IN_CONTEXT") or wrap it in extra
    text, which fails an exact-match check and lets a non-answer flow
    through as if it were real, ending up decontextualized into a bogus
    claim downstream. Verified concretely against a local-backend run."""
    return not answer or len(answer.strip()) < 2 or "NOT_IN_CONTEXT" in answer


def answer_question(
    context_text: str,
    question: str,
    pool: KeyPool,
    model: str,
    doc_summary: str | None = None,
    max_retries: int = 2,
) -> str:
    """
    doc_summary: an optional whole-document summary (from Stage 1's
    generate_document_summary), prepended to the context. Some referents
    (e.g. "the organizing team", "this competition") are established once
    near the top of a paper and never re-named near later mentions -- no
    local window or BM25 top-k reliably re-surfaces that single naming
    fact, so it's included directly rather than relied on to be retrieved.
    """
    context_block = f"Document summary: {doc_summary}\n\n{context_text}" if doc_summary else context_text

    prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using ONLY the context above -- do not use outside/parametric "
        "knowledge, even if you know the real-world answer. "
        "Respond with ONLY the core factual answer as a short phrase or clause -- "
        "no markdown formatting, no hedging phrases like 'Based on the context', "
        "no restating the question. "
        "If the context includes numbers, formulas, or named entities relevant to the answer, "
        "copy them exactly as written -- do not paraphrase or round them.\n"
        "If the context does NOT contain the answer, respond with exactly: NOT_IN_CONTEXT\n"
        "Answer:"
    )
    return generate_with_retry(
        pool,
        model,
        prompt,
        generation_config=types.GenerateContentConfig(temperature=0.0),
        max_retries=max_retries,
        stage="evidence_answer",
    )


def answer_with_bm25(
    question: str,
    bm25_retriever: BM25Retriever,
    pool: KeyPool,
    model: str,
    bm25_top_k: int = 5,
    doc_summary: str | None = None,
) -> dict:
    """Answers a question by retrieving BM25 top-k over the full document.
    Retries once more on the same context if the first attempt comes back
    empty before giving up -- even at temperature=0.0, hosted-model
    batching can make the same prompt return NOT_IN_CONTEXT on one call
    and the correct answer on the next (non-associative floating point in
    batched serving, not true sampling), so a single retry recovers a real
    share of these without masking genuinely-unanswerable questions (which
    still empty out after the retry)."""
    bm25_results = bm25_retriever.retrieve(question, top_k=bm25_top_k)
    context_text = "\n".join(r["text"] for r in bm25_results)

    answer = answer_question(context_text, question, pool, model, doc_summary=doc_summary)
    retried = False
    if _is_empty_answer(answer):
        retried = True
        answer = answer_question(context_text, question, pool, model, doc_summary=doc_summary)

    if _is_empty_answer(answer):
        answer = ""  # give downstream nothing rather than a literal token

    return {"question": question, "answer": answer, "retried": retried}
