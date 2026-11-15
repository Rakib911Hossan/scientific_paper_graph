import json

from google.genai import types

from claim_pipeline.llm.client_pool import KeyPool
from claim_pipeline.llm.router import generate_with_retry
from claim_pipeline.resolution.prompts import ATOMIC_CLAIMS_TEMPLATE


def rewrite_qa_to_claims(
    claim: str,
    question: str,
    answer: str,
    pool: KeyPool,
    model: str,
    max_retries: int = 2,
) -> list[str]:
    """Rewrites a single (question, answer) evidence pair into one or more
    standalone, decontextualized, atomic claim sentences -- splitting the
    answer into separate claims if it packs multiple distinct facts (e.g.
    an answer listing several filtering steps becomes one claim per step),
    resolving any pronouns/vague references using the source claim and
    question as context. Returns an empty list if the answer was empty
    (nothing to rewrite)."""
    if not answer or not answer.strip():
        return []

    prompt = ATOMIC_CLAIMS_TEMPLATE.format(claim=claim, question=question, answer=answer)

    for attempt in range(max_retries):
        # temperature=0.0 can deterministically truncate the same response
        # on every attempt -- verified concretely (3/3 identical truncated
        # outputs, byte-for-byte, for the same prompt). Retrying at the same
        # temperature can never recover from that, so bump temperature on
        # retry to actually have a chance at a different (complete) output.
        temperature = 0.0 if attempt == 0 else 0.3
        response_text = generate_with_retry(
            pool,
            model,
            prompt,
            generation_config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=temperature,
            ),
            stage="atomic_claims",
        )
        try:
            return json.loads(response_text)["claims"]
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                raise
            continue
