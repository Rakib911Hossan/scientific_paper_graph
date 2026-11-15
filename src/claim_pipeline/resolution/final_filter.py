import json

from google.genai import types

from claim_pipeline.llm.client_pool import KeyPool
from claim_pipeline.llm.router import generate_with_retry
from claim_pipeline.resolution.prompts import FINAL_FILTER_TEMPLATE


DEFAULT_BATCH_SIZE = 25


def filter_atomic_claims(
    claims: list[dict],
    pool: KeyPool,
    model: str,
    max_retries: int = 2,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """Keeps only the atomic claims that state an objective, checkable fact
    worth independently verifying -- drops opinions, interpretations,
    hedges/expectations, personal-experience asides, and bare figure/table
    references with no checkable content of their own. Batched in chunks of
    `batch_size` claims per call (not one call for the whole set, and not
    per-claim) -- each claim's KEEP/DROP decision is judged against fixed
    criteria independently of the other claims shown alongside it (unlike
    resolution.dedup, which genuinely needs the whole list in one call to
    find duplicate groups), so chunking doesn't change judgment quality,
    only bounds each individual call's response size.

    Requires the model to return a decision AND a short, claim-specific
    reason for every claim in its chunk (not just dropped ones) --
    verified concretely that a bare keep/drop index list let a legitimate,
    non-numeric general finding ("End-to-end classifiers have a low
    inference cost.") get silently and reproducibly dropped (same result
    at temperature=0 across reruns) with no way to tell why, while
    comparably qualitative claims elsewhere in the same batch survived.
    Symmetric per-item reasoning is the same fix already proven for this
    failure mode in graph.quality's Node/Edge Accuracy checks (a bare
    batch-approval invites pattern-matching instead of individual
    judgment); asking for it only on drops would still let a bad KEEP
    slip through unexamined.

    Chunking (rather than one call with a token budget scaled to the full
    claim count) is the fix for the reasoning requirement's own fallout:
    verified concretely that a fixed max_output_tokens=8192 truncated
    after ~21/100 claims once every claim carries a reason (~390
    tokens/claim including JSON overhead), and scaling the budget with
    total claim count has no ceiling -- it would eventually exceed the
    model's real max output tokens on a large enough paper. A fixed
    per-chunk budget stays bounded regardless of how many claims the
    paper produces.

    claims: list of dicts, each with at least a "text" key (e.g. the
    flattened output of Stage 3's atomic_claims_checkpoint.jsonl). Returns
    the subset of `claims` (same dicts, unmodified) that should be kept,
    in original order.
    """
    kept = []
    for start in range(0, len(claims), batch_size):
        batch = claims[start : start + batch_size]
        numbered_claims = "\n".join(f"{i}. {c['text']}" for i, c in enumerate(batch, start=1))
        prompt = FINAL_FILTER_TEMPLATE.format(numbered_claims=numbered_claims)

        for attempt in range(max_retries):
            temperature = 0.0 if attempt == 0 else 0.3
            response_text = generate_with_retry(
                pool,
                model,
                prompt,
                generation_config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=temperature,
                    max_output_tokens=600 * batch_size + 2000,
                ),
                stage="final_filter",
            )
            try:
                decisions = json.loads(response_text)["decisions"]
                break
            except json.JSONDecodeError:
                if attempt == max_retries - 1:
                    raise
                continue

        keep_local = {
            d["index"] for d in decisions if d.get("decision") == "keep" and 1 <= d.get("index", -1) <= len(batch)
        }
        kept.extend(c for i, c in enumerate(batch, start=1) if i in keep_local)

    return kept
