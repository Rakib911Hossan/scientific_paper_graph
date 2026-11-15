import json

from google.genai import types

from claim_pipeline.llm.client_pool import KeyPool
from claim_pipeline.llm.router import generate_with_retry
from claim_pipeline.resolution.prompts import DEDUP_TEMPLATE


def dedup_atomic_claims(
    claims: list[dict],
    pool: KeyPool,
    model: str,
    max_retries: int = 4,
) -> tuple[list[dict], list[dict]]:
    """Drops claims that restate the same underlying fact as another claim
    in the set (same fact, different wording/specificity), keeping the one
    the model judges most complete. One batched call over the whole list,
    same pattern as filter_atomic_claims() -- this is a single dedup
    judgment over the set, not independent per-item work.

    claims: list of dicts, each with at least a "text" key (e.g.
    final_claims.json from the final filter).

    Returns (kept, duplicate_groups): `kept` is the subset of `claims`
    (same dicts, unmodified) to keep, in original order. `duplicate_groups`
    is the raw per-group decisions (indices, keep, reason) as returned by
    the model -- surfaced so a caller can spot-check whether `reason` is
    genuinely fact-specific per group rather than a copy-pasted generic
    string repeated across groups, the same batch-approval failure mode
    found in Stage 2.5's Node Accuracy check (uniform reasoning across all
    items instead of real per-item discrimination).

    If every retry comes back with malformed JSON -- verified concretely
    that this is not just rare bad luck: the `reason` field makes the
    response long enough that some fraction of attempts hit the same
    deterministic-truncation model bug documented elsewhere in this
    pipeline (finish_reason=STOP with the JSON confidently left
    unclosed, reproducible regardless of temperature) -- this is a single
    batched call with no smaller unit to isolate a failure to, unlike
    per-item stages. Rather than raising and losing the whole final claim
    set for a paper over a dedup-only failure, this degrades to a no-op:
    returns all claims unchanged with an empty `duplicate_groups`, and the
    caller is expected to surface that dedup did not run this time.
    """
    numbered_claims = "\n".join(f"{i}. {c['text']}" for i, c in enumerate(claims, start=1))
    prompt = DEDUP_TEMPLATE.format(numbered_claims=numbered_claims)

    duplicate_groups = None
    for attempt in range(max_retries):
        temperature = 0.0 if attempt == 0 else 0.3
        response_text = generate_with_retry(
            pool,
            model,
            prompt,
            generation_config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=temperature,
            ),
            stage="dedup",
        )
        try:
            duplicate_groups = json.loads(response_text)["duplicate_groups"]
            break
        except json.JSONDecodeError:
            continue

    if duplicate_groups is None:
        print(
            f"Dedup: all {max_retries} attempts returned malformed JSON (likely the "
            "known deterministic-truncation model bug). Skipping dedup for this run -- "
            "no claims were dropped. Rerun to retry."
        )
        return claims, []

    drop_indices = set()
    for group in duplicate_groups:
        keep = group["keep"]
        for i in group["indices"]:
            if i != keep:
                drop_indices.add(i)

    kept = [c for i, c in enumerate(claims, start=1) if i not in drop_indices]
    return kept, duplicate_groups
