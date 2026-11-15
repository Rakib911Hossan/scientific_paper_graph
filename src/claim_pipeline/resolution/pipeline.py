import json
import os
from pathlib import Path

from claim_pipeline.llm.client_pool import KeyPool
from claim_pipeline.resolution.atomic_claims import rewrite_qa_to_claims
from claim_pipeline.resolution.evidence import generate_evidence_questions
from claim_pipeline.resolution.qa import answer_with_bm25
from claim_pipeline.resolution.retrieval import BM25Retriever


def generate_claim_evidence(
    claim_id: str,
    claim_text: str,
    bm25_retriever: BM25Retriever,
    pool: KeyPool,
    model: str,
    doc_summary: str,
    n_questions: int = 5,
    candidate_top_k: int = 8,
    bm25_top_k: int = 5,
) -> dict:
    questions = generate_evidence_questions(
        claim_text, doc_summary, bm25_retriever, pool, model, n=n_questions, candidate_top_k=candidate_top_k
    )
    qa_pairs = [answer_with_bm25(q, bm25_retriever, pool, model, bm25_top_k, doc_summary) for q in questions]

    return {
        "id": claim_id,
        "claim": claim_text,
        "qa_pairs": qa_pairs,
    }


def load_completed_ids(checkpoint_path: Path, required_key: str = "qa_pairs") -> dict:
    completed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if required_key not in rec:
                        continue
                    completed[rec["id"]] = rec
                except json.JSONDecodeError:
                    continue
    return completed


def append_checkpoint(checkpoint_path: Path, record: dict):
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_pipeline(
    theses: list[str],
    full_sentences: list[dict],
    checkpoint_path: Path,
    pool: KeyPool,
    model: str,
    doc_summary: str,
    n_questions: int = 5,
    candidate_top_k: int = 8,
    bm25_top_k: int = 5,
) -> list[dict]:
    """
    theses: synthesized claim strings (from synthesis.thesis.generate_thesis_claims).
    full_sentences: the full sentence pool (the *_sentences.json from
        run_ingest.py), used as the BM25 corpus for both grounding evidence
        question generation and answering those questions.
    doc_summary: Stage 1's whole-document summary, prepended to every
        answering call (see qa.answer_question for why).
    """
    bm25_retriever = BM25Retriever(full_sentences)

    completed = load_completed_ids(checkpoint_path)
    if completed:
        print(f"Resuming: {len(completed)} claims already completed, will skip them.")

    results = []
    for i, claim_text in enumerate(theses, start=1):
        claim_id = f"C{i}"
        if claim_id in completed:
            results.append(completed[claim_id])
            continue

        try:
            result = generate_claim_evidence(
                claim_id, claim_text, bm25_retriever, pool, model, doc_summary, n_questions, candidate_top_k, bm25_top_k
            )
        except Exception as e:
            print(f"Error processing {claim_id}: {e}. Skipping for now (rerun to retry).")
            continue

        append_checkpoint(checkpoint_path, result)
        results.append(result)

    return results


def decontextualize_claim_evidence(evidence_record: dict, pool: KeyPool, model: str) -> dict:
    """Rewrites every (question, answer) pair for one thesis claim into
    standalone, decontextualized atomic claims -- one QA pair may expand
    into several atomic claims if its answer packs multiple distinct
    facts. Skips QA pairs with an empty answer (nothing to rewrite).

    Each QA pair is rewritten independently and a failure on one (e.g. the
    model producing malformed JSON it itself considers complete --
    verified concretely: finish_reason=STOP on a deterministically
    truncated response, not a token-budget or safety cutoff, so retries at
    any temperature reproduce it byte-for-byte) does not discard the
    other, successfully-rewritten QA pairs for this same claim."""
    claim_text = evidence_record["claim"]
    atomic_claims = []

    for qa in evidence_record["qa_pairs"]:
        try:
            texts = rewrite_qa_to_claims(claim_text, qa["question"], qa["answer"], pool, model)
        except Exception as e:
            print(f"  Error rewriting QA pair ({qa['question'][:60]}...): {e}. Skipping this pair.")
            continue
        for text in texts:
            atomic_claims.append({"text": text, "question": qa["question"], "answer": qa["answer"]})

    return {
        "id": evidence_record["id"],
        "source_claim": claim_text,
        "atomic_claims": atomic_claims,
    }


def run_decontextualization(
    evidence_records: list[dict],
    checkpoint_path: Path,
    pool: KeyPool,
    model: str,
) -> list[dict]:
    """evidence_records: the checkpointed output of run_pipeline() (each
    with id, claim, qa_pairs) -- rewrites every QA pair across all of them
    into standalone atomic claims, one output record per source claim."""
    completed = load_completed_ids(checkpoint_path, required_key="atomic_claims")
    if completed:
        print(f"Resuming: {len(completed)} claims already completed, will skip them.")

    results = []
    for record in evidence_records:
        claim_id = record["id"]
        if claim_id in completed:
            results.append(completed[claim_id])
            continue

        try:
            result = decontextualize_claim_evidence(record, pool, model)
        except Exception as e:
            print(f"Error processing {claim_id}: {e}. Skipping for now (rerun to retry).")
            continue

        append_checkpoint(checkpoint_path, result)
        results.append(result)

    return results
