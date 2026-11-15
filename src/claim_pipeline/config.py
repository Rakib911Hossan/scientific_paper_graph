import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class Config:
    model: str
    backend: str
    local_model_id: str
    local_dtype: str
    local_max_output_tokens: int
    chunk_size_sentences: int | None
    min_seconds_between_calls: float
    max_retries: int
    base_delay: float
    max_claims: int
    evidence_max_questions_per_claim: int
    evidence_candidate_top_k: int
    bm25_top_k: int
    data_raw: Path
    data_interim: Path
    data_processed: Path
    data_checkpoints: Path
    data_graph: Path
    api_keys: list


def load_config(config_path: str | Path | None = None) -> Config:
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    backend = os.environ.get("CLAIM_PIPELINE_BACKEND", raw.get("backend", "gemini"))

    api_keys = []
    if backend == "gemini":
        api_keys_env = os.environ.get("GEMINI_API_KEYS", "")
        api_keys = [k.strip() for k in api_keys_env.split(",") if k.strip()]
        if not api_keys:
            raise RuntimeError(
                "GEMINI_API_KEYS is not set. Copy .env.example to .env and fill in "
                "one or more comma-separated Gemini API keys, or set backend: local "
                "in configs/default.yaml (or CLAIM_PIPELINE_BACKEND=local) to run "
                "without the Gemini API."
            )

    local_raw = raw.get("local", {})

    chunk_size_env = os.environ.get("CLAIM_PIPELINE_CHUNK_SIZE_SENTENCES")
    if chunk_size_env is not None:
        chunk_size_sentences = int(chunk_size_env) if chunk_size_env else None
    else:
        chunk_size_sentences = raw.get("chunking", {}).get("max_sentences")

    return Config(
        model=raw["model"],
        backend=backend,
        local_model_id=os.environ.get("CLAIM_PIPELINE_LOCAL_MODEL", local_raw.get("model_id", "Qwen/Qwen2.5-3B-Instruct")),
        local_dtype=os.environ.get("CLAIM_PIPELINE_LOCAL_DTYPE", local_raw.get("dtype", "float16")),
        local_max_output_tokens=local_raw.get("max_output_tokens", 4096),
        chunk_size_sentences=chunk_size_sentences,
        min_seconds_between_calls=raw["rate_limit"]["min_seconds_between_calls"],
        max_retries=raw["retry"]["max_retries"],
        base_delay=raw["retry"]["base_delay"],
        max_claims=raw["thesis"]["max_claims"],
        evidence_max_questions_per_claim=raw["evidence"]["max_questions_per_claim"],
        evidence_candidate_top_k=raw["evidence"]["candidate_top_k"],
        bm25_top_k=raw["evidence"]["bm25_top_k"],
        data_raw=PROJECT_ROOT / raw["paths"]["data_raw"],
        data_interim=PROJECT_ROOT / raw["paths"]["data_interim"],
        data_processed=PROJECT_ROOT / raw["paths"]["data_processed"],
        data_checkpoints=PROJECT_ROOT / raw["paths"]["data_checkpoints"],
        data_graph=PROJECT_ROOT / raw["paths"]["data_graph"],
        api_keys=api_keys,
    )
