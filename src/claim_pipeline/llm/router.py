"""Single dispatch seam between the Gemini API backend and the local
transformers-backed backend. Every pipeline stage imports generate_with_retry from here instead
of directly from llm.gemini, so switching backends is one config value
(configs/default.yaml -> backend: gemini | local, or CLAIM_PIPELINE_BACKEND
env var) rather than a code change per stage -- no stage's own logic or
generation_config construction (temperature, response_mime_type) changes
either way.
"""
import os

from claim_pipeline.llm import gemini as _gemini


def generate_with_retry(pool, model, prompt, generation_config=None, max_retries: int = 2) -> str:
    if os.environ.get("CLAIM_PIPELINE_BACKEND", "gemini") == "local":
        from claim_pipeline.llm import local_model

        return local_model.generate(prompt, generation_config, max_retries=max_retries)
    return _gemini.generate_with_retry(pool, model, prompt, generation_config, max_retries)
