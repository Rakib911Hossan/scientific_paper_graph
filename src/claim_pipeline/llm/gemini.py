import time

import httpx
from google.genai.errors import ClientError, ServerError

from claim_pipeline.llm.client_pool import KeyPool


def generate_with_retry(
    pool: KeyPool,
    model: str,
    prompt: str,
    generation_config=None,
    max_retries: int = 2,
) -> str:
    """Round-robins the key pool, rate-limits per key, and retries on
    429 (RESOURCE_EXHAUSTED) / 503 (UNAVAILABLE) client errors, 503/504
    (UNAVAILABLE/DEADLINE_EXCEEDED) server errors, and on the underlying
    HTTP connection timing out or dropping -- KeyPool bounds every request
    to a finite timeout (rather than blocking indefinitely on a dropped
    connection), so a slow/flaky network path surfaces here as
    httpx.TimeoutException/TransportError instead of hanging; verified
    concretely that this occurs in practice, not just in theory. Used by
    both the ranking (Stage 1) and resolution (Stage 2) calls.

    ClientError (4xx) and ServerError (5xx) are raised as distinct
    exception classes by google-genai -- a bare 504 DEADLINE_EXCEEDED
    (ServerError, not ClientError) previously propagated uncaught on the
    first attempt with no retry at all, verified concretely against a live
    504 from the API.
    """
    for attempt in range(max_retries):
        client, limiter = pool.next()
        limiter.wait()
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=generation_config,
            )
            return response.text.strip()
        except ClientError as e:
            code = getattr(e, "code", None)
            msg = str(e)
            if code == 429 or "RESOURCE_EXHAUSTED" in msg:
                wait = 51
            elif code == 503 or "UNAVAILABLE" in msg:
                wait = 10
            else:
                raise
        except ServerError as e:
            code = getattr(e, "code", None)
            msg = str(e)
            if code in (503, 504) or "UNAVAILABLE" in msg or "DEADLINE_EXCEEDED" in msg:
                wait = 10
            else:
                raise
        except (httpx.TimeoutException, httpx.TransportError):
            wait = 10
        print(f"Retrying after {wait}s (attempt {attempt + 1}/{max_retries})...")
        time.sleep(wait)
        continue
    raise RuntimeError("Exceeded max retries due to rate limiting.")


def call_with_retry(fn, *args, max_retries: int = 5, base_delay: float = 2, **kwargs):
    """Generic exponential-backoff wrapper for transient 503/UNAVAILABLE errors
    around a single-shot call (e.g. ranking, dedup)."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                delay = base_delay * (2**attempt)
                print(f"Model overloaded, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Max retries exceeded")
