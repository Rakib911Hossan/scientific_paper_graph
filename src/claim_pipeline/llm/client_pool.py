import time

from google import genai
from google.genai import types

DEFAULT_REQUEST_TIMEOUT_MS = 120_000  # bounds a single call so a dropped connection fails into the existing retry logic instead of blocking forever


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.last_call = 0.0

    def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


class KeyPool:
    """Round-robins Gemini clients so each API key gets its own rate limiter."""

    def __init__(self, keys: list[str], min_interval: float, request_timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS):
        if not keys:
            raise ValueError("KeyPool requires at least one API key")
        http_options = types.HttpOptions(timeout=request_timeout_ms)
        self.clients = [genai.Client(api_key=k, http_options=http_options) for k in keys]
        self.limiters = [RateLimiter(min_interval) for _ in keys]
        self.i = 0

    def next(self):
        client = self.clients[self.i]
        limiter = self.limiters[self.i]
        self.i = (self.i + 1) % len(self.clients)
        return client, limiter
