import time
from collections import deque
from threading import Lock

from fastapi import HTTPException, Request

from app.config import settings

_rate_lock = Lock()
_request_buckets: dict[str, deque[float]] = {}
_WINDOW_SECONDS = 60.0


def reset_rate_limits() -> None:
    """Clear tracked request timestamps (useful in automated tests)."""
    with _rate_lock:
        _request_buckets.clear()


def enforce_rate_limit(request: Request) -> None:
    """Apply a simple per-client sliding-window request limit."""
    limit = settings.RATE_LIMIT_PER_MINUTE
    if limit <= 0:
        return

    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    cutoff = now - _WINDOW_SECONDS

    with _rate_lock:
        bucket = _request_buckets.setdefault(client_ip, deque())
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= limit:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait a moment before trying again.",
            )
        bucket.append(now)
