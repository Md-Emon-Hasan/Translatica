"""
Rate Limiting Configuration
"""

import time
from typing import Callable, Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from limits import parse_many
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from slowapi.wrappers import Limit

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger("rate_limit")


def get_client_ip(request: Request) -> str:
    """
    Get the client IP behind a reverse proxy.

    Nginx and Render both hide the caller, so get_remote_address alone would
    put every user in the same bucket.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return get_remote_address(request)


limiter = Limiter(key_func=get_client_ip)


def _build_limit(item) -> Limit:
    """Wrap a parsed limit so RateLimitExceeded can carry it."""
    return Limit(
        limit=item,
        key_func=get_client_ip,
        scope=None,
        per_method=False,
        methods=None,
        error_message=None,
        exempt_when=None,
        cost=1,
        override_defaults=False,
    )


def _retry_after(item, key: str) -> int:
    """Get the seconds left until the current window resets."""
    try:
        reset_time, _remaining = limiter.limiter.get_window_stats(item, key)
        return max(1, int(reset_time - time.time()) + 1)
    except Exception:  # pragma: no cover - storage specific
        return int(item.get_expiry())


def check_rate_limit(request: Request, limit_value: str, scope: str) -> None:
    """
    Consume one unit of the given limit for this client.

    Raises:
        RateLimitExceeded: If the client is over the limit.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return

    client = get_client_ip(request)

    for item in parse_many(limit_value):
        key = f"{scope}:{client}"
        if not limiter.limiter.hit(item, key):
            request.state.view_rate_limit = (item, [key])
            request.state.retry_after = _retry_after(item, key)
            logger.warning(
                f"Rate limit exceeded for {client} on {scope} "
                f"({limit_value}); retry after {request.state.retry_after}s"
            )
            raise RateLimitExceeded(_build_limit(item))


def rate_limit(setting_name: str, scope: Optional[str] = None) -> Callable:
    """
    Build a dependency enforcing the limit named by setting_name.

    A dependency rather than slowapi's decorator, which needs a parameter
    literally named `request` - already taken by the translate request body.
    """
    bucket = scope or setting_name

    def dependency(request: Request) -> None:
        check_rate_limit(request, getattr(settings, setting_name), bucket)

    return dependency


def rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Return a JSON 429 with a Retry-After header."""
    retry_after = getattr(request.state, "retry_after", 60)
    message = f"Rate limit exceeded: {exc.detail}. Please retry in {retry_after}s."

    return JSONResponse(
        status_code=429,
        content={
            "detail": message,
            "error": message,
            "limit": str(exc.detail),
            "retry_after": retry_after,
        },
        headers={"Retry-After": str(retry_after)},
    )
