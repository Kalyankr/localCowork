"""FastAPI middleware components.

This module contains:
- API key authentication
- Rate limiting
- Exception handlers
"""

import logging
import secrets
import time
from collections import defaultdict

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from agent.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# API Key Authentication (optional)
# =============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)) -> bool:
    """Verify API key if configured, otherwise allow all requests."""
    if settings.api_key is None:
        # No API key configured - allow all (localhost only by default)
        return True

    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key required. Set X-API-Key header.",
        )

    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return True


# =============================================================================
# Rate Limiter (simple in-memory implementation)
# =============================================================================


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for this client."""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self.requests[client_id] = [
            ts for ts in self.requests[client_id] if ts > window_start
        ]

        if len(self.requests[client_id]) >= self.max_requests:
            return False

        self.requests[client_id].append(now)
        return True

    def get_retry_after(self, client_id: str) -> int:
        """Get seconds until rate limit resets."""
        if not self.requests[client_id]:
            return 0
        oldest = min(self.requests[client_id])
        return max(0, int(self.window_seconds - (time.time() - oldest)))


# Global rate limiter instance
rate_limiter = RateLimiter(
    max_requests=settings.rate_limit_requests, window_seconds=settings.rate_limit_window
)


async def check_rate_limit(request: Request) -> bool:
    """Check rate limit for a request."""
    client_ip = request.client.host if request.client else "unknown"

    if not rate_limiter.is_allowed(client_ip):
        retry_after = rate_limiter.get_retry_after(client_ip)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )
    return True


# =============================================================================
# Exception Handlers
# =============================================================================


async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions gracefully."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
            if settings.ollama_model == "debug"
            else "An unexpected error occurred",
        },
    )
