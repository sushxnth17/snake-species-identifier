import io
import time
import logging
import asyncio
from collections import defaultdict
import threading
from fastapi import HTTPException, Request, Depends
from backend.config import Settings
from backend.dependencies import get_settings

logger = logging.getLogger(__name__)

class InMemoryRateLimiter:
    """
    Thread-safe, sliding-window in-memory rate limiter.
    Tracks timestamps of requests per client IP address.
    """
    def __init__(self):
        self.history = defaultdict(list)
        self.lock = threading.Lock()

    def check_rate_limit(self, ip: str, limit: int, window: int) -> int:
        """
        Checks if the IP has exceeded the limit in the given window.
        Returns 0 if NOT rate limited.
        If rate limited, returns the number of seconds (retry_after) the client needs to wait.
        """
        now = time.time()
        with self.lock:
            timestamps = self.history[ip]
            # Filter timestamps to keep only those within the sliding window
            valid_timestamps = [t for t in timestamps if now - t < window]
            
            if len(valid_timestamps) >= limit:
                # Update the stored timestamps list
                self.history[ip] = valid_timestamps
                # Calculate time until oldest timestamp in the current window slides out
                oldest = valid_timestamps[0]
                retry_after = int(window - (now - oldest)) + 1
                return max(1, retry_after)
                
            # Add current timestamp and update history
            valid_timestamps.append(now)
            self.history[ip] = valid_timestamps
            return 0

    def clean_expired(self, window: int):
        """Cleans up expired timestamps to prevent memory leaks."""
        now = time.time()
        with self.lock:
            for ip in list(self.history.keys()):
                valid_timestamps = [t for t in self.history[ip] if now - t < window]
                if not valid_timestamps:
                    del self.history[ip]
                else:
                    self.history[ip] = valid_timestamps


# Global instance of the rate limiter
rate_limiter_instance = InMemoryRateLimiter()


async def check_rate_limit_dependency(
    request: Request,
    settings: Settings = Depends(get_settings)
) -> None:
    """
    FastAPI dependency that enforces rate limiting on endpoints.
    Can be disabled globally via setting rate_limit_enabled = False.
    """
    if not settings.rate_limit_enabled:
        return

    # Extract client IP (handle proxies using X-Forwarded-For if available)
    client_ip = request.headers.get("x-forwarded-for")
    if client_ip:
        client_ip = client_ip.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"

    retry_after = rate_limiter_instance.check_rate_limit(
        client_ip,
        settings.rate_limit_requests,
        settings.rate_limit_window
    )
    
    if retry_after > 0:
        logger.warning(
            f"Rate limit exceeded for IP {client_ip}. Limit: {settings.rate_limit_requests}/{settings.rate_limit_window}s. Retry after: {retry_after}s",
            extra={
                "event": "rate_limit_exceeded",
                "client_ip": client_ip,
                "retry_after": retry_after
            }
        )
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests. Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(retry_after)}
        )


async def rate_limiter_cleanup_task(window: int, interval: int = 600):
    """
    Background loop that runs periodically to clear out stale rate limit records,
    preventing long-term memory leaks.
    """
    try:
        while True:
            await asyncio.sleep(interval)
            logger.info("Running expired rate limit entries cleanup...", extra={"event": "rate_limit_cleanup"})
            rate_limiter_instance.clean_expired(window)
    except asyncio.CancelledError:
        logger.info("Rate limiter cleanup task cancelled.", extra={"event": "rate_limit_cleanup_cancelled"})
