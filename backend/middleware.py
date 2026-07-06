import time
import logging
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
from backend.logging_config import request_id_var

logger = logging.getLogger(__name__)

class SecurityAndLoggingMiddleware(BaseHTTPMiddleware):
    """
    Lightweight, production-ready middleware that handles:
    1. Request ID Generation/Propagation.
    2. Basic Abuse Detection (URI length limits, path scanning prevention).
    3. Structured Request Logging (latency, status code, IP).
    4. HTTP Security Headers Injection (OWASP/CORS defenses).
    """
    async def dispatch(self, request: Request, call_next):
        # 1. Request ID Handling
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        token = request_id_var.set(request_id)

        # Extract client IP address (handle proxies using X-Forwarded-For if available)
        client_ip = request.headers.get("x-forwarded-for")
        if client_ip:
            client_ip = client_ip.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        try:
            # 2. Abuse Detection - URI Length Limit (Mitigate buffer overflow/URL flooding)
            if len(str(request.url)) > 2048:
                logger.warning(
                    f"Abuse detected: URI length ({len(str(request.url))} chars) exceeds limit of 2048 from IP {client_ip}",
                    extra={
                        "event": "abuse_detected",
                        "detail": "uri_too_long",
                        "client_ip": client_ip,
                        "uri_length": len(str(request.url))
                    }
                )
                return JSONResponse(
                    status_code=414,
                    content={"error": {"code": 414, "message": "URI Too Long"}}
                )

            # 3. Abuse Detection - Path Scanning (Reject probes for common security vectors)
            path_lower = request.url.path.lower()
            blocked_scans = [
                ".env", ".git", "wp-admin", "wp-login", "xmlrpc.php",
                "phpmyadmin", "config.php", "setup.php", "eval-stdin.php",
                "cgi-bin", ".php"
            ]
            for pattern in blocked_scans:
                if pattern in path_lower:
                    logger.warning(
                        f"Abuse detected: Scanning path '{request.url.path}' from IP {client_ip}",
                        extra={
                            "event": "abuse_detected",
                            "detail": "path_scan",
                            "pattern": pattern,
                            "client_ip": client_ip
                        }
                    )
                    return JSONResponse(
                        status_code=400,
                        content={"error": {"code": 400, "message": "Bad Request"}}
                    )

            # 4. Request Logging - Start Timer
            start_time = time.perf_counter()

            # Execute the request pipeline
            response = await call_next(request)

            # 5. Request Logging - Latency and Metadata
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"Request completed: {request.method} {request.url.path} -> {response.status_code} ({duration_ms:.2f}ms)",
                extra={
                    "event": "request_completed",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                    "client_ip": client_ip
                }
            )

            # 6. Apply Security Headers (OWASP Security Best Practices)
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
            response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none';"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            logger.error(
                f"Unhandled error in middleware execution: {e}",
                exc_info=e,
                extra={"event": "middleware_error", "client_ip": client_ip}
            )
            # Re-raise so the global exception handlers can format a JSON response
            raise e
        finally:
            # Always reset the contextual Request ID thread/async context variable
            request_id_var.reset(token)
