"""Local WSGI endpoint for Soumya's Config C pipeline."""

import json
import logging
from pathlib import Path
from threading import BoundedSemaphore


STATIC_ROOT = Path(__file__).with_name("static")
ASSETS = {
    "/": ("index.html", "text/html"),
    "/styles.css": ("styles.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}
MAX_REQUEST_BYTES = 8192
LOGGER = logging.getLogger(__name__)
SEARCH_SLOT = BoundedSemaphore(1)


def respond(start_response, status, body, content_type="application/json"):
    if content_type == "application/json":
        body = json.dumps(body, allow_nan=False).encode("utf-8")
    start_response(status, [
        ("Content-Type", f"{content_type}; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store"),
    ])
    return [body]


def search(environ, start_response):
    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
        if not 0 < length <= MAX_REQUEST_BYTES:
            raise ValueError("Invalid request length")
        payload = json.loads(environ["wsgi.input"].read(length))
        request = payload.get("request") if isinstance(payload, dict) else None
        if not isinstance(request, str) or not 0 < len(request.strip()) <= 500:
            raise ValueError("Invalid shopping request")
    except (ValueError, UnicodeError):
        return respond(start_response, "400 Bad Request", {
            "error": "Enter a shopping request between 1 and 500 characters.",
        })

    try:
        from clickless_llm.config_c import run_config_c
        from clickless_llm.pipeline import PipelineError
        from clickless_llm.validator import ValidationError
        from neo4j.exceptions import AuthError, ServiceUnavailable, SessionExpired
    except ModuleNotFoundError:
        LOGGER.exception("Could not import Config C")
        return respond(start_response, "503 Service Unavailable", {
            "error": "Search is unavailable because the server setup is incomplete.",
        })

    if not SEARCH_SLOT.acquire(blocking=False):
        return respond(start_response, "429 Too Many Requests", {
            "error": "Another search is still finishing. Please retry shortly.",
        })
    try:
        result = run_config_c(request.strip())
        return respond(start_response, "200 OK", result)
    except PipelineError as exc:
        LOGGER.exception("Constraint extraction failed")
        if isinstance(exc.__cause__, ValidationError):
            return respond(start_response, "422 Unprocessable Entity", {
                "error": "We could not interpret this request. Include a product category and try rephrasing it.",
            })
        return respond(start_response, "503 Service Unavailable", {
            "error": "The language service could not finish your request. Please try again shortly.",
        })
    except (ServiceUnavailable, SessionExpired, AuthError):
        LOGGER.exception("Product database unavailable")
        return respond(start_response, "503 Service Unavailable", {
            "error": "The product database is unavailable. Please try again shortly.",
        })
    except Exception:
        LOGGER.exception("Config C search failed")
        return respond(start_response, "502 Bad Gateway", {
            "error": "The search could not finish. Please try again.",
        })
    finally:
        SEARCH_SLOT.release()


def application(environ, start_response):
    path = environ["PATH_INFO"]
    method = environ["REQUEST_METHOD"]
    if path == "/api/search" and method == "POST":
        return search(environ, start_response)
    if path in ASSETS and method == "GET":
        filename, content_type = ASSETS[path]
        return respond(start_response, "200 OK", (STATIC_ROOT / filename).read_bytes(), content_type)
    if path in ASSETS or path == "/api/search":
        return respond(start_response, "405 Method Not Allowed", {"error": "Method not allowed."})
    return respond(start_response, "404 Not Found", {"error": "Page not found."})
