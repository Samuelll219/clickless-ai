"""Local WSGI endpoint for Soumya's Config C pipeline."""

import json
import logging
from pathlib import Path


STATIC_ROOT = Path(__file__).with_name("static")
ASSETS = {
    "/": ("index.html", "text/html"),
    "/styles.css": ("styles.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}
MAX_REQUEST_BYTES = 8192
LOGGER = logging.getLogger(__name__)


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
    except ModuleNotFoundError as exc:
        LOGGER.exception("Could not import Config C")
        message = "Install the project dependencies from requirements.txt."
        if exc.name in {"clickless_llm", "clickless_llm.config_c"}:
            message = "Config C is unavailable. Run this UI with Soumya’s PR #1 included."
        return respond(start_response, "503 Service Unavailable", {"error": message})

    try:
        result = run_config_c(request.strip())
        return respond(start_response, "200 OK", result)
    except Exception:
        LOGGER.exception("Config C search failed")
        return respond(start_response, "502 Bad Gateway", {
            "error": "The search could not finish. Check that Ollama and Neo4j are running, or try rephrasing your request.",
        })


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
