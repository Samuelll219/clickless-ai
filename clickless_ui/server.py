"""Local WSGI endpoint for Soumya's Config C pipeline."""

import json
import logging
import os
from pathlib import Path
from threading import BoundedSemaphore, Lock


STATIC_ROOT = Path(__file__).with_name("static")
ASSETS = {
    "/": ("index.html", "text/html"),
    "/styles.css": ("styles.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
    "/extras.js": ("extras.js", "text/javascript"),
}
MAX_REQUEST_BYTES = 8192
LOGGER = logging.getLogger(__name__)
SEARCH_SLOT = BoundedSemaphore(1)
_SHARED: dict = {}
_SHARED_LOCK = Lock()


def offline_kg_enabled() -> bool:
    """CLICKLESS_KG=offline serves Config C from the catalog (same semantics as Cypher)."""
    return os.environ.get("CLICKLESS_KG", "neo4j").lower() == "offline"


def shared(name):
    """Lazily built singletons: catalog, offline KG, add-on rules, sandbox store."""
    with _SHARED_LOCK:
        if name not in _SHARED:
            from clickless_kg.catalog import load_catalog
            if name == "catalog":
                _SHARED[name] = load_catalog()
            elif name == "offline_kg":
                from clickless_mining.addons import OfflineKG
                _SHARED[name] = OfflineKG(load_catalog())
            elif name == "rules":
                from clickless_mining.addons import RuleIndex
                from clickless_mining.apriori_rules import load_rules
                rules = load_rules(os.environ.get("CLICKLESS_RULES", "data/addon_rules.json"))
                _SHARED[name] = RuleIndex(rules["rules"], rules["popularity"],
                                          rank_by=rules["meta"].get("rank_by", "lift"))
            elif name == "store":
                from .store import SandboxStore
                _SHARED[name] = SandboxStore()
        return _SHARED[name]


def read_json(environ, max_bytes=MAX_REQUEST_BYTES):
    length = int(environ.get("CONTENT_LENGTH") or 0)
    if not 0 < length <= max_bytes:
        raise ValueError("Invalid request length")
    return json.loads(environ["wsgi.input"].read(length))


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
        if offline_kg_enabled():
            result = run_config_c(request.strip(), kg=shared("offline_kg"))
        else:
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


def addons(environ, start_response):
    """POST {category, constraints} -> up to 3 add-on products with the rule behind each."""
    try:
        payload = read_json(environ)
        category = payload.get("category")
        constraints = payload.get("constraints") or {}
        from clickless_kg import vocab
        if category not in vocab.CATEGORY_NAMES or not isinstance(constraints, dict):
            raise ValueError("bad category")
        tags = [t for t in constraints.get("dietary_tags", []) if t in vocab.DIETARY_TAG_NAMES]
        allergens = [a for a in constraints.get("excluded_allergens", []) if a in vocab.ALLERGEN_NAMES]
    except (ValueError, UnicodeError, AttributeError, TypeError):
        return respond(start_response, "400 Bad Request", {"error": "Invalid add-on request."})
    try:
        from clickless_mining.addons import recommend_addons
        from clickless_mining.apriori_rules import format_rule
        index = shared("rules")
        kg = shared("offline_kg")
        items = recommend_addons([category], {"dietary_tags": tags, "excluded_allergens": allergens},
                                 index, kg, k=3)
    except FileNotFoundError:
        return respond(start_response, "503 Service Unavailable",
                       {"error": "Add-on rules are not available. Run scripts/run_apriori.py."})
    out = [{"category": a["category"], "source": a["source"],
            "explanation": format_rule(a["rule"]) if a["rule"] else "popular category",
            "product": a["product"]} for a in items]
    return respond(start_response, "200 OK", {"addons": out})


def cart(environ, start_response, method):
    from .store import StoreError
    store = shared("store")
    if method == "GET":
        return respond(start_response, "200 OK", store.get_cart())
    try:
        payload = read_json(environ, max_bytes=65536)
        return respond(start_response, "200 OK", store.replace_cart(payload.get("lines")))
    except (ValueError, UnicodeError, AttributeError, StoreError) as exc:
        message = str(exc) if isinstance(exc, StoreError) else "Invalid cart update."
        return respond(start_response, "400 Bad Request", {"error": message})


def log_event(environ, start_response):
    from .store import StoreError
    try:
        payload = read_json(environ)
        shared("store").log(payload.get("event"), payload.get("detail"))
    except (ValueError, UnicodeError, AttributeError, StoreError):
        return respond(start_response, "400 Bad Request", {"error": "Invalid event."})
    return respond(start_response, "200 OK", {"ok": True})


def application(environ, start_response):
    path = environ["PATH_INFO"]
    method = environ["REQUEST_METHOD"]
    if path == "/api/search" and method == "POST":
        return search(environ, start_response)
    if path == "/api/addons" and method == "POST":
        return addons(environ, start_response)
    if path == "/api/cart" and method in ("GET", "PUT"):
        return cart(environ, start_response, method)
    if path == "/api/log" and method == "POST":
        return log_event(environ, start_response)
    if path in ASSETS and method == "GET":
        filename, content_type = ASSETS[path]
        return respond(start_response, "200 OK", (STATIC_ROOT / filename).read_bytes(), content_type)
    if path in ASSETS or path in ("/api/search", "/api/addons", "/api/cart", "/api/log"):
        return respond(start_response, "405 Method Not Allowed", {"error": "Method not allowed."})
    return respond(start_response, "404 Not Found", {"error": "Page not found."})
