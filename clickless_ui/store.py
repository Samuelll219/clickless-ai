"""Sandbox store: server-side cart (deterministic updates) + interaction log.

The browser cart (localStorage) stays the UI's working copy; every approved
change is sent here as the full cart. The server re-prices every line from the
catalog (the client cannot set prices), rejects unknown barcodes, and appends
one JSON line per event to the interaction log used by the click model.

Files (gitignored): data/sandbox/cart.json, data/sandbox/interactions.jsonl
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock

from clickless_kg.catalog import load_catalog

SANDBOX_DIR = Path(os.environ.get("CLICKLESS_SANDBOX_DIR", "data/sandbox"))
MAX_LINES = 100
MAX_QUANTITY = 99
EVENTS = {"search_submit", "search_result", "add_to_cart", "addon_shown", "addon_add",
          "quantity_change", "remove", "clear", "undo", "confirm_order"}


class StoreError(ValueError):
    pass


class SandboxStore:
    def __init__(self, directory: Path | None = None, catalog: list[dict] | None = None):
        self.dir = Path(directory or SANDBOX_DIR)
        self.catalog = {p["code"]: p for p in (catalog if catalog is not None else load_catalog())}
        self.lock = Lock()

    # ------------------------------------------------------------------ cart
    @property
    def cart_path(self) -> Path:
        return self.dir / "cart.json"

    @property
    def log_path(self) -> Path:
        return self.dir / "interactions.jsonl"

    def get_cart(self) -> dict:
        with self.lock:
            return self._view(self._read())

    def replace_cart(self, lines) -> dict:
        """Set the cart to `lines` = [{code, quantity}, ...] (validated, re-priced)."""
        if not isinstance(lines, list) or len(lines) > MAX_LINES:
            raise StoreError("cart must be a list of at most 100 lines")
        clean: dict[str, int] = {}
        for line in lines:
            if not isinstance(line, dict):
                raise StoreError("each cart line must be an object")
            code, qty = line.get("code"), line.get("quantity")
            if not isinstance(code, str) or code not in self.catalog:
                raise StoreError(f"unknown product code {code!r}")
            if isinstance(qty, bool) or not isinstance(qty, int) or not 1 <= qty <= MAX_QUANTITY:
                raise StoreError("quantity must be a whole number between 1 and 99")
            if code in clean:
                raise StoreError(f"duplicate product code {code!r}")
            clean[code] = qty
        with self.lock:
            before = self._read()
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.cart_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(clean, indent=1), encoding="utf-8")
            tmp.replace(self.cart_path)
            self._append({"event": "cart_update", "before": before, "after": clean})
            return self._view(clean)

    def _read(self) -> dict[str, int]:
        try:
            data = json.loads(self.cart_path.read_text(encoding="utf-8"))
            return {k: v for k, v in data.items() if k in self.catalog}
        except (FileNotFoundError, json.JSONDecodeError, AttributeError):
            return {}

    def _view(self, cart: dict[str, int]) -> dict:
        lines = []
        for code, qty in cart.items():
            p = self.catalog[code]
            cents = round(p["price"] * 100)
            lines.append({"code": code, "name": p["name"], "cents": cents, "quantity": qty,
                          "line_cents": cents * qty})
        return {"lines": lines, "subtotal_cents": sum(l["line_cents"] for l in lines),
                "items": sum(l["quantity"] for l in lines)}

    # ------------------------------------------------------------------ log
    def log(self, event: str, detail=None) -> None:
        if event not in EVENTS:
            raise StoreError(f"unknown event {event!r}")
        if detail is not None and len(json.dumps(detail)) > 4000:
            raise StoreError("event detail too large")
        with self.lock:
            self._append({"event": event, "detail": detail})

    def _append(self, record: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        record = {"ts": round(time.time(), 3), **record}
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def events(self) -> list[dict]:
        try:
            return [json.loads(l) for l in self.log_path.read_text(encoding="utf-8").splitlines() if l]
        except FileNotFoundError:
            return []


def agent_clicks(events: list[dict]) -> list[dict]:
    """Per task (search -> confirm): submissions, approvals, corrections.

    Clicks_Agent = submissions + approvals + corrections (proposal: 2 + e when
    one submission and one approval succeed).
    """
    tasks, cur = [], None
    for e in events:
        kind = e["event"]
        if kind == "search_submit":
            if cur and cur["approvals"] == 0:
                cur["corrections"] += 1          # re-phrased before approving anything
                continue
            cur = {"submissions": 1, "approvals": 0, "corrections": 0}
            tasks.append(cur)
        elif cur and kind in ("add_to_cart", "addon_add"):
            cur["approvals"] += 1
        elif cur and kind in ("quantity_change", "remove", "undo"):
            cur["corrections"] += 1
    for t in tasks:
        t["clicks"] = t["submissions"] + t["approvals"] + t["corrections"]
    return tasks
