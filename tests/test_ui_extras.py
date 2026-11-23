"""Tests for the sandbox store, add-on/cart/log routes and offline-KG search."""
import json
from io import BytesIO
from unittest.mock import patch

import pytest

from clickless_ui import server
from clickless_ui.store import SandboxStore, StoreError, agent_clicks

CATALOG = [
    {"code": "001", "name": "Oat Milk", "price": 3.49, "category": "plant_milk",
     "dietary_tags": ["vegan"], "allergens": []},
    {"code": "002", "name": "Corn Flakes", "price": 4.10, "category": "breakfast_cereal",
     "dietary_tags": ["vegan"], "allergens": ["gluten"]},
]


def call(path, method="POST", body=None):
    raw = b"" if body is None else json.dumps(body).encode()
    status = []
    out = b"".join(server.application({
        "PATH_INFO": path, "REQUEST_METHOD": method, "CONTENT_LENGTH": str(len(raw)),
        "wsgi.input": BytesIO(raw)}, lambda code, headers: status.append(code)))
    return status[0], json.loads(out)


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = SandboxStore(tmp_path, CATALOG)
    monkeypatch.setitem(server._SHARED, "store", s)
    yield s
    server._SHARED.pop("store", None)


def test_store_reprices_from_catalog_and_logs(store):
    view = store.replace_cart([{"code": "001", "quantity": 2, "cents": 1}])
    assert view["subtotal_cents"] == 698 and view["items"] == 2
    assert store.get_cart()["lines"][0]["name"] == "Oat Milk"
    assert store.events()[-1]["event"] == "cart_update"


@pytest.mark.parametrize("lines", [[{"code": "999", "quantity": 1}], [{"code": "001", "quantity": 0}],
                                   [{"code": "001", "quantity": True}], "nope",
                                   [{"code": "001", "quantity": 1}, {"code": "001", "quantity": 2}]])
def test_store_rejects_bad_lines(store, lines):
    with pytest.raises(StoreError):
        store.replace_cart(lines)


def test_cart_and_log_routes(store):
    assert call("/api/cart", "PUT", {"lines": [{"code": "002", "quantity": 1}]})[1]["subtotal_cents"] == 410
    assert call("/api/cart", "GET")[1]["items"] == 1
    assert call("/api/cart", "PUT", {"lines": [{"code": "bad", "quantity": 1}]})[0].startswith("400")
    assert call("/api/log", body={"event": "search_submit", "detail": {"request": "x"}})[0] == "200 OK"
    assert call("/api/log", body={"event": "hack"})[0].startswith("400")
    assert call("/api/cart", "DELETE")[0].startswith("405")


def test_addons_route_uses_rules_and_constraints(monkeypatch):
    from clickless_mining.addons import OfflineKG, RuleIndex
    rules = [{"antecedents": ["plant_milk"], "consequent": "breakfast_cereal",
              "support": .1, "confidence": .5, "lift": 1.5}]
    monkeypatch.setitem(server._SHARED, "rules", RuleIndex(rules, {"breakfast_cereal": .3}))
    monkeypatch.setitem(server._SHARED, "offline_kg", OfflineKG(CATALOG))
    code, body = call("/api/addons", body={"category": "plant_milk",
                                           "constraints": {"dietary_tags": ["vegan"], "excluded_allergens": []}})
    assert code == "200 OK"
    assert body["addons"][0]["product"]["code"] == "002"
    assert "plant_milk -> breakfast_cereal" in body["addons"][0]["explanation"]
    code, body = call("/api/addons", body={"category": "plant_milk",
                                           "constraints": {"excluded_allergens": ["gluten"]}})
    assert body["addons"] == []                      # only cereal contains gluten
    assert call("/api/addons", body={"category": "snacks"})[0].startswith("400")   # group, not leaf
    server._SHARED.pop("rules", None)
    server._SHARED.pop("offline_kg", None)


def test_offline_kg_search(monkeypatch):
    monkeypatch.setenv("CLICKLESS_KG", "offline")
    with patch("clickless_llm.config_c.run_config_c", return_value={"status": "no_match"}) as run:
        assert call("/api/search", body={"request": "oat milk"})[0] == "200 OK"
    assert "kg" in run.call_args.kwargs


def test_agent_clicks_model():
    ev = [{"event": e} for e in ("search_submit", "add_to_cart", "search_submit", "search_submit",
                                 "add_to_cart", "quantity_change")]
    tasks = agent_clicks(ev)
    assert tasks[0]["clicks"] == 2                       # 1 submission + 1 approval (= 2 + e, e = 0)
    assert tasks[1] == {"submissions": 1, "approvals": 1, "corrections": 2, "clicks": 4}
