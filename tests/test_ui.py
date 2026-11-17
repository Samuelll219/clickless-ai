"""Smoke checks for the UI boundary with controlled Config C responses."""

import json
from io import BytesIO
from unittest.mock import patch

import pytest
from neo4j.exceptions import ServiceUnavailable

from clickless_llm.pipeline import PipelineError
from clickless_llm.validator import ValidationError
from clickless_ui.server import application


def request(path="/api/search", method="POST", body=b'{"request":"two bars"}'):
    status = []
    payload = b"".join(application({
        "PATH_INFO": path, "REQUEST_METHOD": method,
        "CONTENT_LENGTH": str(len(body)), "wsgi.input": BytesIO(body),
    }, lambda code, headers: status.append(code)))
    return status[0], payload


@pytest.mark.parametrize("path", ["/", "/app.js", "/styles.css"])
def test_assets(path):
    assert request(path, "GET")[0] == "200 OK"


@pytest.mark.parametrize("body", [b"{", b"[]", b'{"request":" "}',
                                  b'{"request":1}', b"x" * 8193])
def test_invalid_request(body):
    with patch("clickless_llm.config_c.run_config_c") as run:
        assert request(body=body)[0] == "400 Bad Request"
        run.assert_not_called()


@pytest.mark.parametrize("products,status", [([], "no_match"),
    ([{"code": "001234", "price": 2.15}], "matched")])
def test_response_contract(products, status):
    result = {"status": status, "products": products, "quantity": 2,
              "constraints": {"quantity": 2}}
    with patch("clickless_llm.config_c.run_config_c", return_value=result):
        code, payload = request()
    assert code == "200 OK"
    assert json.loads(payload) == result


@pytest.mark.parametrize("error,status,message", [
    (ServiceUnavailable("internal connection details"), "503", "product database"),
    (PipelineError("internal model details"), "503", "language service"),
    (RuntimeError("internal details"), "502", "could not finish"),
])
def test_service_failures(error, status, message):
    with patch("clickless_llm.config_c.run_config_c", side_effect=error):
        code, payload = request()
    assert code.startswith(status)
    assert message in json.loads(payload)["error"]
    assert b"internal" not in payload


def test_invalid_extraction_has_actionable_message():
    error = PipelineError("invalid extraction")
    error.__cause__ = ValidationError("unknown category")
    with patch("clickless_llm.config_c.run_config_c", side_effect=error):
        assert request()[0].startswith("422")


def test_busy_search_does_not_block_assets_or_start_another_pipeline():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    started, finish = Event(), Event()

    def slow_pipeline(_):
        started.set()
        assert finish.wait(5)
        return {"status": "no_match", "products": []}

    with patch("clickless_llm.config_c.run_config_c", side_effect=slow_pipeline) as run:
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(request)
            try:
                assert started.wait(5)
                code, payload = request()
                assert code.startswith("429")
                assert "retry" in json.loads(payload)["error"]
                assert request("/", "GET")[0] == "200 OK"
                assert run.call_count == 1
            finally:
                finish.set()
            assert first.result()[0] == "200 OK"
        assert request()[0] == "200 OK"


def test_failed_search_releases_capacity_for_retry():
    with patch("clickless_llm.config_c.run_config_c", side_effect=RuntimeError("failure")):
        assert request()[0].startswith("502")
    with patch("clickless_llm.config_c.run_config_c", return_value={"status": "no_match"}):
        assert request()[0] == "200 OK"
