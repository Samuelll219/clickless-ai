"""Serve the local shopping UI with python -m clickless_ui."""

from wsgiref.simple_server import make_server

from .server import application


if __name__ == "__main__":
    with make_server("127.0.0.1", 8000, application) as server:
        print("ClickLess AI is running at http://127.0.0.1:8000", flush=True)
        server.serve_forever()
