"""Serve the local shopping UI with python -m clickless_ui."""

from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, make_server

from .server import application


class ThreadedWSGIServer(ThreadingMixIn, WSGIServer):
    """Keep assets and busy responses available while Config C is running."""

    daemon_threads = True


if __name__ == "__main__":
    with make_server("127.0.0.1", 8000, application, server_class=ThreadedWSGIServer) as server:
        print("ClickLess AI is running at http://127.0.0.1:8000", flush=True)
        server.serve_forever()
