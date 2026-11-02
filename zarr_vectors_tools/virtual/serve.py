"""Serves virtual zarr-vectors stores over HTTP, for any Neuroglancer with the
zarr-vectors data source: ``http://host:port/<file name>/|zarr-vectors:``.

Byte ranges, CORS and HTML folder listings (which Neuroglancer's http store
reads, for object attribute columns) are answered.
"""

from __future__ import annotations

import html
import re
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import open_virtual


def make_handler(stores: dict):
    class Handler(BaseHTTPRequestHandler):
        def _headers(self, status: int, length: int, content_type: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Range")
            self.end_headers()

        def do_HEAD(self):
            self._answer(head=True)

        def do_GET(self):
            self._answer(head=False)

        def _answer(self, head: bool) -> None:
            path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path).lstrip("/")
            name, _, key = path.partition("/")
            store = stores.get(name)
            if store is None:
                if path == "":
                    body = "".join(f'<li><a href="{html.escape(n)}/">{html.escape(n)}/</a></li>' for n in sorted(stores)).encode()
                    self._headers(200, len(body), "text/html")
                    if not head:
                        self.wfile.write(body)
                    return
                self._headers(404, 0, "text/plain")
                return
            if key == "" or key.endswith("/"):
                names = store.list(key)
                if names is None:
                    self._headers(404, 0, "text/plain")
                    return
                body = ("<!DOCTYPE html><html><body><ul>" + "".join(f'<li><a href="{html.escape(urllib.parse.quote(n))}">{html.escape(n)}</a></li>' for n in names) + "</ul></body></html>").encode()
                self._headers(200, len(body), "text/html")
                if not head:
                    self.wfile.write(body)
                return
            data = store.get(key)
            if data is None:
                self._headers(404, 0, "text/plain")
                return
            m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", "") or "")
            if m:
                size = len(data)
                if m.group(1) == "":
                    start, end = max(0, size - int(m.group(2))), size
                else:
                    start = int(m.group(1))
                    end = min(size, int(m.group(2)) + 1) if m.group(2) else size
                if start >= size and size > 0:
                    self._headers(416, 0, "text/plain", {"Content-Range": f"bytes */{size}"})
                    return
                part = data[start:end]
                self._headers(206, len(part), "application/octet-stream", {"Content-Range": f"bytes {start}-{max(start, end - 1)}/{size}"})
                if not head:
                    self.wfile.write(part)
                return
            self._headers(200, len(data), "application/octet-stream")
            if not head:
                self.wfile.write(data)

        def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler's name
            pass

    return Handler


def serve(paths: list[str | Path], host: str = "127.0.0.1", port: int = 8000) -> None:
    """Opens each file (reading it once) and serves them until interrupted."""
    stores = {}
    for p in paths:
        path = Path(p)
        stores[path.name] = open_virtual(path)
        print(f"{path.name}: {stores[path.name].count:,} streamlines")
    server = ThreadingHTTPServer((host, port), make_handler(stores))
    for name in stores:
        print(f"  http://{host}:{server.server_port}/{urllib.parse.quote(name)}/|zarr-vectors:")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
