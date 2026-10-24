#!/usr/bin/env python3
"""Static server for the demo with caching disabled.

python3 -m http.server does not send Cache-Control, so Chrome caches ES modules
heuristically and keeps running an old copy after an edit. This sends
`no-store` for every response.

    python3 serve.py [port]
"""

import http.server
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".js": "text/javascript",
        ".onnx": "application/octet-stream",
        ".json": "application/json",
        ".wasm": "application/wasm",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    with Server(("127.0.0.1", PORT), Handler) as httpd:
        print(f"Light Translate -> http://127.0.0.1:{PORT}  (caching disabled)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
