"""Authenticated local HTTP server for an application exposing handle(path, body) (AML Add/Search shapes)."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import sqlite3

from .errors import (ConfigurationError, ConflictError, MemoryErrorBase,
                     ModelError, NotFoundError, SearchLoopError)
from .model import encode
from .schema import require
from .collections.store import CollectionStore


def make_server(application, host: str, port: int, token: str) -> ThreadingHTTPServer:
    if not token:
        raise ConfigurationError("启动接口前请配置 MEMORY_SERVICE_TOKEN")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            # BaseHTTPRequestHandler logs URLs; payloads and credentials should
            # never appear in an access log. Callers can retain explicit traces.
            pass

        def send_json(self, status: int, value: dict):
            raw = encode(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                # A health response checks local storage, not model quality.
                try:
                    with CollectionStore(application.database) as store:
                        store.db.execute("SELECT 1").fetchone()
                    self.send_json(200, {"status": "ok"})
                except sqlite3.Error:
                    self.send_json(503, {"error": "本地存储不可用"})
                except ConfigurationError as error:
                    self.send_json(503, {"error": str(error), "type": "ConfigurationError"})
            else:
                self.send_json(404, {"error": "没有这个接口"})

        def do_POST(self):
            authorization = self.headers.get("Authorization", "")
            candidates = ["Bearer " + token, "Token " + token]
            authenticated = any(hmac.compare_digest(authorization.encode(), candidate.encode()) for candidate in candidates)
            authenticated |= hmac.compare_digest(self.headers.get("X-Api-Key", "").encode(), token.encode())
            if not authenticated:
                self.send_json(401, {"error": "接口密钥无效"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                require(length > 0, "请求需要非空 JSON 内容")
                body = json.loads(self.rfile.read(length))
                result = application.handle(self.path, body)
                self.send_json(200, result)
            except (ValueError, UnicodeError) as error:
                self.send_json(400, {"error": "请求不是有效 JSON"})
            except MemoryErrorBase as error:
                status = 409 if isinstance(error, ConflictError) else 404 if isinstance(error, NotFoundError) else 503 if isinstance(error, ConfigurationError) else 502 if isinstance(error, (ModelError, SearchLoopError)) else 400
                self.send_json(status, {"error": str(error), "type": type(error).__name__})
            except sqlite3.Error:
                self.send_json(503, {"error": "本地存储请求失败；未返回替代结果"})
            except Exception:
                self.send_json(500, {"error": "内部请求失败；未返回替代结果"})

    return ThreadingHTTPServer((host, port), Handler)
