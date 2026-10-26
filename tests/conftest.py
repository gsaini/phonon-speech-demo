"""A fake Ollama: /api/version, /api/show and /v1/audio/speech, enough to test everything offline."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

VOICES = ["default", "Zoey", "Captain Silas Blackwater"]
BYTES_PER_CHAR = 960  # 20 ms of 24 kHz 16-bit mono per character: enough to make durations predictable


class FakeOllama:
    def __init__(self) -> None:
        self.version = "0.40.11"
        self.capabilities = ["speech"]
        self.requests: list[dict[str, Any]] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _json(self, status: int, payload: Any) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/api/version":
                    self._json(200, {"version": fake.version})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/show":
                    if body["model"] != "phonon":
                        self._json(404, {"error": f"model '{body['model']}' not found"})
                    else:
                        self._json(200, {"capabilities": fake.capabilities, "voices": VOICES})
                    return
                if self.path != "/v1/audio/speech":
                    self._json(404, {"error": "not found"})
                    return
                fake.requests.append(body)
                voice = body.get("voice", "default")
                if voice.lower() not in {v.lower() for v in VOICES}:
                    msg = f'voice "{voice}" not found; available voices: {", ".join(VOICES)}'
                    self._json(400, {"error": {"message": msg}})
                    return
                if len(body["input"]) > 4096:
                    self._json(400, {"error": {"message": "input is longer than 4096 characters"}})
                    return
                audio = b"\x01\x00" * (BYTES_PER_CHAR // 2) * len(body["input"])
                self.send_response(200)
                self.send_header("Content-Type", "audio/pcm")
                self.end_headers()
                for i in range(0, len(audio), 8192):  # streamed, like the real thing
                    self.wfile.write(audio[i : i + 8192])

            def log_message(self, format: str, *args: Any) -> None:
                pass

        return Handler


@pytest.fixture
def fake() -> Iterator[FakeOllama]:
    server = FakeOllama()
    yield server
    server.server.shutdown()
