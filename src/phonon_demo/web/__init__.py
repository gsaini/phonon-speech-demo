"""`phonon-demo serve`: a local web page for Phonon. Standard library only; binds to 127.0.0.1."""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from phonon_demo.audio import wav_bytes
from phonon_demo.client import Phonon, PhononError
from phonon_demo.text import speakable

PAGE = Path(__file__).with_name("index.html")
MAX_TEXT = 20_000  # characters per click; longer text is split into several requests anyway


def handler_for(phonon: Phonon) -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()  # one speech request at a time: the model is a single local process
    voices: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:
            path = urlsplit(self.path).path  # "/?text=…&voice=…" is still the page
            if path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/info":
                try:
                    if not voices:
                        voices.extend(phonon.voices())
                    self._json(200, {"model": phonon.model, "voices": voices})
                except PhononError as exc:
                    self._json(502, {"error": str(exc)})
            else:
                self._json(404, {"error": "Not found."})

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/speak":
                self._json(404, {"error": "Not found."})
                return
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 4 * MAX_TEXT)
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._json(400, {"error": "Send a JSON object."})
                return
            if not isinstance(body, dict):
                self._json(400, {"error": "Send a JSON object."})
                return
            text = str(body.get("text", "")).strip()[:MAX_TEXT]
            if body.get("markdown"):
                text = speakable(text)
            if not text:
                self._json(400, {"error": "Type something to say first."})
                return
            voice = str(body["voice"]) if body.get("voice") else None
            try:
                with lock:
                    speech = phonon.speak(text, voice)
            except PhononError as exc:
                self._json(502, {"error": str(exc)})
                return
            self._send(
                200,
                wav_bytes(speech.pcm),
                "audio/wav",
                {
                    "X-Audio-Seconds": f"{speech.seconds:.3f}",
                    "X-First-Audio-Ms": f"{speech.first_audio_ms:.1f}",
                    "X-Total-Ms": f"{speech.total_ms:.1f}",
                    "X-Requests": str(speech.requests),
                },
            )

        def log_message(self, format: str, *args: Any) -> None:  # keep the terminal quiet
            pass

    return Handler


def serve(phonon: Phonon, *, port: int = 8766, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(phonon))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Phonon page at {url}  (model {phonon.model}). Ctrl+C to stop.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
