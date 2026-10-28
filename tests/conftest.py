"""A fake Ollama: version, show, speech, and the blob and create calls, enough to test everything offline."""

from __future__ import annotations

import hashlib
import json
import struct
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from phonon_demo import voices

VOICES = ["default", "Zoey", "Captain Silas Blackwater"]
BYTES_PER_CHAR = 960  # 20 ms of 24 kHz 16-bit mono per character: enough to make durations predictable


class FakeOllama:
    def __init__(self) -> None:
        self.version = "0.40.11"
        self.capabilities = ["speech"]
        self.requests: list[dict[str, Any]] = []
        self.blobs: dict[str, bytes] = {}
        self.models: dict[str, list[str]] = {"phonon": VOICES}  # model → voices
        self.created: dict[str, Any] = {}  # the last /api/create body
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

            def do_HEAD(self) -> None:
                found = self.path.removeprefix("/api/blobs/") in fake.blobs
                self.send_response(200 if found else 404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_DELETE(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if fake.models.pop(body["model"], None) is None:
                    self._json(404, {"error": f"model '{body['model']}' not found"})
                else:
                    self._json(200, {})

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path.startswith("/api/blobs/"):
                    digest = self.path.removeprefix("/api/blobs/")
                    assert digest == "sha256:" + hashlib.sha256(raw).hexdigest()
                    fake.blobs[digest] = raw
                    self._json(201, {})
                    return
                body = json.loads(raw)
                if self.path == "/api/create":
                    assert all(d in fake.blobs for d in body["files"].values())
                    fake.created = body
                    names = [f[len("voices/") : -len(".safetensors")] for f in body["files"] if f.startswith("voices/")]
                    fake.models[body["model"]] = ["default", *sorted(names)]
                    self._json(200, {"status": "success"})
                    return
                if self.path == "/api/show":
                    if body["model"] not in fake.models:
                        self._json(404, {"error": f"model '{body['model']}' not found"})
                    else:
                        self._json(200, {"capabilities": fake.capabilities, "voices": fake.models[body["model"]]})
                    return
                if self.path != "/v1/audio/speech":
                    self._json(404, {"error": "not found"})
                    return
                fake.requests.append(body)
                voice, voices = body.get("voice", "default"), fake.models[body["model"]]
                if voice.lower() not in {v.lower() for v in voices}:
                    msg = f'voice "{voice}" not found; available voices: {", ".join(voices)}'
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


@pytest.fixture(autouse=True)
def _own_home(tmp_path, monkeypatch) -> None:
    """Never read or write the real ~/.phonon-demo."""
    monkeypatch.setenv("PHONON_DEMO_HOME", str(tmp_path / "home"))


@pytest.fixture
def home(tmp_path) -> Path:
    """An empty ~/.phonon-demo whose checkpoint is already "downloaded", so nothing touches the network."""
    checkpoint = tmp_path / "home" / "checkpoint" / "phonon-31m-en"
    for name in voices.CHECKPOINT_FILES:
        write_voice(checkpoint / name) if name.endswith("voice.safetensors") else _touch(checkpoint / name, name)
    for name in ("Zoey", "Captain Silas Blackwater"):
        write_voice(checkpoint / "builtin-voices" / f"{name}.safetensors")
    (checkpoint / ".complete").touch()
    return tmp_path / "home"


def write_voice(path: Path, *, shape: list[int] | None = None, sig: str = voices.VOICE_SIG, seed: int = 0) -> Path:
    """A Phonon voice file: one float32 tensor speaker_wavs, 1×512×125, in safetensors format."""
    shape = shape or voices.VOICE_SHAPE
    size = 4 * shape[0] * shape[1] * shape[2]
    header = {"speaker_wavs": {"dtype": "F32", "shape": shape, "data_offsets": [0, size]}, "__metadata__": {"sig": sig}}
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes([seed % 256]) * size)
    return path


def _touch(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
