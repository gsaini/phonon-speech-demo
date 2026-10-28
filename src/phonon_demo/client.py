"""A small client for Ollama's speech endpoint, standard library only.

Ollama serves Phonon through its OpenAI-compatible `POST /v1/audio/speech`: JSON in, audio out,
streamed while it is generated. This client always asks for `pcm` (raw 16-bit little-endian mono
at 24 kHz) and writes its own WAV header, because the `wav` stream's header is written before the
length is known and carries placeholder sizes that some players and editors reject.

Inputs longer than the endpoint's 4,096-character limit are split at paragraph and sentence
breaks (text.py) and spoken as consecutive requests; the audio is joined in order.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phonon_demo.text import MAX_INPUT, chunks

SAMPLE_RATE = 24_000
SAMPLE_WIDTH = 2  # bytes: 16-bit
MIN_OLLAMA = (0, 40, 9)
DEFAULT_MODEL = "phonon"
CUSTOM_MODEL = "phonon-mine"  # the checkpoint again, with your voices added (voices.py)
READ_SIZE = 4096


class PhononError(RuntimeError):
    """Something a person can fix: Ollama not running or too old, model not pulled, unknown voice."""


def base_url(host: str | None = None) -> str:
    """`--host`, then $OLLAMA_HOST (which Ollama itself writes without a scheme), then localhost."""
    host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    if host.startswith("0.0.0.0"):  # a bind address, not one to connect to
        host = host.replace("0.0.0.0", "localhost", 1)
    return host if "://" in host else f"http://{host}"


def parse_version(text: str) -> tuple[int, ...]:
    parts = []
    for piece in text.split("-")[0].split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


@dataclass(frozen=True)
class Speech:
    pcm: bytes
    first_audio_ms: float  # request sent → first audio bytes back
    total_ms: float  # request sent → last audio bytes back
    requests: int = 1

    @property
    def seconds(self) -> float:
        return len(self.pcm) / SAMPLE_WIDTH / SAMPLE_RATE

    @property
    def realtime(self) -> float:
        """How many times faster than real time: seconds of audio per second of waiting."""
        return self.seconds / (self.total_ms / 1000) if self.total_ms else 0.0


class Phonon:
    def __init__(self, host: str | None = None, model: str = DEFAULT_MODEL, timeout: float = 120.0) -> None:
        self.url = base_url(host)
        self.model = model
        self.timeout = timeout

    # --- plumbing ------------------------------------------------------------------------------

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None, data: bytes | None = None) -> Any:
        if body is not None:
            data = json.dumps(body).encode()
        request = urllib.request.Request(
            self.url + path, data=data, method=method, headers={"content-type": "application/json"}
        )
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise PhononError(self._explain(exc)) from exc
        except urllib.error.URLError as exc:
            raise PhononError(f"No Ollama at {self.url} ({exc.reason}). Start it with `ollama serve`.") from exc

    def _explain(self, exc: urllib.error.HTTPError) -> str:
        raw = exc.read().decode(errors="replace")
        try:
            payload = json.loads(raw)
            error = payload.get("error", payload)
            message = error.get("message", raw) if isinstance(error, dict) else str(error)
        except ValueError:
            message = raw or exc.reason
        if exc.code == 404 and "not found" in message.lower() and self.model == CUSTOM_MODEL:
            return f"{message}. Rebuild it from your voice files with `phonon-demo voices --build`."
        if exc.code == 404 and "not found" in message.lower():
            return f"{message}. Pull it with `ollama pull {self.model}`."
        return f"Ollama answered {exc.code}: {message}"

    def _json(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        with self._request(method, path, body) as response:
            return json.load(response)

    # --- what the model can do -----------------------------------------------------------------

    def version(self) -> str:
        return str(self._json("GET", "/api/version")["version"])

    def show(self) -> dict[str, Any]:
        return self._json("POST", "/api/show", {"model": self.model})

    def check(self) -> list[str]:
        """Fail early with a fix, or return the model's voices."""
        version = self.version()
        if parse_version(version) < MIN_OLLAMA:
            need = ".".join(map(str, MIN_OLLAMA))
            raise PhononError(f"Ollama {version} is too old for speech models; Phonon needs {need} or later.")
        info = self.show()
        if "speech" not in (info.get("capabilities") or []):
            raise PhononError(f"{self.model!r} is not a speech model (capabilities: {info.get('capabilities')}).")
        return list(info.get("voices") or [])

    def voices(self) -> list[str]:
        return list(self.show().get("voices") or [])

    # --- making models --------------------------------------------------------------------------

    def upload(self, path: Path) -> str:
        """Copy a file into Ollama's blob store unless it is there already; return its digest."""
        data = path.read_bytes()
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        try:
            self._request("HEAD", f"/api/blobs/{digest}").close()
        except PhononError:
            self._request("POST", f"/api/blobs/{digest}", data=data).close()
        return digest

    def create(self, model: str, files: dict[str, Path]) -> None:
        """Import a checkpoint as `model`; `files` maps each path inside the checkpoint to a local file."""
        digests = {name: self.upload(path) for name, path in files.items()}
        self._json("POST", "/api/create", {"model": model, "files": digests, "stream": False})

    def delete(self, model: str) -> None:
        self._request("DELETE", "/api/delete", {"model": model}).close()

    # --- speaking ------------------------------------------------------------------------------

    def stream(self, text: str, voice: str | None = None) -> Iterator[bytes]:
        """Raw PCM for one request (≤ 4,096 characters), yielded as it arrives."""
        if len(text) > MAX_INPUT:
            raise ValueError(f"one request takes at most {MAX_INPUT} characters; use speak() to split longer text")
        body = {"model": self.model, "input": text, "response_format": "pcm"}
        if voice:
            body["voice"] = voice
        with self._request("POST", "/v1/audio/speech", body) as response:
            while chunk := response.read(READ_SIZE):
                yield chunk

    def speak(self, text: str, voice: str | None = None, on_audio: Callable[[bytes], None] | None = None) -> Speech:
        """All of `text`, split into requests as needed. `on_audio` gets each piece as it arrives."""
        pieces = chunks(text)
        if not pieces:
            raise PhononError("Nothing to say: the text is empty once formatting is removed.")
        audio, first = bytearray(), None
        started = time.perf_counter()
        for piece in pieces:
            for chunk in self.stream(piece, voice):
                if first is None:
                    first = time.perf_counter() - started
                audio += chunk
                if on_audio:
                    on_audio(chunk)
        total = time.perf_counter() - started
        return Speech(bytes(audio), (first or total) * 1000, total * 1000, len(pieces))
