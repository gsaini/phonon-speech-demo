"""How fast Phonon speaks on this machine: time to first audio, and speed relative to real time.

Cold is the first request after the model is unloaded (`ollama stop`), so it includes loading the
weights. Warm runs follow, over three lengths of text. Each warm number is the median of `runs`.
"""

from __future__ import annotations

import shutil
import statistics
import subprocess
from typing import Any

from phonon_demo.client import Phonon, Speech

TEXTS = {
    "one sentence": "The build finished in four minutes, and every test passed.",
    "one paragraph": (
        "Good morning. Here is a short update before your first meeting. Checkout recovered at 14:32 "
        "after the rollback, and the payments team is writing up what happened. Two customers were "
        "charged twice; both refunds of $49.99 went out this morning. Nothing else needs you today."
    ),
    "a page": (
        "Sound travels through a solid as a wave of tiny vibrations, passed from one atom to the next. "
        "Physicists call the smallest unit of that vibration a phonon, the way they call the smallest "
        "unit of light a photon. "
    )
    * 12,
}


def unload(model: str) -> bool:
    """Ask Ollama to unload the model so the next request is cold. False if the CLI isn't here."""
    ollama = shutil.which("ollama")
    if not ollama:
        return False
    return subprocess.run([ollama, "stop", model], capture_output=True, check=False).returncode == 0


def row(label: str, speech: list[Speech], chars: int) -> dict[str, Any]:
    return {
        "text": label,
        "chars": chars,
        "audio_s": statistics.median(s.seconds for s in speech),
        "first_audio_ms": statistics.median(s.first_audio_ms for s in speech),
        "total_ms": statistics.median(s.total_ms for s in speech),
        "realtime": statistics.median(s.realtime for s in speech),
    }


def run(phonon: Phonon, voice: str | None = None, runs: int = 3) -> list[dict[str, Any]]:
    rows = []
    cold_text = TEXTS["one sentence"]
    if unload(phonon.model):
        rows.append(row("one sentence, cold", [phonon.speak(cold_text, voice)], len(cold_text)))
    else:
        phonon.speak(cold_text, voice)  # load it, so the warm rows are warm
    for label, text in TEXTS.items():
        rows.append(row(label, [phonon.speak(text, voice) for _ in range(runs)], len(text)))
    return rows


def markdown(rows: list[dict[str, Any]], model: str, voice: str | None) -> str:
    lines = [
        f"`{model}`, voice `{voice or 'default'}`. Warm rows are the median of the runs.",
        "",
        "| text | characters | audio | first audio | done after | faster than real time |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['text']} | {r['chars']:,} | {r['audio_s']:.1f} s | {r['first_audio_ms']:,.0f} ms | "
            f"{r['total_ms']:,.0f} ms | {r['realtime']:.0f}× |"
        )
    return "\n".join(lines)
