"""WAV files and playback for Phonon's audio: 16-bit little-endian PCM, mono, 24 kHz."""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from phonon_demo.client import SAMPLE_RATE, SAMPLE_WIDTH


def wav_bytes(pcm: bytes) -> bytes:
    """A complete WAV file with correct sizes in its header."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(SAMPLE_WIDTH)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm)
    return buffer.getvalue()


def save(pcm: bytes, path: Path) -> Path:
    """Write a .wav, or an .mp3 through ffmpeg when the path asks for one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".mp3":
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("Saving .mp3 needs ffmpeg; save as .wav instead.")
        subprocess.run(
            [ffmpeg, "-loglevel", "error", "-y", "-i", "pipe:0", "-codec:a", "libmp3lame", "-q:a", "4", str(path)],
            input=wav_bytes(pcm),
            check=True,
        )
    else:
        path.write_bytes(wav_bytes(pcm))
    return path


class Player:
    """Plays PCM as it arrives through ffplay when it is installed; otherwise all at once at the end.

    Use as a context manager: feed() each chunk, and leaving the block waits for playback to finish.
    """

    def __init__(self) -> None:
        self.ffplay = shutil.which("ffplay")
        self.buffer = bytearray()
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def streaming(self) -> bool:
        return self.ffplay is not None

    def __enter__(self) -> Player:
        if self.ffplay:
            self.process = subprocess.Popen(
                [self.ffplay, "-loglevel", "error", "-nodisp", "-autoexit",
                 "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ch_layout", "mono", "-i", "pipe:0"],
                stdin=subprocess.PIPE,
            )  # fmt: skip
        return self

    def feed(self, chunk: bytes) -> None:
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(chunk)
                self.process.stdin.flush()
            except BrokenPipeError:  # the player was closed early
                self.process = None
        else:
            self.buffer += chunk

    def __exit__(self, *exc: object) -> None:
        if self.process and self.process.stdin:
            self.process.stdin.close()
            self.process.wait()
            return
        if self.buffer:
            play_file(self.buffer)


def play_file(pcm: bytes) -> None:
    """Play a finished clip with whatever the platform has: afplay on macOS, else ffplay or aplay."""
    for player in (["afplay"], ["ffplay", "-loglevel", "error", "-nodisp", "-autoexit"], ["aplay", "-q"]):
        if shutil.which(player[0]):
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                tmp.write(wav_bytes(pcm))
                tmp.flush()
                subprocess.run([*player, tmp.name], check=False)
            return
    raise RuntimeError("No audio player found (afplay, ffplay or aplay). Use --out to save the audio instead.")
