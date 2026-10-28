"""Your own voices: Phonon voice files added to a local copy of the model, as `phonon-mine`.

A Phonon voice is not a recording. It is a speaker embedding: one float32 tensor `speaker_wavs` of
shape 1×512×125, made by Gradium's speaker encoder from about ten seconds of speech. That encoder
isn't in the open weights, so a recording can't become a voice on this machine. Make the voice in
Gradium Studio (clone or design it there) and export it as .safetensors, or take one from the
public pack at huggingface.co/gradium/phonon-voices.

Ollama can't add a voice to a model it already has (a safetensors import takes no FROM), so
build() imports the whole open checkpoint again: gradium/phonon-31m-en, the same checkpoint as
Ollama's `phonon` tag, with the built-in English voices and yours beside it. The checkpoint (about
100 MB) and the built-in voices are downloaded once, into ~/.phonon-demo/checkpoint/.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from phonon_demo.client import CUSTOM_MODEL, Phonon, PhononError

HF = "https://huggingface.co"
CHECKPOINT_REPO = "gradium/phonon-31m-en"  # Apache 2.0
VOICE_PACK_REPO = "gradium/phonon-voices"  # Apache 2.0
CHECKPOINT_FILES = (
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "default-voice.safetensors",
    "model.safetensors",
)
VOICE_SHAPE = [1, 512, 125]  # 125 frames at 12.5 Hz: ten seconds of the speaker
VOICE_SIG = "ba7a5a45"  # the speaker encoder; config.json lists it under embeddings_ext
NAME = re.compile(r"\w[\w .'-]{0,63}")  # \w includes accented letters: "Noémie" is fine


def home() -> Path:
    return Path(os.environ.get("PHONON_DEMO_HOME") or Path.home() / ".phonon-demo")


def voices_dir() -> Path:
    return home() / "voices"


def mine() -> list[str]:
    """The names of your voices, from the files in ~/.phonon-demo/voices/."""
    return sorted((p.stem for p in voices_dir().glob("*.safetensors")), key=str.lower)


def find(name: str) -> str | None:
    """Voice names are case-insensitive, as they are in Ollama."""
    return next((v for v in mine() if v.lower() == name.strip().lower()), None)


def check_voice(path: Path) -> None:
    """Fail unless `path` holds exactly one Phonon speaker embedding."""
    try:
        with path.open("rb") as f:
            (size,) = struct.unpack("<Q", f.read(8))
            if size > 100_000:
                raise ValueError("header too large")
            header = json.loads(f.read(size))
    except (OSError, ValueError, struct.error) as exc:
        raise PhononError(f"{path} is not a safetensors file ({exc}).") from exc
    meta = header.pop("__metadata__", None) or {}
    tensor = header.get("speaker_wavs") or {}
    if set(header) != {"speaker_wavs"} or tensor.get("dtype") != "F32" or tensor.get("shape") != VOICE_SHAPE:
        found = ", ".join(f"{k} {v.get('dtype')} {v.get('shape')}" for k, v in list(header.items())[:2])
        if len(header) > 2:
            found = f"{len(header)} tensors ({found}, …)"
        raise PhononError(
            f"{path.name} is not a Phonon voice: it should hold one float32 tensor speaker_wavs "
            f"of shape 1×512×125, and holds {found or 'no tensors'}."
        )
    if meta.get("sig", VOICE_SIG) != VOICE_SIG:
        raise PhononError(
            f"{path.name} comes from a different speaker encoder ({meta['sig']}); Phonon reads {VOICE_SIG} voices."
        )


def add(source: Path, name: str, replace: bool = False) -> Path:
    """Check a voice file and keep a copy of it under `name`."""
    name = name.strip()
    if not NAME.fullmatch(name) or name.lower() == "default":
        raise PhononError(
            f"{name!r} can't be a voice name: use letters, digits, spaces, - . or ', up to 64 characters, "
            "and not 'default'."
        )
    check_voice(source)
    existing = find(name)
    if existing and not replace:
        raise PhononError(f"You already have a voice called {existing}. Add --replace to swap it.")
    if existing:
        (voices_dir() / f"{existing}.safetensors").unlink()
    target = voices_dir() / f"{name}.safetensors"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def remove(name: str) -> str:
    found = find(name)
    if not found:
        yours = ", ".join(mine()) or "none yet"
        raise PhononError(f"You have no voice called {name!r}. Yours: {yours}.")
    (voices_dir() / f"{found}.safetensors").unlink()
    return found


def _download(url: str, target: Path) -> None:
    partial = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise PhononError(f"Couldn't download {url} ({getattr(exc, 'reason', exc)}).") from exc
    partial.replace(target)


def checkpoint(report: Callable[[str], None] = lambda _: None) -> Path:
    """The open checkpoint and its built-in English voices, downloaded on first use."""
    folder = home() / "checkpoint" / CHECKPOINT_REPO.split("/")[1]
    if (folder / ".complete").exists():
        return folder
    for name in CHECKPOINT_FILES:
        if not (folder / name).exists():
            report(f"Downloading {CHECKPOINT_REPO}/{name}")
            _download(f"{HF}/{CHECKPOINT_REPO}/resolve/main/{name}", folder / name)
    index = folder / "voices.jsonl"
    if not index.exists():
        _download(f"{HF}/{VOICE_PACK_REPO}/resolve/main/voices.jsonl", index)
    entries = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
    english = [e for e in entries if e.get("language") == "en"]
    report(f"Downloading the {len(english)} built-in English voices from {VOICE_PACK_REPO}")
    for entry in english:
        target = folder / "builtin-voices" / f"{entry['name']}.safetensors"
        if not target.exists():
            _download(f"{HF}/{VOICE_PACK_REPO}/resolve/main/{entry['filename']}", target)
    (folder / ".complete").touch()
    return folder


def build(phonon: Phonon, report: Callable[[str], None] = lambda _: None) -> list[str]:
    """Import the checkpoint, the built-in voices and yours into Ollama as `phonon-mine`."""
    names = mine()
    if not names:
        raise PhononError("You have no voices to build with. Add one with `phonon-demo add-voice FILE --name NAME`.")
    base = checkpoint(report)
    files = {name: base / name for name in CHECKPOINT_FILES}
    yours = {n.lower() for n in names}
    for path in sorted((base / "builtin-voices").glob("*.safetensors")):
        if path.stem.lower() not in yours:  # a voice of yours named like a built-in one replaces it
            files[f"voices/{path.name}"] = path
    for name in names:
        files[f"voices/{name}.safetensors"] = voices_dir() / f"{name}.safetensors"
    report(f"Importing into Ollama as {CUSTOM_MODEL}")
    phonon.create(CUSTOM_MODEL, files)
    return names
