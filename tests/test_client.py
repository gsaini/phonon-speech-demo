"""The client, the WAV writer and the CLI, against the fake Ollama in conftest.py."""

import io
import json
import urllib.request
import wave

import pytest

from phonon_demo.audio import save, wav_bytes
from phonon_demo.cli import main
from phonon_demo.client import Phonon, PhononError, base_url, parse_version
from phonon_demo.web import handler_for

from .conftest import BYTES_PER_CHAR, VOICES


def test_base_url_accepts_what_ollama_host_holds(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert base_url() == "http://localhost:11434"
    assert base_url("127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert base_url("0.0.0.0:11434") == "http://localhost:11434"
    monkeypatch.setenv("OLLAMA_HOST", "https://gpu-box:443/")
    assert base_url() == "https://gpu-box:443"


def test_versions_compare_as_numbers():
    assert parse_version("0.40.11") > parse_version("0.40.9") > parse_version("0.15.1")
    assert parse_version("0.41.0-rc1") == (0, 41, 0)


def test_check_returns_voices(fake):
    assert Phonon(fake.url).check() == VOICES


def test_check_explains_what_to_fix(fake):
    fake.version = "0.15.1"
    with pytest.raises(PhononError, match="too old"):
        Phonon(fake.url).check()
    fake.version, fake.capabilities = "0.40.11", ["completion"]
    with pytest.raises(PhononError, match="not a speech model"):
        Phonon(fake.url).check()
    with pytest.raises(PhononError, match="ollama pull nope"):
        Phonon(fake.url, model="nope").check()
    with pytest.raises(PhononError, match="No Ollama at"):
        Phonon("http://127.0.0.1:9").check()


def test_speak_asks_for_pcm_and_times_it(fake):
    speech = Phonon(fake.url).speak("Hello there.", voice="Zoey")
    assert fake.requests == [{"model": "phonon", "input": "Hello there.", "response_format": "pcm", "voice": "Zoey"}]
    assert len(speech.pcm) == BYTES_PER_CHAR * len("Hello there.")
    assert speech.seconds == pytest.approx(0.24)
    assert 0 < speech.first_audio_ms <= speech.total_ms and speech.realtime > 0 and speech.requests == 1


def test_long_text_becomes_several_requests_in_order(fake):
    text = "One short sentence here. " * 400  # about 10,000 characters
    received: list[bytes] = []
    speech = Phonon(fake.url).speak(text, on_audio=received.append)
    assert speech.requests == len(fake.requests) >= 3
    assert all(len(r["input"]) <= 4096 for r in fake.requests)
    assert b"".join(received) == speech.pcm


def test_unknown_voice_lists_the_real_ones(fake):
    with pytest.raises(PhononError, match="available voices: default, Zoey"):
        Phonon(fake.url).speak("Hi.", voice="Nobody")


def test_wav_header_is_complete():
    pcm = b"\x00\x10" * 24_000  # one second
    with wave.open(io.BytesIO(wav_bytes(pcm))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 24_000, 24_000)


def test_cli_say_saves_a_wav_without_playing(fake, tmp_path, capsys):
    out = tmp_path / "hello.wav"
    assert main(["--host", fake.url, "say", "Hello there.", "--voice", "zoey", "--out", str(out), "--no-play"]) == 0
    with wave.open(str(out)) as w:
        assert w.getnframes() == BYTES_PER_CHAR * len("Hello there.") // 2
    assert "real time" in capsys.readouterr().err


def test_cli_read_cleans_markdown_first(fake, tmp_path, capsys):
    notes = tmp_path / "notes.md"
    notes.write_text("# Title\n\nSee [the docs](https://x.y).\n\n```\ncode\n```\n", encoding="utf-8")
    assert main(["--host", fake.url, "read", str(notes), "--show"]) == 0
    assert capsys.readouterr().out.strip() == "Title.\n\nSee the docs."
    assert main(["--host", fake.url, "read", str(notes), "--no-play"]) == 0
    assert fake.requests[-1]["input"] == "Title.\n\nSee the docs."


def test_cli_reports_problems_instead_of_crashing(fake, capsys):
    fake.version = "0.15.1"
    assert main(["--host", fake.url, "voices"]) == 2
    assert "too old" in capsys.readouterr().err


def test_gallery_writes_one_file_per_voice_and_an_index(fake, tmp_path):
    out = tmp_path / "gallery"
    assert main(["--host", fake.url, "gallery", "--out", str(out), "--voices", "zoey", "default"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["README.md", "default.wav", "zoey.wav"]
    assert "| Zoey | [zoey.wav](zoey.wav) |" in (out / "README.md").read_text(encoding="utf-8")


def test_save_writes_wav_by_default(tmp_path):
    path = save(b"\x00\x00" * 2400, tmp_path / "sub" / "a.wav")
    assert path.exists() and path.read_bytes()[:4] == b"RIFF"


def test_web_page_speaks_and_returns_timings(fake):
    from http.server import ThreadingHTTPServer
    from threading import Thread

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(Phonon(fake.url)))
    Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/?text=Hello&voice=zoey&go=1") as r:  # a share link
            assert b"Say it out loud" in r.read()
        with urllib.request.urlopen(base + "/api/info") as r:
            assert json.load(r) == {"model": "phonon", "voices": VOICES, "mine": []}
        request = urllib.request.Request(
            base + "/api/speak",
            data=json.dumps({"text": "**Hello** there.", "voice": "Zoey", "markdown": True}).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request) as r:
            assert r.headers["Content-Type"] == "audio/wav"
            assert float(r.headers["X-Audio-Seconds"]) == pytest.approx(0.24)
            assert float(r.headers["X-First-Audio-Ms"]) <= float(r.headers["X-Total-Ms"])
            assert r.read()[:4] == b"RIFF"
        assert fake.requests[-1]["input"] == "Hello there."
        bad = urllib.request.Request(base + "/api/speak", data=b'{"text": "  "}')
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(bad)
        assert err.value.code == 400
    finally:
        server.shutdown()
