"""Adding your own voices: checking the file, keeping it, and importing phonon-mine into the fake Ollama."""

import pytest

from phonon_demo import voices
from phonon_demo.cli import main
from phonon_demo.client import Phonon, PhononError

from .conftest import write_voice


def test_check_voice_accepts_only_a_speaker_embedding(tmp_path):
    voices.check_voice(write_voice(tmp_path / "ok.safetensors"))
    with pytest.raises(PhononError, match="not a Phonon voice"):
        voices.check_voice(write_voice(tmp_path / "shape.safetensors", shape=[1, 512, 50]))
    with pytest.raises(PhononError, match="different speaker encoder"):
        voices.check_voice(write_voice(tmp_path / "sig.safetensors", sig="deadbeef"))
    (tmp_path / "notes.md").write_text("# not a voice\n", encoding="utf-8")
    with pytest.raises(PhononError, match="not a safetensors file"):
        voices.check_voice(tmp_path / "notes.md")


def test_add_keeps_a_copy_and_names_are_case_insensitive(home, tmp_path):
    source = write_voice(tmp_path / "export.safetensors")
    assert voices.add(source, "Noémie") == home / "voices" / "Noémie.safetensors"
    assert voices.mine() == ["Noémie"] and voices.find("NOÉMIE") == "Noémie"
    with pytest.raises(PhononError, match="already have a voice called Noémie"):
        voices.add(source, "noémie")
    voices.add(source, "noémie", replace=True)
    assert voices.mine() == ["noémie"]
    for bad in ("../evil", "default", "", "a/b"):
        with pytest.raises(PhononError, match="can't be a voice name"):
            voices.add(source, bad)
    assert voices.remove("NOÉMIE") == "noémie" and voices.mine() == []
    with pytest.raises(PhononError, match="no voice called"):
        voices.remove("Nobody")


def test_build_imports_the_checkpoint_with_built_in_voices_and_yours(fake, home, tmp_path):
    voices.add(write_voice(tmp_path / "a.safetensors", seed=1), "Me")
    voices.add(write_voice(tmp_path / "b.safetensors", seed=2), "zoey")  # replaces the built-in Zoey
    voices.build(Phonon(fake.url))
    files = fake.created["files"]
    assert fake.created["model"] == "phonon-mine"
    assert set(voices.CHECKPOINT_FILES) < set(files)
    assert sorted(f for f in files if f.startswith("voices/")) == [
        "voices/Captain Silas Blackwater.safetensors",
        "voices/Me.safetensors",
        "voices/zoey.safetensors",
    ]
    assert fake.blobs[files["voices/zoey.safetensors"]] == (home / "voices" / "zoey.safetensors").read_bytes()


def test_cli_add_voice_then_speak_with_it_then_remove_it(fake, home, tmp_path, capsys):
    source = write_voice(tmp_path / "export.safetensors")
    assert main(["--host", fake.url, "add-voice", str(source), "--name", "Me", "--yes"]) == 0
    assert "Added Me to phonon-mine" in capsys.readouterr().err
    # with a voice of your own, commands use phonon-mine unless --model says otherwise
    assert main(["--host", fake.url, "say", "Hello.", "--voice", "me", "--no-play"]) == 0
    assert fake.requests[-1]["model"] == "phonon-mine"
    assert main(["--host", fake.url, "voices"]) == 0
    assert "yours: Me" in capsys.readouterr().err
    assert main(["--host", fake.url, "remove-voice", "me"]) == 0
    assert "phonon-mine" not in fake.models
    assert main(["--host", fake.url, "say", "Hello.", "--no-play"]) == 0
    assert fake.requests[-1]["model"] == "phonon"


def test_cli_add_voice_needs_consent_and_a_real_voice(fake, home, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    source = write_voice(tmp_path / "export.safetensors")
    assert main(["--host", fake.url, "add-voice", str(source), "--name", "Me"]) == 2
    assert "--yes" in capsys.readouterr().err and voices.mine() == []
    assert main(["--host", fake.url, "add-voice", str(tmp_path / "missing.safetensors"), "--name", "Me", "--yes"]) == 2
    assert "not a safetensors file" in capsys.readouterr().err


def test_a_missing_phonon_mine_says_how_to_rebuild_it(fake, home, tmp_path, capsys):
    voices.add(write_voice(tmp_path / "a.safetensors"), "Me")  # saved, but never imported
    assert main(["--host", fake.url, "voices"]) == 2
    assert "phonon-demo voices --build" in capsys.readouterr().err
    assert main(["--host", fake.url, "voices", "--build"]) == 0
    assert "Me" in capsys.readouterr().out
