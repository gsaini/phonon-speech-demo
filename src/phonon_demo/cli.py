"""phonon-demo: local text-to-speech with Gradium's Phonon on Ollama.

say "text"        speak it, streaming, and print how fast it was
read notes.md     a Markdown file read aloud: formatting dropped, split under the request limit
voices            the model's voices
add-voice FILE    add a Phonon voice file (.safetensors), such as your own voice exported from Gradium Studio
remove-voice NAME remove one of your voices
gallery           one sample per voice, saved to a folder
bench             time to first audio and speed versus real time, cold and warm
serve             a local web page: type, pick a voice, hear it, see the waveform
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

from phonon_demo import bench, voices
from phonon_demo.audio import Player, save
from phonon_demo.client import CUSTOM_MODEL, DEFAULT_MODEL, Phonon, PhononError, Speech
from phonon_demo.text import speakable

GALLERY_TEXT = "Hello. This is how I sound, reading the same sentence as every other voice."


def report(speech: Speech) -> str:
    split = f", {speech.requests} requests" if speech.requests > 1 else ""
    return (
        f"{speech.seconds:.1f} s of audio in {speech.total_ms / 1000:.2f} s "
        f"({speech.realtime:.0f}× real time{split}); first audio after {speech.first_audio_ms:,.0f} ms"
    )


def speak_out(phonon: Phonon, text: str, args: argparse.Namespace) -> Speech:
    """Speak `text`: play it as it streams unless --no-play, save it if --out."""
    if args.no_play:
        speech = phonon.speak(text, args.voice)
    else:
        with Player() as player:
            speech = phonon.speak(text, args.voice, on_audio=player.feed)
    if args.out:
        print(f"Saved {save(speech.pcm, args.out)}", file=sys.stderr)
    print(report(speech), file=sys.stderr)
    return speech


def cmd_say(phonon: Phonon, args: argparse.Namespace) -> int:
    text = (sys.stdin.read() if args.text == "-" else args.text).strip()
    if not text:
        print('Give some text: phonon-demo say "Good morning."', file=sys.stderr)
        return 2
    speak_out(phonon, text, args)
    return 0


def cmd_read(phonon: Phonon, args: argparse.Namespace) -> int:
    source = args.file.read_text(encoding="utf-8") if str(args.file) != "-" else sys.stdin.read()
    text = speakable(source) if args.markdown else source
    if args.show:
        print(text)
        return 0
    speak_out(phonon, text, args)
    return 0


def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def cmd_voices(phonon: Phonon, args: argparse.Namespace) -> int:
    if args.build:
        voices.build(phonon, progress)
    names = phonon.check()
    print(f"{phonon.model}: {len(names)} voices", file=sys.stderr)
    if phonon.model == CUSTOM_MODEL:
        print(f"yours: {', '.join(voices.mine())}", file=sys.stderr)
    print("\n".join(names))
    return 0


def consented(args: argparse.Namespace) -> bool:
    """Cloning needs the speaker's agreement (Phonon's README says so too), so ask once."""
    if args.yes:
        return True
    if not sys.stdin.isatty():
        print("Add --yes to confirm the voice is yours or its owner agreed to this.", file=sys.stderr)
        return False
    answer = input(f"Is {args.file.name} your own voice, or one whose owner agreed to this use? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


def cmd_add_voice(phonon: Phonon, args: argparse.Namespace) -> int:
    if not consented(args):
        return 2
    path = voices.add(args.file, args.name, args.replace)
    progress(f"Saved {path}")
    phonon = Phonon(phonon.url, CUSTOM_MODEL, phonon.timeout)
    try:
        voices.build(phonon, progress)
    except PhononError as exc:
        raise PhononError(
            f"{exc}\nYour voice file is saved; run `phonon-demo voices --build` once that is fixed."
        ) from exc
    name = path.stem
    progress(f'Added {name} to {CUSTOM_MODEL}. Try: phonon-demo say "Hello, this is my voice." --voice "{name}"')
    return 0


def cmd_remove_voice(phonon: Phonon, args: argparse.Namespace) -> int:
    name = voices.remove(args.name)
    phonon = Phonon(phonon.url, CUSTOM_MODEL, phonon.timeout)
    if voices.mine():
        voices.build(phonon, progress)
    else:
        with contextlib.suppress(PhononError):  # already gone
            phonon.delete(CUSTOM_MODEL)
    progress(f"Removed {name}.")
    return 0


def cmd_gallery(phonon: Phonon, args: argparse.Namespace) -> int:
    voices = phonon.check()
    if args.voices:
        wanted = {v.lower() for v in args.voices}
        voices = [v for v in voices if v.lower() in wanted]
    rows = []
    for voice in voices:
        speech = phonon.speak(args.text, voice)
        path = save(speech.pcm, args.out / f"{voice.lower().replace(' ', '-')}.{args.format}")
        rows.append(f"| {voice} | [{path.name}]({path.name}) | {speech.seconds:.1f} s |")
        print(f"{voice:<28} {report(speech)}", file=sys.stderr)
    index = args.out / "README.md"
    index.write_text(
        f"# Voices of `{phonon.model}`\n\nEach file reads: *{args.text}*\n\n"
        "| voice | file | length |\n|---|---|---|\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(rows)} samples and {index}", file=sys.stderr)
    return 0


def cmd_bench(phonon: Phonon, args: argparse.Namespace) -> int:
    phonon.check()
    print(bench.markdown(bench.run(phonon, args.voice, args.runs), phonon.model, args.voice))
    return 0


def cmd_serve(phonon: Phonon, args: argparse.Namespace) -> int:
    from phonon_demo.web import serve

    phonon.check()
    serve(phonon, port=args.port, open_browser=not args.no_browser)
    return 0


COMMANDS = {
    "say": cmd_say,
    "read": cmd_read,
    "voices": cmd_voices,
    "add-voice": cmd_add_voice,
    "remove-voice": cmd_remove_voice,
    "gallery": cmd_gallery,
    "bench": cmd_bench,
    "serve": cmd_serve,
}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="phonon-demo", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    root.add_argument(
        "--model",
        default=None,
        help=f"Ollama speech model (default {DEFAULT_MODEL}, or {CUSTOM_MODEL} once you have added a voice)",
    )
    root.add_argument("--host", default=None, help="Ollama URL (default $OLLAMA_HOST or http://localhost:11434)")
    sub = root.add_subparsers(dest="cmd", required=True)

    def speaking(p: argparse.ArgumentParser) -> None:
        p.add_argument("--voice", default=None, help="a voice name; see `phonon-demo voices`")
        p.add_argument("--out", type=Path, default=None, help="save to this .wav or .mp3 as well")
        p.add_argument("--no-play", action="store_true", help="don't play it")

    p = sub.add_parser("say", help="speak some text")
    p.add_argument("text", nargs="?", default="-", help="the text; '-' or nothing reads stdin")
    speaking(p)
    p = sub.add_parser("read", help="read a file aloud")
    p.add_argument("file", type=Path, help="a text or Markdown file; '-' reads stdin")
    p.add_argument("--plain", dest="markdown", action="store_false", help="read the file as is, no clean-up")
    p.add_argument("--show", action="store_true", help="print the text that would be spoken, and stop")
    speaking(p)
    p = sub.add_parser("voices", help="list the model's voices")
    p.add_argument("--build", action="store_true", help=f"rebuild {CUSTOM_MODEL} from your voice files first")
    p = sub.add_parser("add-voice", help="add a Phonon voice file, such as your own voice")
    p.add_argument("file", type=Path, help="a Phonon voice: .safetensors with one speaker_wavs tensor")
    p.add_argument("--name", required=True, help="what to call it, as in --voice NAME")
    p.add_argument("--replace", action="store_true", help="replace a voice of yours with the same name")
    p.add_argument("--yes", action="store_true", help="confirm it is your voice, or its owner agreed")
    p = sub.add_parser("remove-voice", help="remove one of your voices")
    p.add_argument("name")
    p = sub.add_parser("gallery", help="one sample per voice")
    p.add_argument("--out", type=Path, default=Path("out/gallery"))
    p.add_argument("--text", default=GALLERY_TEXT)
    p.add_argument("--format", choices=["wav", "mp3"], default="wav")
    p.add_argument("--voices", nargs="*", help="only these voices")
    p = sub.add_parser("bench", help="measure latency and speed")
    p.add_argument("--voice", default=None)
    p.add_argument("--runs", type=int, default=3)
    p = sub.add_parser("serve", help="a local web page")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-browser", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    phonon = Phonon(args.host, args.model or (CUSTOM_MODEL if voices.mine() else DEFAULT_MODEL))
    try:
        return COMMANDS[args.cmd](phonon, args)
    except PhononError as exc:
        print(exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
