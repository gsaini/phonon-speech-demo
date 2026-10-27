# Plan: speak in your own voice

**Status: plan only, nothing built yet.** Written 2026-10-26, six days after Phonon's release, from Gradium's docs and press coverage. Re-check the open questions before building.

## Goal

Record about 10 seconds of your own voice, then have `phonon-demo say`, `read` and the web page speak in it, the same way they use built-in voices like Zoey today:

```bash
uv run phonon-demo clone me.wav --name Me
uv run phonon-demo say "Checkout recovered at 14:32." --voice Me
```

## What we know

- **The Ollama `phonon` model only has built-in voices.** `ollama show phonon` lists 38 named voices. It shows no way to add one, and `POST /v1/audio/speech` takes only a voice name.
- **Gradium clones voices in its cloud.** `POST https://api.gradium.ai/api/voices/` (multipart, `x-api-key` header) takes `audio_file`, `name` and `language` (`en`, `fr`, `de`, `es`, `pt`). It recommends at least 10 seconds of clear audio. It returns an id that Gradium's cloud TTS accepts as `voice_id`. Gradium's Python SDK wraps this as `gradium.voices.create(...)`.
- **Gradium says cloned voices run in Phonon locally.** Coverage of the release says a voice made in Gradium Studio or through the API is "downloaded, loaded into Phonon and runs locally". It also says creating voices is free, and that Phonon's README requires the speaker's consent to clone a voice.
- **Not confirmed: whether Ollama can load a downloaded voice.** I found no Modelfile directive, API field or doc that loads one. Phonon's own Rust inference code may support it even if Ollama doesn't.
- **Licenses differ by checkpoint, as they do for the built-in voices.** The default English tag is Apache 2.0. The 40M checkpoints, including the multilingual one, are CC BY-NC 4.0.

## Step 1: find out what the local path is (half a day)

Answer these before writing any code. Each answer chooses between route A and route B below.

1. Does Gradium's voice API return a downloadable voice file, and in what format? Make one test clone of your own voice and look at what comes back.
2. Can Ollama load that file? Check the Ollama speech docs, the Ollama source for the Phonon runner, and Ollama's issue tracker. Look for a Modelfile directive, a voice directory, or a request field.
3. If Ollama can't, can Phonon's own inference code, run outside Ollama?
4. Which checkpoints accept a custom voice, and does the license change?

## Route A: a Gradium cloud clone, run locally (preferred, if step 1 confirms it)

The recording goes to Gradium once to make the voice. After that, every request is local, and the speed numbers in the README still apply.

- `phonon_demo/voices.py`: upload a recording to the voices API, download the voice file into `~/.phonon-demo/voices/<name>/`, and list or delete local voices. Standard library only (`urllib`, multipart by hand), matching `client.py`. The key comes from `$GRADIUM_API_KEY` and is never stored.
- Wire the voice into Ollama however step 1 says: a derived model such as `ollama create phonon-me` from a Modelfile, or a request field. `Phonon.voices()` merges custom voices into the list, so `--voice Me` works unchanged.
- CLI: `phonon-demo clone <audio> --name <name> [--language en] [--start 0]`, `phonon-demo voices --custom`, and `phonon-demo forget <name>`.
- Web page: a "Your voices" group in the voice picker. Recording in the browser comes later, if ever: a file upload is enough at first.

## Route B: a local cloning model (if route A can't run locally)

If the voice can only be used through Gradium's cloud, keep Phonon for the built-in voices and add a second, local engine that clones from a short reference clip. Candidates that run on Apple Silicon:

- **Qwen3-TTS Base through `mlx-audio`.** It clones from about 10 seconds of audio, and the download is about 1.7 GB.
- **`mlx-speech`.** It includes several models that clone from reference audio plus its transcript. It needs Python 3.13 or later. Its optional denoiser uses non-commercial weights.

What this costs: the project's first runtime dependency (make it an optional extra, `pip install phonon-speech-demo[clone]`), a much larger download, and probably slower speech. Measure it with `bench` before choosing. The `Phonon` client would gain a sibling with the same `speak()` result type, so `audio.py`, `text.py` and the web page don't change.

## Consent and safety

- Clone only your own voice, or a voice whose owner has agreed, which Phonon's README requires. `clone` asks the user to confirm that before uploading.
- Keep voice files outside the repo, in `~/.phonon-demo/voices/`. Never commit recordings or voice files. The samples in `samples/` stay built-in voices only.
- Say plainly in the README that route A sends the recording to Gradium once.

## Tests

The tests stay offline, like the current 20. A fake voices API returns a canned voice file, and a fake Ollama checks that the custom voice reaches the request. No real recording goes into `tests/`; use generated tones.

## Done when

- `clone` then `say --voice <name>` works end to end on an M-series Mac, and the clip sounds like the speaker.
- The README has a short "Your own voice" section with the consent note and the measured timings.
- CI still needs neither Ollama, Apple Silicon nor a Gradium key.

## Sources

- [Gradium docs: custom voices](https://docs.gradium.ai/guides/voices/custom-voices.md)
- [Gradium: how to clone a voice](https://gradium.ai/content/how-to-clone-a-voice-create-custom-voice)
- [Gradium Phonon: on-device TTS](https://gradium.ai/content/gradium-phonon-on-device-tts)
- [Voice AI Space: Gradium open sources Phonon](https://www.voiceaispace.com/articles/gradium-open-sources-phonon-on-device-text-to-speech-models)
- [Ollama docs: speech](https://docs.ollama.com/capabilities/speech)
- [mlx-speech](https://github.com/appautomaton/mlx-speech)
- [mlx-qwen3tts with a front end for voice cloning](https://www.voiceaispace.com/knowledge/repo/mlx-qwen3tts-with-frontend-for-voicecloning)
