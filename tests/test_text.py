"""Making Markdown speakable, and splitting text under the 4,096-character request limit."""

from phonon_demo.text import MAX_INPUT, chunks, speakable

README = """# Phonon demo

[![CI](https://x.y/badge.svg)](https://x.y) ![License](https://x.y/mit.svg)

Read the [Ollama docs](https://docs.ollama.com) or visit https://example.com for more.

- **Fast**: first audio in `30 ms`.
- _Local_: nothing leaves the machine.

```bash
ollama pull phonon
```

| voice | style |
|---|---|
| Zoey | calm |

<!-- a note for maintainers -->
> Quoted line.
"""


def test_speakable_keeps_the_words_and_drops_the_formatting():
    text = speakable(README)
    assert text.startswith("Phonon demo.")  # a heading ends like a sentence, so the voice pauses
    assert "Read the Ollama docs or visit for more." in text
    assert "Fast: first audio in 30 ms." in text and "Local: nothing leaves the machine." in text
    assert "Zoey, calm." in text and "voice, style." in text
    assert "Quoted line." in text
    noise = ["```", "ollama pull", "https://", "|", "**", "#", "<!--", "maintainers", "---"]
    badges = ["CI", "License", "[", "]", "("]
    for fragment in noise + badges:
        assert fragment not in text


def test_speakable_leaves_plain_text_alone():
    plain = "Good morning. Checkout recovered at 14:32, and refunds of $49.99 went out."
    assert speakable(plain) == plain


def test_short_text_is_one_chunk():
    assert chunks("Hello there. General Kenobi.") == ["Hello there. General Kenobi."]
    assert chunks("   ") == []


def test_long_text_splits_at_sentences_under_the_limit():
    sentence = "This sentence is exactly as long as it needs to be for the test. "
    text = sentence * 200  # about 13,000 characters
    pieces = chunks(text)
    assert len(pieces) > 1
    assert all(len(p) <= MAX_INPUT for p in pieces)
    assert all(p.endswith(".") for p in pieces)  # never cut mid-sentence
    assert " ".join(pieces).split() == text.split()  # nothing lost or reordered


def test_paragraph_breaks_survive_inside_a_chunk():
    assert chunks("First paragraph.\n\nSecond paragraph.") == ["First paragraph.\n\nSecond paragraph."]


def test_a_sentence_longer_than_the_limit_splits_at_words():
    words = ("word " * 2000).strip()  # 9,999 characters, no full stop
    pieces = chunks(words, limit=1000)
    assert all(len(p) <= 1000 for p in pieces)
    assert " ".join(pieces).split() == words.split()
