"""Turning written text into text worth speaking, and into pieces the endpoint accepts.

Phonon reads what it is given. The speech endpoint skips emoji, bullets and Markdown markers, but
a README still has code blocks, link targets, tables and HTML that sound like noise when read out.
`speakable` keeps the words a person would read aloud and drops the rest, images included
(a README's images are mostly badges).

One request takes at most 4,096 characters, so `chunks` packs the text into pieces that size,
breaking at paragraphs, then sentences, then words, never inside a word.
"""

from __future__ import annotations

import re

MAX_INPUT = 4096

# Line-anchored patterns use [ \t], never \s, which would match newlines and swallow blank lines.
_FENCE = re.compile(r"^```.*?^```[^\n]*$", re.S | re.M)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # empty text too: a badge is an image inside a link
_REF_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_REF_DEF = re.compile(r"^[ \t]*\[[^\]]+\]:[ \t]*\S+.*$", re.M)
_BARE_URL = re.compile(r"https?://\S+")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*(.*?)[ \t]*#*[ \t]*$", re.M)
_TABLE_RULE = re.compile(r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(\|[ \t]*:?-{3,}:?[ \t]*)*\|?[ \t]*$", re.M)
_BULLET = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+", re.M)
_QUOTE = re.compile(r"^[ \t]*>[ \t]?", re.M)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~)(?=\S)(.+?)(?<=\S)\1")
_RULE = re.compile(r"^[ \t]*([-*_])([ \t]*\1){2,}[ \t]*$", re.M)
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")


def speakable(markdown: str) -> str:
    """Markdown in, plain sentences out: code blocks, URLs, tables' pipes and HTML removed."""
    text = _FENCE.sub("", markdown)
    text = _HTML_COMMENT.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = _REF_DEF.sub("", text)
    text = _IMAGE.sub("", text)  # in a README, images are mostly badges: "CI, Python, License" is noise
    text = _LINK.sub(lambda m: m.group(1), text)
    text = _REF_LINK.sub(lambda m: m.group(1), text)
    text = _BARE_URL.sub("", text)
    text = _INLINE_CODE.sub(lambda m: m.group(1), text)
    text = _TABLE_RULE.sub("", text)
    text = _RULE.sub("", text)
    text = _HEADING.sub(lambda m: _end_sentence(m.group(1)), text)
    text = _BULLET.sub("", text)
    text = _QUOTE.sub("", text)
    text = _EMPHASIS.sub(lambda m: m.group(2), text)
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("|"):  # a table row: read the cells as a list
            cells = [c.strip() for c in line.strip().strip("|").split("|") if c.strip()]
            line = _end_sentence(", ".join(cells))
        lines.append(line.rstrip())
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _end_sentence(text: str) -> str:
    """A heading or table row read on its own should end like a sentence, so the voice pauses."""
    text = text.strip()
    return text if not text or text[-1] in ".!?:;…" else text + "."


def chunks(text: str, limit: int = MAX_INPUT) -> list[str]:
    """Pieces of at most `limit` characters, split at paragraph, then sentence, then word breaks."""
    pieces: list[str] = []
    current = ""
    for unit in _units(text.strip(), limit):
        joiner = "\n\n" if unit.startswith("\n") else " "
        unit = unit.strip()
        if current and len(current) + len(joiner) + len(unit) <= limit:
            current += joiner + unit
        else:
            if current:
                pieces.append(current)
            current = unit
    if current:
        pieces.append(current)
    return pieces


def _units(text: str, limit: int) -> list[str]:
    """Sentences (a paragraph's first one marked with a leading newline), each ≤ limit."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        for i, sentence in enumerate(s for s in _SENTENCE_END.split(paragraph) if s.strip()):
            for j, piece in enumerate(_by_words(sentence, limit)):
                units.append(("\n" if i == 0 and j == 0 and units else "") + piece)
    return units


def _by_words(sentence: str, limit: int) -> list[str]:
    if len(sentence) <= limit:
        return [sentence]
    out, current = [], ""
    for word in sentence.split():
        while len(word) > limit:  # a single "word" longer than a request: cut it
            out.append(word[:limit])
            word = word[limit:]
        if current and len(current) + 1 + len(word) > limit:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out
