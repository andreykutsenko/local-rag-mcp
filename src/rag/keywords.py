"""Query expansion: extract search keywords from a question with the local LLM.

One model call per question, temperature 0, thinking disabled. The model
(qwen3:0.6b) does not keep the requested format, so parsing is tolerant and
the fallback to the original question fires only when nothing parses.
"""

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_MODEL, OLLAMA_URL

log = logging.getLogger(__name__)

MAX_RESPONSE_CHARS = 200
MAX_KEYWORD_CHARS = 60
REQUEST_TIMEOUT_SECONDS = 120
KEYWORD_PROMPT = (
    "Extract 3 to 6 search keywords from the question. "
    "Reply with the keywords only, in one line, separated by commas. "
    "No explanations.\n"
    "Question: {question}\n"
    "Keywords:"
)
LIST_MARKER = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")
LABEL_PREFIX = re.compile(r"^\s*keywords?\s*:\s*", re.IGNORECASE)
SPLIT_PATTERN = re.compile(r"[,\n]")


class OllamaUnavailableError(RuntimeError):
    """Ollama did not answer; the message tells the user what to start."""


@dataclass
class KeywordResult:
    keywords: list[str] = field(default_factory=list)
    raw_response: str = ""
    used_fallback: bool = False
    needed_tolerant_parse: bool = False


def ask_model(prompt: str) -> str:
    """Single non-streaming generation with thinking disabled."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": 0},
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.ConnectionError as error:
        raise OllamaUnavailableError(
            f"Ollama is not reachable at {OLLAMA_URL}. Start it with `ollama serve` "
            f"and make sure the model is pulled: `ollama pull {OLLAMA_MODEL}`."
        ) from error
    return response.json().get("response", "")


def parse_keywords_strict(raw: str) -> list[str]:
    """Format as requested: one line, comma-separated, no markers."""
    if "\n" in raw.strip() or LABEL_PREFIX.match(raw):
        return []
    items = [item.strip() for item in raw.split(",")]
    if any(LIST_MARKER.match(item) for item in items):
        return []
    return [item for item in items if item]


def parse_keywords_tolerant(raw: str) -> list[str]:
    """Split on commas and newlines, strip list markers, dedupe, drop long items."""
    keywords = []
    seen = set()
    for piece in SPLIT_PATTERN.split(LABEL_PREFIX.sub("", raw, count=1)):
        item = LIST_MARKER.sub("", piece).strip().strip("\"'`")
        if not item or len(item) > MAX_KEYWORD_CHARS:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        keywords.append(item)
    return keywords


def extract_keywords_detailed(question: str, ask=ask_model) -> KeywordResult:
    """Keywords plus diagnostics; `ask` is injectable so tests need no model."""
    raw = ask(KEYWORD_PROMPT.format(question=question))
    result = KeywordResult(raw_response=raw)

    if len(raw) > MAX_RESPONSE_CHARS:
        log.warning("keyword response too long (%d chars), falling back to the question", len(raw))
        result.used_fallback = True
        result.keywords = [question]
        return result

    keywords = parse_keywords_tolerant(raw)
    if not keywords:
        log.warning("no keywords parsed from %r, falling back to the question", raw)
        result.used_fallback = True
        result.keywords = [question]
        return result

    result.needed_tolerant_parse = keywords != parse_keywords_strict(raw)
    result.keywords = keywords
    return result


def extract_keywords(question: str, ask=ask_model) -> list[str]:
    return extract_keywords_detailed(question, ask).keywords


def expand_query(question: str, keywords: list[str]) -> str:
    """Search text: the original question followed by the keywords."""
    extra = [k for k in keywords if k.strip().lower() != question.strip().lower()]
    if not extra:
        return question
    return f"{question}\n{', '.join(extra)}"


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Какие настройки нужны для asyncpg?"
    r = extract_keywords_detailed(q)
    print(f"raw: {r.raw_response!r}")
    print(f"keywords: {r.keywords} fallback={r.used_fallback} tolerant={r.needed_tolerant_parse}")
