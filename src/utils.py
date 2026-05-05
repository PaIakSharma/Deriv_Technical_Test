"""Small utilities: JSON I/O, hashing, timestamping, JSON-from-LLM extraction."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    """UTC ISO-8601 timestamp, second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_hex(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def extract_json_block(text: str) -> Any:
    """Extract the first JSON object/array from an LLM response.

    Tolerates fenced code blocks (```json ... ```), prose preamble, and
    trailing commentary. Returns the parsed Python object.
    """
    if text is None:
        raise ValueError("LLM response was None.")
    text = text.strip()

    # ```json ... ``` or ``` ... ```
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Find the first '{' or '[' and try progressively shorter trailing slices.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        end = text.rfind(closer)
        while end > start:
            chunk = text[start : end + 1]
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                end = text.rfind(closer, start, end)
        # try greedy single object
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            pass

    # last resort
    return json.loads(text)
