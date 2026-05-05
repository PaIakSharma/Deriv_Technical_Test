"""Human-in-the-loop review for low-confidence LLM outputs.

Every LLM stage now asks the model to attach a ``confidence`` value in
[0.0, 1.0]. When that value is below the threshold (default 0.7), the
pipeline pauses for human input.

Behaviour is controlled by env vars:

* ``HUMAN_REVIEW_THRESHOLD`` — float, default 0.7.
* ``HUMAN_REVIEW`` —
    * ``auto`` (default): prompt if stdin is a TTY; otherwise auto-accept
      and record the auto-acceptance.
    * ``off``: never prompt.
    * ``always``: prompt regardless of confidence.

All review events (including auto-accepts) are appended to
``human_reviews.jsonl`` for full replay traceability.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import now_iso


REVIEW_LOG = Path("human_reviews.jsonl")
DEFAULT_THRESHOLD = 0.7


def _threshold() -> float:
    raw = os.environ.get("HUMAN_REVIEW_THRESHOLD", str(DEFAULT_THRESHOLD)).strip()
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_THRESHOLD


def _mode() -> str:
    return os.environ.get("HUMAN_REVIEW", "auto").strip().lower() or "auto"


def _append_log(record: Dict[str, Any]) -> None:
    REVIEW_LOG.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def reset_log() -> None:
    if REVIEW_LOG.exists():
        REVIEW_LOG.unlink()


def maybe_review(
    *,
    stage: str,
    strategy_id: str,
    confidence: Optional[float],
    summary: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a dict with at minimum ``{"triggered": bool}``.

    If a review fires and the human supplies override text, the dict will
    also contain ``"override_notes": str`` and ``"action": "edit"|"accept"|"skip"``.
    The caller is responsible for actually applying any overrides — this
    function just gathers them.
    """
    threshold = _threshold()
    mode = _mode()
    if mode == "off":
        return {"triggered": False, "reason": "HUMAN_REVIEW=off"}

    conf_known = isinstance(confidence, (int, float))
    needs_review = (mode == "always") or (conf_known and float(confidence) < threshold)
    if not needs_review:
        # still record (lightly) so the trail is complete
        rec = {
            "timestamp": now_iso(),
            "stage": stage,
            "strategy_id": strategy_id,
            "confidence": confidence,
            "threshold": threshold,
            "triggered": False,
            "action": "no_review",
            "notes": "",
        }
        _append_log(rec)
        return {"triggered": False}

    print("", flush=True)
    print("=" * 72, flush=True)
    print(
        f"[human-review] stage={stage} strategy={strategy_id} "
        f"confidence={confidence if conf_known else 'unknown'} "
        f"(threshold={threshold})",
        flush=True,
    )
    print(f"[human-review] summary: {summary}", flush=True)
    print("=" * 72, flush=True)

    interactive = sys.stdin.isatty() if hasattr(sys.stdin, "isatty") else False

    action = "accept"
    notes = ""

    if interactive or mode == "always" or not sys.stdin.closed:
        # Try to read from stdin even if not a TTY (so a piped input demo works).
        try:
            print(
                "[human-review] choose: (a)ccept / (e)dit notes / (s)kip — "
                "blank input = accept",
                flush=True,
            )
            line = sys.stdin.readline()
            choice = (line or "").strip().lower()
            if choice in ("e", "edit"):
                print("[human-review] enter override note (single line):", flush=True)
                notes_line = sys.stdin.readline()
                notes = (notes_line or "").strip()
                action = "edit"
            elif choice in ("s", "skip"):
                action = "skip"
            else:
                action = "accept"
        except Exception as e:  # noqa: BLE001
            print(f"[human-review] could not read stdin ({e}); auto-accepting", flush=True)
            action = "auto_accept_no_input"
    else:
        action = "auto_accept_no_tty"

    print(f"[human-review] action={action}; notes={notes!r}", flush=True)

    rec = {
        "timestamp": now_iso(),
        "stage": stage,
        "strategy_id": strategy_id,
        "confidence": confidence,
        "threshold": threshold,
        "triggered": True,
        "action": action,
        "notes": notes,
        "summary": summary,
    }
    _append_log(rec)
    return {"triggered": True, "action": action, "override_notes": notes}
