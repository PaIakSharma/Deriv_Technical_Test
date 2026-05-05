"""Stage 1 — strategy formalisation.

One LLM call per strategy. Input: the *original* informal description, the
JSON schema we expect, and a strict instruction to surface ambiguities and
to refuse to make performance claims. Output: a JSON file at
``specs/{strategy_id}.json``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from .human_review import maybe_review
from .llm_client import LLMClient
from .utils import extract_json_block, write_json


SPECS_DIR = Path("specs")

REQUIRED_TOP_LEVEL = {
    "strategy_id",
    "instrument",
    "timeframe",
    "data_source",
    "entry_conditions",
    "exit_conditions",
    "position_sizing_rule",
    "stop_loss_rule",
    "take_profit_rule",
    "session_filters",
    "risk_controls",
    "explicit_ambiguities",
}

SCHEMA_DOC = """\
{
  "strategy_id": "string",
  "instrument": "string",
  "timeframe": "string",
  "data_source": "string",
  "entry_conditions": [
    {
      "condition_id": "string",
      "expression": "string (a precise rule, no prose hedging)",
      "indicators_required": ["string"]
    }
  ],
  "exit_conditions": [
    {
      "condition_id": "string",
      "expression": "string"
    }
  ],
  "position_sizing_rule": "string",
  "stop_loss_rule": "string | null",
  "take_profit_rule": "string | null",
  "session_filters": ["string"],
  "risk_controls": ["string"],
  "explicit_ambiguities": [
    {
      "ambiguity": "string (the specific thing the description leaves open)",
      "assumption_used_for_backtest": "string (the concrete choice we will simulate)",
      "impact_if_different": "string (what changes materially if the assumption is wrong)"
    }
  ],
  "confidence": <number in [0.0, 1.0]>
}"""

SYSTEM_PROMPT = """\
You are a careful trading-strategy formaliser. Your job is to translate a \
retail trader's informal description into a strict JSON specification \
suitable for a deterministic backtester.

Hard rules:
1. Output a single JSON object — no prose, no markdown fences.
2. Every field in the schema must be present, even if the value is null or [].
3. PRESERVE AMBIGUITY. Do not silently choose a "reasonable" interpretation \
   when the description is unclear. Surface it in `explicit_ambiguities` \
   and record both the assumption you used and how the result would shift \
   if the assumption is wrong.
4. List at least three substantive ambiguities. "Market conditions may vary" \
   or "results not guaranteed" are NOT acceptable — every entry must point \
   at something concrete in the strategy text (timezone, indicator method, \
   sizing semantics, fill model, news-day rule, etc.).
5. Do NOT compute, predict, or claim any performance number. No returns, \
   no win rates, no Sharpe ratios, no drawdowns. Performance is not your \
   concern at this stage.
6. Do NOT clean up, rewrite, or improve the strategy logic. If the trader \
   says "skip Wednesdays cause of news", encode that filter as-is.
7. If a session, day-of-week, or news-day rule is mentioned, it MUST appear \
   in `session_filters` or `risk_controls`. Silent omission is forbidden.
8. Add a top-level field `confidence` in [0.0, 1.0] reflecting how confident \
   you are that this spec faithfully captures the trader's intent. Lower the \
   value when the description is unusually vague or contradictory — outputs \
   below 0.7 will be routed to a human reviewer."""


# ---------------------------------------------------------------------------
# Mock responses — used when no API key is available. Hand-crafted to match
# the public fixture; structure-valid for any strategy_id.
# ---------------------------------------------------------------------------
def _mock_spec_for(strategy_id: str, name: str, description: str) -> Dict[str, Any]:
    desc_l = description.lower()

    if "eurusd" in desc_l or "london" in desc_l:
        return {
            "strategy_id": strategy_id,
            "confidence": 0.85,
            "instrument": "EURUSD",
            "timeframe": "1h",
            "data_source": "yfinance:EURUSD=X",
            "entry_conditions": [
                {
                    "condition_id": "long_breakout",
                    "expression": "After the first 1h bar following 08:00 London local time, enter long if any subsequent intraday bar trades at or above (opening_hour_high + 5 pips). Entry price = opening_hour_high + 5 pips.",
                    "indicators_required": ["session_opening_range"],
                },
                {
                    "condition_id": "short_breakout",
                    "expression": "Mirror of long_breakout: enter short if any bar trades at or below (opening_hour_low - 5 pips). Entry price = opening_hour_low - 5 pips.",
                    "indicators_required": ["session_opening_range"],
                },
            ],
            "exit_conditions": [
                {"condition_id": "stop_loss", "expression": "Long: exit at opening_hour_low. Short: exit at opening_hour_high."},
                {"condition_id": "take_profit", "expression": "Target = entry + 1.5 * |entry - stop|, signed by direction."},
                {"condition_id": "end_of_day", "expression": "Close any open position before NY session close (assumed 21:00 UTC)."},
            ],
            "position_sizing_rule": "1 unit per signal (size normalised; performance is reported per-unit).",
            "stop_loss_rule": "Long stop = opening_hour_low; short stop = opening_hour_high.",
            "take_profit_rule": "1.5R from entry, where R = |entry - stop|.",
            "session_filters": [
                "Trade only intraday bars after the 08:00 London opening hour.",
                "Skip all of Wednesday (news-day rule from the trader).",
                "Force-close any open position before 21:00 UTC (assumed 'NY close').",
            ],
            "risk_controls": [
                "One position at a time.",
                "Stop-loss is mandatory and equals the opposite end of the opening range.",
            ],
            "explicit_ambiguities": [
                {
                    "ambiguity": "'8am London' is timezone-ambiguous: London is UTC+0 in winter (GMT) and UTC+1 in summer (BST). The strategy text does not specify how DST is handled.",
                    "assumption_used_for_backtest": "Use 08:00 UTC year-round as the 'London open hour'. This matches London local time only in winter.",
                    "impact_if_different": "Using 07:00 UTC during BST would shift the opening range by one hour and likely change which bar's high/low anchors the breakout, materially altering entry levels.",
                },
                {
                    "ambiguity": "'NY close' is undefined: equity-market close is 21:00 UTC (winter) / 20:00 UTC (summer); FX 5pm ET rollover is 22:00 UTC / 21:00 UTC.",
                    "assumption_used_for_backtest": "Close any open position by 21:00 UTC.",
                    "impact_if_different": "An earlier or later cutoff changes how often trades exit at-market versus on stop/target, biasing the win-rate estimate.",
                },
                {
                    "ambiguity": "The pip definition for EURUSD is not stated (standard pip = 0.0001 vs fractional/pipette = 0.00001).",
                    "assumption_used_for_backtest": "1 pip = 0.0001; 5 pips = 0.0005.",
                    "impact_if_different": "If the trader meant pipettes, the breakout buffer is 10x smaller and the system would generate many more trades with much smaller risk units.",
                },
                {
                    "ambiguity": "'Skip Wednesdays cause of news' is a coarse heuristic; the trader does not specify whether it should adapt to actual high-impact news events.",
                    "assumption_used_for_backtest": "Skip every Wednesday unconditionally.",
                    "impact_if_different": "A news-calendar-driven filter would skip fewer days but more selectively, likely improving risk-adjusted returns versus blanket Wednesday avoidance.",
                },
                {
                    "ambiguity": "Position sizing is not described.",
                    "assumption_used_for_backtest": "1 unit per signal — PnL reported per unit; metrics are sizing-agnostic.",
                    "impact_if_different": "Fixed-risk-per-trade sizing (e.g. 1% of equity at risk) would change drawdown geometry and Sharpe materially.",
                },
                {
                    "ambiguity": "Intrabar ordering when both stop and target are touched in the same bar is not specified.",
                    "assumption_used_for_backtest": "Conservative: assume the stop is hit first.",
                    "impact_if_different": "Optimistic ordering (target first) would inflate win-rate by an amount proportional to the share of bars that touch both extremes.",
                },
            ],
        }

    if "rsi" in desc_l or "qqq" in desc_l:
        return {
            "strategy_id": strategy_id,
            "confidence": 0.55,
            "instrument": "QQQ",
            "timeframe": "15m",
            "data_source": "yfinance:QQQ",
            "entry_conditions": [
                {
                    "condition_id": "rsi_below_25_first_half",
                    "expression": "When RSI(14) on close drops below 25 and no position is open, enter long with size = 0.5.",
                    "indicators_required": ["RSI(14)"],
                },
                {
                    "condition_id": "rsi_below_20_second_half",
                    "expression": "If already in a half-size long position and RSI(14) drops below 20, add another 0.5 size at the new bar's open.",
                    "indicators_required": ["RSI(14)"],
                },
            ],
            "exit_conditions": [
                {"condition_id": "rsi_above_50", "expression": "Exit the entire position when RSI(14) crosses above 50."},
                {"condition_id": "end_of_day", "expression": "Close any open position at the regular session close."},
            ],
            "position_sizing_rule": "Two halves: 0.5 unit on first signal, +0.5 unit on second signal (total 1.0).",
            "stop_loss_rule": None,
            "take_profit_rule": None,
            "session_filters": [
                "No new entries within the last 30 minutes of the regular session.",
                "All positions force-closed at the regular session close.",
                "Long-only: shorts forbidden.",
            ],
            "risk_controls": [
                "Maximum one position at a time (with up to two scale-in tranches).",
                "No stop-loss is specified — exits rely on RSI mean reversion or EOD.",
            ],
            "explicit_ambiguities": [
                {
                    "ambiguity": "'RSI(14)' does not specify the smoothing method — Wilder's smoothing vs simple moving average vs exponential.",
                    "assumption_used_for_backtest": "Wilder's RSI on close prices, the standard Welles Wilder original.",
                    "impact_if_different": "SMA-based RSI is more reactive and would produce more sub-25 readings, likely increasing trade count and changing entry prices.",
                },
                {
                    "ambiguity": "'Half position' is undefined. There is no notion of total capital, leverage, or risk-per-trade.",
                    "assumption_used_for_backtest": "Half position = 0.5 units, full = 1.0 units. PnL is reported per unit.",
                    "impact_if_different": "Real risk-based sizing (e.g. fixed % of equity) would change drawdown and Sharpe but not win-rate.",
                },
                {
                    "ambiguity": "'End of day' is undefined: regular hours close 16:00 ET, extended hours run later.",
                    "assumption_used_for_backtest": "Regular session close: last 15-min bar of the regular trading day.",
                    "impact_if_different": "Holding into extended hours would expose the strategy to thin-liquidity gaps not modelled here.",
                },
                {
                    "ambiguity": "The strategy does not say whether to re-enter the same day after an exit.",
                    "assumption_used_for_backtest": "Allow re-entry after a full exit, subject to all other filters.",
                    "impact_if_different": "A 'one trade per day' rule would cap exposure and mechanically lower trade count.",
                },
                {
                    "ambiguity": "There is no stop-loss rule; the position is exposed indefinitely until RSI mean-reverts or EOD hits.",
                    "assumption_used_for_backtest": "No stop-loss; exits only on RSI > 50 or EOD.",
                    "impact_if_different": "Adding a max adverse excursion stop would cap tail-loss but also truncate winning recoveries.",
                },
                {
                    "ambiguity": "Behaviour when RSI re-crosses 25→20→25→20 within the same position.",
                    "assumption_used_for_backtest": "Second tranche fires only once per position; further crosses while in-position do nothing.",
                    "impact_if_different": "A 'pyramid' interpretation would build larger and larger exposure on continued weakness.",
                },
            ],
        }

    # default: martingale-like
    return {
        "strategy_id": strategy_id,
        "confidence": 0.78,
        "instrument": "Volatility 75 Index (synthetic)",
        "timeframe": "1m",
        "data_source": "synthetic:gbm(sigma=0.75/yr,seed=123)",
        "entry_conditions": [
            {
                "condition_id": "always_predict_up",
                "expression": "Every bar, place a fixed-direction UP prediction. Stake = current_stake.",
                "indicators_required": [],
            }
        ],
        "exit_conditions": [
            {"condition_id": "settled_next_bar", "expression": "Trade settles on the next bar's close. Win if next_close > entry_close, else lose."}
        ],
        "position_sizing_rule": "Start at $1. Double after every loss; reset to $1 after every win (martingale).",
        "stop_loss_rule": "Session-level: stop trading if cumulative drawdown exceeds $200.",
        "take_profit_rule": None,
        "session_filters": ["Single continuous session (no day-of-week or hour filters specified)."],
        "risk_controls": [
            "Hard cap: stop session if drawdown > $200.",
            "Hard cap: stop session after 50 trades.",
        ],
        "explicit_ambiguities": [
            {
                "ambiguity": "Payout assumption is not stated. Real Vol75 binary contracts pay roughly 0.95x stake on a win.",
                "assumption_used_for_backtest": "Even-money payout: win = +stake, loss = -stake.",
                "impact_if_different": "A realistic 0.95x payout would erode expectancy further and accelerate ruin.",
            },
            {
                "ambiguity": "'Tick' is undefined. Real Vol75 ticks every 2 seconds; the description also mentions '1min'.",
                "assumption_used_for_backtest": "Treat each 1-minute bar as one trade; outcome is sign of next-bar close minus current close.",
                "impact_if_different": "Higher tick frequency packs more trades per session and brings the cumulative-loss-streak event horizon closer.",
            },
            {
                "ambiguity": "Drawdown definition: peak-to-trough on equity, or session-cumulative loss?",
                "assumption_used_for_backtest": "Session-cumulative PnL falling below -$200 triggers the stop.",
                "impact_if_different": "Peak-to-trough drawdown would trip the stop sooner if the session had any winning streak before the eventual losing run.",
            },
            {
                "ambiguity": "What counts toward the 50-trade cap — entries placed or trades fully settled?",
                "assumption_used_for_backtest": "Settled trades count.",
                "impact_if_different": "Counting placements would change the off-by-one boundary at the cap but not the qualitative outcome.",
            },
            {
                "ambiguity": "'Predict UP' tie-break: is next_close == current_close a win or a loss?",
                "assumption_used_for_backtest": "Tie counts as a loss (strict greater-than for a win).",
                "impact_if_different": "Counting ties as wins would slightly improve the empirical win-rate but doesn't change the ruin geometry.",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def formalise_one(
    *,
    strategy: Dict[str, Any],
    llm: LLMClient,
) -> Dict[str, Any]:
    sid = strategy["id"]
    name = strategy.get("name", "")
    description = strategy["description"]

    user_prompt = (
        f"STRATEGY ID: {sid}\n"
        f"STRATEGY NAME: {name}\n\n"
        f"ORIGINAL DESCRIPTION (verbatim, do NOT clean up):\n"
        f"\"\"\"\n{description}\n\"\"\"\n\n"
        f"Required JSON schema (every field must be present):\n"
        f"{SCHEMA_DOC}\n\n"
        f"Return ONLY the JSON object."
    )

    raw = llm.call(
        stage="formalisation",
        strategy_id=sid,
        prompt=user_prompt,
        system=SYSTEM_PROMPT,
        input_artifacts=["strategies.json"],
        output_artifact=f"specs/{sid}.json",
        mock_response=lambda: json.dumps(_mock_spec_for(sid, name, description)),
        max_tokens=3000,
    )

    spec = extract_json_block(raw)
    spec["strategy_id"] = sid  # never let the LLM change the ID

    confidence = spec.get("confidence")
    n_amb = len(spec.get("explicit_ambiguities", []) or [])
    summary = (
        f"instrument={spec.get('instrument', '?')}, timeframe={spec.get('timeframe', '?')}, "
        f"ambiguities={n_amb}, sizing={spec.get('position_sizing_rule', '?')[:60]}"
    )
    review = maybe_review(
        stage="formalisation",
        strategy_id=sid,
        confidence=confidence if isinstance(confidence, (int, float)) else None,
        summary=summary,
        payload=spec,
    )
    if review.get("triggered"):
        spec["human_review"] = {
            "triggered": True,
            "action": review.get("action"),
            "override_notes": review.get("override_notes", ""),
            "reviewed_confidence": confidence,
        }
        notes = (review.get("override_notes") or "").strip()
        if review.get("action") == "edit" and notes:
            # Append the human's note as an additional ambiguity-style record so
            # the override is visible in the spec downstream.
            spec.setdefault("explicit_ambiguities", []).append({
                "ambiguity": "[human review override]",
                "assumption_used_for_backtest": notes,
                "impact_if_different": "Recorded by a human reviewer at low-confidence prompt; "
                                       "supersedes the model's interpretation where applicable.",
            })

    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(SPECS_DIR / f"{sid}.json", spec)
    return spec


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------
class SpecValidationError(ValueError):
    pass


_PLACEHOLDER_PATTERNS = [
    r"market conditions may vary",
    r"results not guaranteed",
    r"past performance",
    r"\bno guarantee\b",
    r"\bgeneral market risk\b",
]


def _is_placeholder_ambiguity(item: Dict[str, Any]) -> bool:
    text = " ".join(str(item.get(k, "")) for k in ("ambiguity", "assumption_used_for_backtest", "impact_if_different")).lower()
    if len(text.strip()) < 40:
        return True
    for pat in _PLACEHOLDER_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def validate_spec(spec: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable problems. Empty = valid."""
    problems: List[str] = []
    missing = REQUIRED_TOP_LEVEL - set(spec)
    if missing:
        problems.append(f"missing top-level keys: {sorted(missing)}")
        return problems  # don't keep validating a half-built object

    if not isinstance(spec["entry_conditions"], list) or not spec["entry_conditions"]:
        problems.append("entry_conditions must be a non-empty list")
    if not isinstance(spec["exit_conditions"], list) or not spec["exit_conditions"]:
        problems.append("exit_conditions must be a non-empty list")

    amb = spec.get("explicit_ambiguities", [])
    if not isinstance(amb, list):
        problems.append("explicit_ambiguities must be a list")
    else:
        substantive = [a for a in amb if isinstance(a, dict) and not _is_placeholder_ambiguity(a)]
        if len(substantive) < 3:
            problems.append(
                f"need at least 3 substantive ambiguities (got {len(substantive)} after "
                f"filtering placeholders out of {len(amb)} total)"
            )
        for i, a in enumerate(amb):
            if not isinstance(a, dict):
                problems.append(f"ambiguity[{i}] must be an object")
                continue
            for key in ("ambiguity", "assumption_used_for_backtest", "impact_if_different"):
                if not a.get(key):
                    problems.append(f"ambiguity[{i}] missing field '{key}'")

    return problems
