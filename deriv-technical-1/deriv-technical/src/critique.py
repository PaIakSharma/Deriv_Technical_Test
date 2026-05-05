"""Stage 2 — strategy critique.

One LLM call per strategy *after* metrics have been computed deterministically.
The LLM is given:
    - the formal JSON spec
    - the pre-computed metrics summary
    - per-trade ledger summary statistics
    - an equity curve sampled to ~50 points
    - the documented backtest assumptions

It is asked for a robustness critique. For martingale-style strategies the
critique must explicitly address ruin risk, path dependency, drawdown
acceleration, and why a high win rate may be misleading.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .backtest import BACKTEST_ASSUMPTIONS
from .human_review import maybe_review
from .llm_client import LLMClient
from .strategies import infer_template
from .utils import extract_json_block


CRITIQUES_PATH = Path("critiques.json")


SYSTEM_PROMPT = """\
You are a senior quantitative risk reviewer. Your job is to read a strategy \
specification, its DETERMINISTICALLY computed performance metrics, and a \
sampled equity curve, then produce a robustness critique.

Hard rules:
1. Output a single JSON object — no prose, no markdown fences.
2. Do NOT recompute or restate metrics with different numbers. The numbers \
   you are given are authoritative.
3. Do NOT make forward-looking return predictions.
4. Be specific. Tie every concern to a fact in the spec, the metrics, or \
   the equity curve. Generic "past performance is no guarantee" boilerplate \
   is not acceptable on its own.
5. If the strategy uses martingale / loss-doubling / fixed-direction binary \
   bets, you MUST explicitly address ruin risk, path dependency, drawdown \
   acceleration, and why a high win rate is misleading. Set risk_level to \
   "high" in that case.
6. risk_level must be one of: "low", "medium", "high".
7. is_high_risk must be true if and only if risk_level == "high"."""


SCHEMA_DOC = """\
{
  "strategy_id": "string",
  "risk_level": "low | medium | high",
  "is_high_risk": true | false,
  "is_martingale_or_loss_escalating": true | false,
  "overfitting_risk": "string — what specifically might be in-sample-fitted",
  "regime_dependence": "string — which market regimes this depends on / would break",
  "assumption_sensitivity": "string — which assumption_used_for_backtest entries change the conclusion most",
  "execution_realism": "string — discuss slippage, fills, news days, gaps, fees the backtest does not model",
  "likely_failure_modes": ["string", "..."],
  "robustness_verdict": "robust | fragile",
  "warnings": ["string", "..."],
  "ruin_risk_discussion": "string — REQUIRED for martingale/loss-escalating; for others, may say 'not applicable: <why>'",
  "high_win_rate_misleading_explanation": "string — REQUIRED for martingale/loss-escalating; otherwise 'not applicable: <why>'",
  "confidence": <number in [0.0, 1.0]>
}"""


def _equity_curve_sample(equity_curve: List[Dict[str, Any]], n: int = 50) -> List[Dict[str, Any]]:
    if len(equity_curve) <= n:
        return equity_curve
    idx = np.linspace(0, len(equity_curve) - 1, n).round().astype(int)
    return [equity_curve[int(i)] for i in idx]


def _ledger_summary(ledger: pd.DataFrame) -> Dict[str, Any]:
    if ledger.empty:
        return {
            "n_trades": 0, "exit_reason_counts": {}, "max_winning_streak": 0,
            "max_losing_streak": 0, "biggest_win": 0.0, "biggest_loss": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0,
        }
    pnls = ledger["pnl"].astype(float).tolist()
    counts = ledger["exit_reason"].value_counts().to_dict()
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    def streak(arr, sign):
        cur = best = 0
        for p in arr:
            ok = (p > 0) if sign > 0 else (p < 0)
            if ok:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    return {
        "n_trades": int(len(ledger)),
        "exit_reason_counts": {str(k): int(v) for k, v in counts.items()},
        "max_winning_streak": streak(pnls, 1),
        "max_losing_streak": streak(pnls, -1),
        "biggest_win": float(max(wins)) if wins else 0.0,
        "biggest_loss": float(min(losses)) if losses else 0.0,
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
    }


def _is_martingale(spec: Dict[str, Any]) -> bool:
    return infer_template(spec) == "martingale_fixed"


# ---------------------------------------------------------------------------
# Mock critique — used when no API key. Hand-crafted to satisfy validator
# (Strategy C must come back high-risk).
# ---------------------------------------------------------------------------
def _mock_critique(spec: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    sid = spec["strategy_id"]
    martingale = _is_martingale(spec)
    n_trades = metrics.get("n_trades", 0)
    sharpe = metrics.get("sharpe_annualised", 0.0)
    pf = metrics.get("profit_factor", 0.0)
    max_dd = metrics.get("max_drawdown_pnl_units", 0.0)

    if martingale:
        return {
            "strategy_id": sid,
            "confidence": 0.92,
            "risk_level": "high",
            "is_high_risk": True,
            "is_martingale_or_loss_escalating": True,
            "overfitting_risk": (
                "Not classical curve-fitting risk because the rule has no parameters tuned on data, "
                "but the empirical 'profitability' over short windows is an artefact of the specific "
                "loss-streak distribution observed, not a stable edge."
            ),
            "regime_dependence": (
                "Catastrophically dependent on never observing a long enough loss streak to breach the "
                "drawdown cap. Any regime with mean-reverting returns or higher autocorrelation in losses "
                "(e.g. trending down) accelerates ruin."
            ),
            "assumption_sensitivity": (
                "Most sensitive to (a) payout assumption — even-money was assumed; real Vol75 binaries "
                "pay <1.0x, which makes expected value strictly negative; and (b) tick frequency — at 2s "
                "tick rate the trader sees ~30x more events per session, packing the loss-streak event "
                "into a much smaller wall-clock window."
            ),
            "execution_realism": (
                "Backtest assumes perfect fills at bar close, no slippage, no platform latency, and no "
                "platform-side stake limits. Real martingale runs hit broker max-stake caps before the "
                "drawdown cap, which converts a paper -$200 cap into an unrecoverable position."
            ),
            "likely_failure_modes": [
                "8+ consecutive losses → cumulative -$255 exceeds -$200 cap; session ends in ruin.",
                "Broker max-stake cap hits before doubling completes; loss can't be recovered.",
                "Sub-1.0 payout makes expectancy negative even with 50% win rate.",
                "Path dependency: P(losing streak ≥ k) grows with session length.",
            ],
            "robustness_verdict": "fragile",
            "warnings": [
                "MARTINGALE / LOSS-DOUBLING — high risk of ruin regardless of in-sample win rate.",
                "Geometric stake growth means a single tail event wipes out many small winners.",
                "Performance metrics are misleading: Sharpe is dominated by the absence of the tail event in this short sample.",
            ],
            "ruin_risk_discussion": (
                f"Stake doubling produces cumulative risk 2^k - 1 after k consecutive losses. To breach the "
                f"$200 cap requires only 8 losses in a row — at a 50% win rate that has probability 1/256 "
                f"per any 8-length window, so over even a moderate session the cumulative probability of "
                f"ruin is non-trivial and approaches 1 as session length grows. Backtest measured "
                f"max_dd={max_dd:.2f} on this seed but a different seed could exceed it without changing "
                f"any rule."
            ),
            "high_win_rate_misleading_explanation": (
                f"With even-money payouts and 50% win probability the expectancy of each trade is exactly "
                f"zero. The strategy wins ~50% of trades in the sample (n={n_trades}), but each of those "
                f"wins yields the small reset stake while a single 8-loss run loses 255x that stake. A "
                f"high win-rate accounting therefore disguises strongly negative skew."
            ),
        }

    # Non-martingale defaults — tailor a bit by template
    is_breakout = "breakout" in str(spec.get("instrument", "")).lower() or "EUR" in str(spec.get("instrument", ""))
    is_rsi = any("rsi" in c.get("expression", "").lower() for c in spec.get("entry_conditions", []))

    if is_rsi:
        regime = (
            "RSI mean-reversion entries depend on rangebound or mildly trending behaviour. In a strong "
            "downtrend RSI<25 prints repeatedly while price keeps falling, and entries get run over."
        )
        failure = [
            "Persistent downtrends — RSI oversold readings stay oversold while price keeps dropping.",
            "Gap-down opens around earnings or macro news — no stop-loss is defined.",
            "Whipsaw days where RSI bounces 25 → 50 → 25 generate frequent in-and-out trades.",
        ]
        overfit = (
            "Thresholds 25 / 20 / 50 are the textbook defaults; the rule is not parameter-tuned to this "
            "sample, but choice of those exact levels is convention rather than evidence."
        )
    elif is_breakout:
        regime = (
            "Opening-range breakout requires intraday continuation after the London open. Range-bound days "
            "with no follow-through generate stop-outs; gappy news mornings tend to over-trigger and reverse."
        )
        failure = [
            "Range days: the price re-enters the opening range and stops the trade out at the opposite extreme.",
            "Real news on non-Wednesday days bypasses the day-of-week filter.",
            "Wide opening-hour ranges produce stop distances that dominate the 1.5R target.",
        ]
        overfit = (
            "Fixed 5-pip breakout buffer, fixed 1.5R target, and the Wednesday skip are unjustified by the "
            "data presented. Performance could be sensitive to all three."
        )
    else:
        regime = "Generic price-momentum dependency; could degrade in the opposite regime."
        failure = ["Out-of-regime drawdown.", "Assumption drift.", "Sample-period bias."]
        overfit = "Limited information to assess overfitting from the description alone."

    verdict = "fragile" if (n_trades < 30 or sharpe < 0.5 or (isinstance(pf, (int, float)) and pf < 1.1)) else "robust"

    # If the underlying spec was uncertain (low formalisation confidence) the
    # critique should also be less confident — propagate that downward so the
    # human reviewer is asked.
    spec_confidence = float(spec.get("confidence", 0.8) or 0.8)
    crit_confidence = max(0.4, min(0.95, spec_confidence - 0.05))

    return {
        "strategy_id": sid,
        "confidence": crit_confidence,
        "risk_level": "medium" if verdict == "fragile" else "low",
        "is_high_risk": False,
        "is_martingale_or_loss_escalating": False,
        "overfitting_risk": overfit,
        "regime_dependence": regime,
        "assumption_sensitivity": (
            "Most sensitive to the timezone and 'end of day' assumptions; secondary sensitivity to "
            "intrabar ordering between stop and target."
        ),
        "execution_realism": (
            "Backtest assumes perfect fills at trigger prices with no slippage and no spread cost. "
            "Real intraday fills, especially around news, would widen effective stops."
        ),
        "likely_failure_modes": failure,
        "robustness_verdict": verdict,
        "warnings": [
            "Sample size is limited to the available history window; conclusions do not generalise.",
            "No transaction costs are modelled; net returns will be lower in practice.",
        ],
        "ruin_risk_discussion": "not applicable: this strategy does not use loss-escalating sizing.",
        "high_win_rate_misleading_explanation": "not applicable: win rate is not used as the primary edge claim.",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def critique_one(
    *,
    spec: Dict[str, Any],
    metrics: Dict[str, Any],
    equity_curve: List[Dict[str, Any]],
    ledger_df: pd.DataFrame,
    llm: LLMClient,
) -> Dict[str, Any]:
    sid = spec["strategy_id"]
    sample = _equity_curve_sample(equity_curve, 50)
    summary = _ledger_summary(ledger_df)
    martingale = _is_martingale(spec)

    payload = {
        "spec": spec,
        "metrics_summary": metrics,
        "ledger_summary": summary,
        "equity_curve_sampled_50pts": sample,
        "backtest_assumptions": BACKTEST_ASSUMPTIONS,
        "is_martingale_or_loss_escalating": martingale,
    }

    user_prompt = (
        f"STRATEGY ID: {sid}\n\n"
        "All numbers below were computed deterministically by Python code. "
        "Do not recompute or contradict them.\n\n"
        f"INPUT JSON:\n{json.dumps(payload, indent=2, default=str)}\n\n"
        f"Required schema for your response:\n{SCHEMA_DOC}\n\n"
        "Return ONLY the JSON object."
    )

    raw = llm.call(
        stage="critique",
        strategy_id=sid,
        prompt=user_prompt,
        system=SYSTEM_PROMPT,
        input_artifacts=[f"specs/{sid}.json", "metrics.json"],
        output_artifact="critiques.json",
        mock_response=lambda: json.dumps(_mock_critique(spec, metrics)),
        max_tokens=2500,
    )

    parsed = extract_json_block(raw)
    parsed["strategy_id"] = sid

    confidence = parsed.get("confidence")
    summary = (
        f"verdict={parsed.get('robustness_verdict', '?')}, "
        f"risk_level={parsed.get('risk_level', '?')}, "
        f"is_high_risk={parsed.get('is_high_risk', '?')}"
    )
    review = maybe_review(
        stage="critique",
        strategy_id=sid,
        confidence=confidence if isinstance(confidence, (int, float)) else None,
        summary=summary,
        payload=parsed,
    )
    if review.get("triggered"):
        parsed["human_review"] = {
            "triggered": True,
            "action": review.get("action"),
            "override_notes": review.get("override_notes", ""),
            "reviewed_confidence": confidence,
        }
        notes = (review.get("override_notes") or "").strip()
        if review.get("action") == "edit" and notes:
            warns = list(parsed.get("warnings", []) or [])
            warns.append(f"[human review] {notes}")
            parsed["warnings"] = warns

    # Safety net: even if the LLM wavers, we DO NOT let martingale strategies
    # be filed as anything other than high-risk. The numerical / structural
    # decision is ours, not the model's.
    if martingale:
        parsed["risk_level"] = "high"
        parsed["is_high_risk"] = True
        parsed["is_martingale_or_loss_escalating"] = True
        if not str(parsed.get("ruin_risk_discussion", "")).strip() or \
                "not applicable" in str(parsed.get("ruin_risk_discussion", "")).lower():
            parsed["ruin_risk_discussion"] = (
                "Loss-doubling sizing produces cumulative risk 2^k - 1 after k consecutive losses; "
                "ruin is path-dependent and grows with session length."
            )
        if not str(parsed.get("high_win_rate_misleading_explanation", "")).strip() or \
                "not applicable" in str(parsed.get("high_win_rate_misleading_explanation", "")).lower():
            parsed["high_win_rate_misleading_explanation"] = (
                "Wins return the small reset stake while a single long losing run loses 2^k-1 times that "
                "stake; high win rate is consistent with strongly negative skew."
            )
        # Ensure a clear textual warning is present.
        warns = list(parsed.get("warnings", []) or [])
        if not any("martingale" in str(w).lower() or "ruin" in str(w).lower() for w in warns):
            warns.insert(0, "MARTINGALE / LOSS-DOUBLING — high risk of ruin regardless of in-sample win rate.")
            parsed["warnings"] = warns

    return parsed
