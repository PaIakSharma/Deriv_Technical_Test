"""Parameter sensitivity sweeps + LLM interpretation.

Strategy A: breakout threshold around 5 pips, swept ±50% (2.5 ... 7.5).
Strategy B: RSI entry threshold around 25, swept ±50% (12.5 ... 37.5).
Other strategies are skipped — sensitivity is only requested for A and B in
the brief, but the structure is generic.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .llm_client import LLMClient
from .metrics import compute_metrics
from .strategies import infer_template, run_backtest
from .utils import extract_json_block


def _sweep_breakout(spec: Dict[str, Any], ohlcv: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pips in np.linspace(2.5, 7.5, 6):
        result = run_backtest(spec, ohlcv, template="breakout_session", breakout_pips=float(pips))
        m = compute_metrics(result)
        rows.append({"parameter": "breakout_pips", "value": float(pips), "metrics": m})
    return rows


def _sweep_rsi(spec: Dict[str, Any], ohlcv: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for thr in np.linspace(12.5, 37.5, 6):
        # Keep second-tranche threshold at 5 below first, mirroring the original spec gap.
        result = run_backtest(
            spec, ohlcv, template="rsi_mean_reversion",
            entry_first_threshold=float(thr),
            entry_second_threshold=float(max(5.0, thr - 5.0)),
        )
        m = compute_metrics(result)
        rows.append({"parameter": "rsi_entry_threshold", "value": float(thr), "metrics": m})
    return rows


SYSTEM_PROMPT = (
    "You are a quantitative reviewer interpreting a parameter sweep table. "
    "Output a single JSON object — no prose, no markdown. The numbers in the "
    "table are authoritative; do not recompute them."
)

SCHEMA = """\
{
  "strategy_id": "string",
  "parameter": "string",
  "stability_summary": "string — does performance smoothly degrade or are there cliffs?",
  "best_value": <number>,
  "worst_value": <number>,
  "interpretation": "string — what does the shape of this curve imply about robustness?"
}"""


def _mock_interpretation(strategy_id: str, parameter: str, sweep: List[Dict[str, Any]]) -> Dict[str, Any]:
    # pick best/worst by total_pnl
    by_pnl = sorted(sweep, key=lambda r: r["metrics"]["total_pnl"])
    worst = by_pnl[0]
    best = by_pnl[-1]
    pnls = [r["metrics"]["total_pnl"] for r in sweep]
    spread = max(pnls) - min(pnls)
    smooth = (
        "smooth: PnL changes monotonically across the swept range, suggesting the strategy is not "
        "sharply tuned to the central value."
        if pnls == sorted(pnls) or pnls == sorted(pnls, reverse=True)
        else "non-monotonic: PnL has at least one local optimum, hinting at parameter-space cliffs."
    )
    interp = (
        f"Spread between best and worst PnL across the {parameter} sweep is {spread:.4f}. "
        f"This {'is large relative to the central value' if abs(spread) > 1e-6 else 'is essentially flat'} "
        f"and indicates that the original choice was {'not robust' if spread > 0 else 'reasonably robust'} "
        f"in this sample."
    )
    return {
        "strategy_id": strategy_id,
        "parameter": parameter,
        "stability_summary": smooth,
        "best_value": float(best["value"]),
        "worst_value": float(worst["value"]),
        "interpretation": interp,
    }


def run_sensitivity(
    *,
    spec: Dict[str, Any],
    ohlcv: pd.DataFrame,
    llm: LLMClient,
) -> Dict[str, Any]:
    sid = spec["strategy_id"]
    template = infer_template(spec)
    if template == "breakout_session":
        sweep = _sweep_breakout(spec, ohlcv)
        parameter = "breakout_pips"
    elif template == "rsi_mean_reversion":
        sweep = _sweep_rsi(spec, ohlcv)
        parameter = "rsi_entry_threshold"
    else:
        return {"strategy_id": sid, "skipped": True, "reason": f"no sweep defined for template {template}"}

    table = [{"value": r["value"], "total_pnl": r["metrics"]["total_pnl"],
              "n_trades": r["metrics"]["n_trades"], "sharpe": r["metrics"]["sharpe_annualised"],
              "max_dd": r["metrics"]["max_drawdown_pnl_units"]} for r in sweep]

    user_prompt = (
        f"STRATEGY ID: {sid}\nPARAMETER: {parameter}\n\n"
        f"Sweep results (computed deterministically):\n{json.dumps(table, indent=2)}\n\n"
        f"Required schema:\n{SCHEMA}\n\nReturn ONLY the JSON object."
    )
    raw = llm.call(
        stage="sensitivity_interpretation",
        strategy_id=sid,
        prompt=user_prompt,
        system=SYSTEM_PROMPT,
        input_artifacts=[f"specs/{sid}.json", "metrics.json"],
        output_artifact="parameter_sensitivity.json",
        mock_response=lambda: json.dumps(_mock_interpretation(sid, parameter, sweep)),
        max_tokens=1500,
    )
    interpretation = extract_json_block(raw)

    return {
        "strategy_id": sid,
        "parameter": parameter,
        "table": table,
        "interpretation": interpretation,
    }
