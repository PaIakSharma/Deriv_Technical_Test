"""Adversarial scenario generation.

The LLM proposes a small set of synthetic price-path *parameters* designed
to stress each strategy. Then the parameters are converted into deterministic
GBM-with-shock paths in pure Python, and the backtest is replayed against
each path. The LLM never produces price values directly.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .data_loader import gbm_ohlcv
from .llm_client import LLMClient
from .metrics import compute_metrics
from .strategies import infer_template, run_backtest
from .utils import extract_json_block


SYSTEM_PROMPT = (
    "You design adversarial market scenarios to stress-test trading strategies. "
    "You DO NOT produce price values. You only produce SCENARIO PARAMETERS that "
    "a deterministic Python simulator will turn into prices. Output a single JSON "
    "object — no prose, no markdown."
)


SCENARIO_SCHEMA = """\
{
  "scenarios": [
    {
      "name": "string",
      "description": "string — what regime this represents and why it stresses this strategy",
      "process": "gbm_with_shocks",
      "drift_per_year": <number>,
      "sigma_per_year": <number>,
      "shock_bars": [<int bar_index>, ...],   // 0..n_bars-1; OPTIONAL
      "shock_pct": [<number>, ...],            // signed % moves applied at shock_bars
      "expected_failure_mode": "string"
    },
    ... exactly 3 scenarios
  ]
}"""


def _mock_scenarios(spec: Dict[str, Any]) -> Dict[str, Any]:
    template = infer_template(spec)
    if template == "breakout_session":
        scenarios = [
            {
                "name": "trendless_chop",
                "description": "Low-vol mean-reverting day-after-day; opening ranges trigger breakouts that immediately reverse.",
                "process": "gbm_with_shocks",
                "drift_per_year": 0.0,
                "sigma_per_year": 0.04,
                "shock_bars": [],
                "shock_pct": [],
                "expected_failure_mode": "Both long and short breakouts stop out repeatedly; max losing streak grows.",
            },
            {
                "name": "news_gap_open",
                "description": "Wednesday-skipped, but a Tuesday morning shock 50 bars in expands range and gaps through stops.",
                "process": "gbm_with_shocks",
                "drift_per_year": 0.0,
                "sigma_per_year": 0.07,
                "shock_bars": [50],
                "shock_pct": [-1.5],
                "expected_failure_mode": "Long breakout trigger hits, then sudden reversal blows through stop with slippage we don't model.",
            },
            {
                "name": "single_direction_grind",
                "description": "Low-vol persistent uptrend; opening ranges always break upward — but 1.5R targets miss because of small bar bodies.",
                "process": "gbm_with_shocks",
                "drift_per_year": 0.30,
                "sigma_per_year": 0.05,
                "shock_bars": [],
                "shock_pct": [],
                "expected_failure_mode": "Trades enter long but EOD-exit before targets, capping wins.",
            },
        ]
    elif template == "rsi_mean_reversion":
        scenarios = [
            {
                "name": "persistent_downtrend",
                "description": "Strong downtrend keeps RSI under 25; entries stack and ride the trend down.",
                "process": "gbm_with_shocks",
                "drift_per_year": -0.40,
                "sigma_per_year": 0.30,
                "shock_bars": [],
                "shock_pct": [],
                "expected_failure_mode": "Repeated full-size entries with no stop-loss; large unrealised drawdown until EOD.",
            },
            {
                "name": "earnings_cliff",
                "description": "Quiet drift then a single -8% shock 100 bars in.",
                "process": "gbm_with_shocks",
                "drift_per_year": 0.0,
                "sigma_per_year": 0.10,
                "shock_bars": [100],
                "shock_pct": [-8.0],
                "expected_failure_mode": "If a position is open through the shock, no stop catches it; large biggest_loss.",
            },
            {
                "name": "rsi_whipsaw",
                "description": "High vol mean-reverting around the entry threshold.",
                "process": "gbm_with_shocks",
                "drift_per_year": 0.0,
                "sigma_per_year": 0.45,
                "shock_bars": [],
                "shock_pct": [],
                "expected_failure_mode": "Many small frictionless trades but no consistent edge; profit factor near 1.",
            },
        ]
    else:  # martingale
        scenarios = [
            {
                "name": "long_loss_streak",
                "description": "Slight negative drift produces clusters of losing predictions.",
                "process": "gbm_with_shocks",
                "drift_per_year": -0.10,
                "sigma_per_year": 0.75,
                "shock_bars": [],
                "shock_pct": [],
                "expected_failure_mode": "8+ consecutive losses breach the $200 drawdown cap; ruin.",
            },
            {
                "name": "high_vol_random",
                "description": "Pure random walk at higher sigma; tail of loss streaks gets fatter.",
                "process": "gbm_with_shocks",
                "drift_per_year": 0.0,
                "sigma_per_year": 1.20,
                "shock_bars": [],
                "shock_pct": [],
                "expected_failure_mode": "Probability of an 8-loss streak in 50 trades rises; ruin frequency increases.",
            },
            {
                "name": "downward_shock_then_recover",
                "description": "Stable random walk with a single concentrated 5-bar down move at bar 20.",
                "process": "gbm_with_shocks",
                "drift_per_year": 0.0,
                "sigma_per_year": 0.75,
                "shock_bars": [20, 21, 22, 23, 24],
                "shock_pct": [-0.5, -0.5, -0.5, -0.5, -0.5],
                "expected_failure_mode": "Shock window forces 5 consecutive losses, doubling stake to $32; subsequent random losses can finish the ruin.",
            },
        ]
    return {"scenarios": scenarios}


def _build_path(scn: Dict[str, Any], n_bars: int, bar_minutes: float, initial_price: float, seed: int) -> pd.DataFrame:
    df = gbm_ohlcv(
        n_bars=n_bars,
        bar_minutes=bar_minutes,
        sigma_per_year=float(scn.get("sigma_per_year", 0.20)),
        drift_per_year=float(scn.get("drift_per_year", 0.0)),
        seed=seed,
        initial_price=initial_price,
    )
    shock_bars = scn.get("shock_bars") or []
    shock_pct = scn.get("shock_pct") or []
    if shock_bars and shock_pct and len(shock_bars) == len(shock_pct):
        df = df.copy()
        for b, p in zip(shock_bars, shock_pct):
            b = int(b)
            if 0 <= b < len(df):
                factor = 1.0 + float(p) / 100.0
                df.iloc[b:, :4] = df.iloc[b:, :4].values * factor
    return df


def run_adversarial(
    *,
    spec: Dict[str, Any],
    base_ohlcv: pd.DataFrame,
    llm: LLMClient,
    seed: int = 999,
) -> Dict[str, Any]:
    sid = spec["strategy_id"]
    template = infer_template(spec)

    user_prompt = (
        f"STRATEGY ID: {sid}\nTEMPLATE: {template}\n\n"
        f"Strategy spec:\n{json.dumps(spec, indent=2)}\n\n"
        "Propose exactly 3 adversarial scenarios. Use ONLY the parameter shapes "
        "in the schema below — a Python simulator will build the prices.\n\n"
        f"Schema:\n{SCENARIO_SCHEMA}\n\nReturn ONLY the JSON object."
    )
    raw = llm.call(
        stage="adversarial_scenario_proposal",
        strategy_id=sid,
        prompt=user_prompt,
        system=SYSTEM_PROMPT,
        input_artifacts=[f"specs/{sid}.json"],
        output_artifact="adversarial_scenarios.json",
        mock_response=lambda: json.dumps(_mock_scenarios(spec)),
        max_tokens=2000,
    )
    proposal = extract_json_block(raw)
    scenarios_in = proposal.get("scenarios", [])[:3]

    n_bars = max(200, min(len(base_ohlcv), 1500))
    if len(base_ohlcv) >= 2:
        delta = (base_ohlcv.index[1] - base_ohlcv.index[0]).total_seconds() / 60.0
        bar_minutes = max(1.0, float(delta) or 60.0)
    else:
        bar_minutes = 60.0
    initial_price = float(base_ohlcv["Close"].iloc[0]) if len(base_ohlcv) else 100.0

    out_scenarios = []
    for i, scn in enumerate(scenarios_in):
        path = _build_path(scn, n_bars=n_bars, bar_minutes=bar_minutes, initial_price=initial_price, seed=seed + i)
        result = run_backtest(spec, path)
        m = compute_metrics(result)
        out_scenarios.append({
            "name": scn.get("name", f"scenario_{i}"),
            "description": scn.get("description", ""),
            "generated_path_assumptions": {
                "process": scn.get("process", "gbm_with_shocks"),
                "drift_per_year": scn.get("drift_per_year"),
                "sigma_per_year": scn.get("sigma_per_year"),
                "shock_bars": scn.get("shock_bars"),
                "shock_pct": scn.get("shock_pct"),
                "n_bars": int(n_bars),
                "bar_minutes": float(bar_minutes),
                "seed": int(seed + i),
                "initial_price": float(initial_price),
            },
            "backtest_result": m,
            "failure_mode_observed": _failure_mode_observed(m, scn.get("expected_failure_mode", "")),
        })
    return {"strategy_id": sid, "scenarios": out_scenarios}


def _failure_mode_observed(metrics: Dict[str, Any], expected: str) -> str:
    if metrics["n_trades"] == 0:
        observed = "no trades triggered"
    elif metrics["total_pnl"] < 0:
        observed = (
            f"net loss of {metrics['total_pnl']:.4f} over {metrics['n_trades']} trades; "
            f"max losing streak {metrics['largest_losing_streak']}, max DD {metrics['max_drawdown_pnl_units']:.4f}"
        )
    else:
        observed = (
            f"strategy survived (total_pnl={metrics['total_pnl']:.4f}, n_trades={metrics['n_trades']}, "
            f"max_dd={metrics['max_drawdown_pnl_units']:.4f})"
        )
    if expected:
        return f"expected: {expected} | observed: {observed}"
    return observed
