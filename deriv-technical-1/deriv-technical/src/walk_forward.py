"""Walk-forward validation: split history into 3 chronological windows and
re-run each strategy on each window. Flag stability per strategy.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .metrics import compute_metrics
from .strategies import run_backtest


def _stability_flag(window_metrics: List[Dict[str, Any]]) -> str:
    if not window_metrics:
        return "insufficient_data"
    pnls = [m["total_pnl"] for m in window_metrics]
    n_total = sum(m["n_trades"] for m in window_metrics)
    if n_total == 0:
        return "insufficient_data"
    if len(pnls) < 3:
        return "insufficient_data"

    # "Stable" = all windows positive (or all non-positive) and no window
    #           swings by more than 3x the median absolute window PnL.
    # "Degrading" = strictly decreasing PnL across windows.
    # "Unstable" = signs differ across windows.
    signs = [(1 if p > 0 else (-1 if p < 0 else 0)) for p in pnls]
    if pnls[0] > pnls[1] > pnls[2]:
        return "degrading"
    if len(set(s for s in signs if s != 0)) > 1:
        return "unstable"

    abs_pnls = [abs(p) for p in pnls]
    med = sorted(abs_pnls)[1]
    if med == 0:
        return "unstable" if any(abs_pnls) else "insufficient_data"
    if max(abs_pnls) > 3.0 * med:
        return "unstable"
    return "stable"


def walk_forward_one(spec: Dict[str, Any], ohlcv: pd.DataFrame) -> Dict[str, Any]:
    sid = spec["strategy_id"]
    n = len(ohlcv)
    if n < 30:
        return {"strategy_id": sid, "windows": [], "stability": "insufficient_data"}

    cuts = [0, n // 3, 2 * n // 3, n]
    windows: List[Dict[str, Any]] = []
    for i in range(3):
        sub = ohlcv.iloc[cuts[i] : cuts[i + 1]]
        if len(sub) < 10:
            windows.append({
                "window_index": i,
                "start": str(sub.index[0]) if len(sub) else None,
                "end": str(sub.index[-1]) if len(sub) else None,
                "metrics": None,
                "note": "too_few_bars",
            })
            continue
        result = run_backtest(spec, sub)
        m = compute_metrics(result)
        windows.append({
            "window_index": i,
            "start": str(sub.index[0]),
            "end": str(sub.index[-1]),
            "metrics": m,
        })

    flag = _stability_flag([w["metrics"] for w in windows if w.get("metrics")])
    return {"strategy_id": sid, "windows": windows, "stability": flag}
