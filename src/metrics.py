"""Deterministic performance metrics.

All metrics are computed here in pure Python. The LLM never runs any of this.
Inputs: a per-trade ledger and an equity curve. Outputs: a metrics dict that
the validator can reconcile against the ledger.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .backtest import BacktestResult


def _annualisation_factor_from_equity(equity_df: pd.DataFrame) -> float:
    """Compute sqrt(N) where N = approx bars per year given equity timestamps."""
    if len(equity_df) < 3:
        return 1.0
    ts = pd.to_datetime(equity_df["timestamp"], utc=True, errors="coerce").dropna()
    if len(ts) < 3:
        return 1.0
    spans = ts.diff().dropna().dt.total_seconds()
    spans = spans[spans > 0]
    if len(spans) == 0:
        return 1.0
    median_seconds = float(np.median(spans))
    if median_seconds <= 0:
        return 1.0
    bars_per_year = (365.0 * 24 * 3600) / median_seconds
    return math.sqrt(bars_per_year)


def _max_drawdown(equity_series: np.ndarray) -> float:
    """Peak-to-trough drawdown on absolute equity (PnL units), expressed
    as a positive number. Equity here is cumulative PnL (can be negative).
    """
    if len(equity_series) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_series)
    drawdown = running_max - equity_series  # always >= 0
    return float(drawdown.max())


def _largest_losing_streak(pnls: List[float]) -> int:
    cur = best = 0
    for p in pnls:
        if p < 0:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _avg_trade_duration_seconds(trades: List[Dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    durations: List[float] = []
    for t in trades:
        try:
            entry = pd.Timestamp(t["entry_time"])
            exit_ = pd.Timestamp(t["exit_time"])
            durations.append(max(0.0, (exit_ - entry).total_seconds()))
        except Exception:  # noqa: BLE001
            continue
    if not durations:
        return 0.0
    return float(np.mean(durations))


def compute_metrics(result: BacktestResult) -> Dict[str, Any]:
    rows = [t.as_row() for t in result.trades]
    if rows:
        ledger = pd.DataFrame(rows)
    else:
        ledger = pd.DataFrame(
            columns=["pnl", "return_pct", "entry_time", "exit_time", "exit_reason"]
        )

    n_trades = int(len(ledger))
    pnls = ledger["pnl"].astype(float).tolist() if n_trades else []
    total_pnl = float(sum(pnls))

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = (n_wins / n_trades) if n_trades else 0.0

    gross_profit = float(sum(wins))
    gross_loss = float(-sum(losses))  # positive number
    if gross_loss > 0:
        profit_factor: Any = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # Equity curve → drawdown and bar-level returns for Sharpe/Sortino
    eq_df = pd.DataFrame(result.equity_curve)
    if len(eq_df) >= 2:
        eq_series = eq_df["equity"].astype(float).values
        max_dd = _max_drawdown(eq_series)
        bar_returns = np.diff(eq_series)
        ann = _annualisation_factor_from_equity(eq_df)
    else:
        max_dd = 0.0
        bar_returns = np.array([])
        ann = 1.0

    if bar_returns.size > 1 and bar_returns.std(ddof=1) > 0:
        sharpe = float((bar_returns.mean() / bar_returns.std(ddof=1)) * ann)
    else:
        sharpe = 0.0

    downside = bar_returns[bar_returns < 0]
    if downside.size > 1 and downside.std(ddof=1) > 0:
        sortino = float((bar_returns.mean() / downside.std(ddof=1)) * ann)
    else:
        sortino = 0.0

    exposure_pct = (
        float(result.bars_in_position) / float(result.bars_total)
        if result.bars_total
        else 0.0
    )

    avg_dur = _avg_trade_duration_seconds(rows)
    largest_losing_streak = _largest_losing_streak(pnls)

    # Total return: for absolute PnL strategies (Strategy C), express as PnL.
    # For unit-priced strategies (A, B), report total PnL too — there is no
    # capital base specified, so a pure % return is undefined. We report
    # both an absolute total PnL and the sum of per-trade return_pct as
    # complementary views, with the bar-level Sharpe as the primary risk-
    # adjusted metric.
    sum_return_pct = float(ledger["return_pct"].astype(float).sum()) if n_trades else 0.0

    return {
        "strategy_id": result.strategy_id,
        "n_trades": n_trades,
        "total_pnl": total_pnl,
        "sum_return_pct": sum_return_pct,
        "win_rate": win_rate,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "profit_factor": profit_factor if profit_factor != float("inf") else "inf",
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "max_drawdown_pnl_units": max_dd,
        "sharpe_annualised": sharpe,
        "sortino_annualised": sortino,
        "avg_trade_duration_seconds": avg_dur,
        "exposure_pct": exposure_pct,
        "largest_losing_streak": largest_losing_streak,
        "annualisation_factor_used": ann,
        "bars_total": int(result.bars_total),
        "bars_in_position": int(result.bars_in_position),
        "engine_notes": list(result.notes),
    }


def reconcile_ledger(ledger_csv: str, metrics: Dict[str, Any], tolerance: float = 1e-2) -> List[str]:
    """Return a list of reconciliation problems (empty = reconciled)."""
    problems: List[str] = []
    df = pd.read_csv(ledger_csv)
    n = len(df)
    if n != metrics["n_trades"]:
        problems.append(f"trade count mismatch: ledger={n}, metrics={metrics['n_trades']}")
    if n > 0:
        ledger_pnl = float(df["pnl"].sum())
        if abs(ledger_pnl - metrics["total_pnl"]) > tolerance:
            problems.append(
                f"total pnl mismatch: ledger={ledger_pnl:.6f}, metrics={metrics['total_pnl']:.6f}"
            )
        wins = int((df["pnl"] > 0).sum())
        if wins != metrics["n_wins"]:
            problems.append(f"win count mismatch: ledger={wins}, metrics={metrics['n_wins']}")
    return problems
