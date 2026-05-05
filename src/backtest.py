"""Deterministic backtest engine.

Every numerical output produced by this pipeline originates here or in
``metrics.py``. The LLM never executes any of this code and never sees the
raw OHLCV bars.

Intrabar ordering assumption (documented and applied uniformly):
    If both stop-loss and take-profit are touched in the same bar, assume
    the STOP-LOSS is hit first unless the strategy explicitly overrides.
    This is the conservative choice and is recorded in
    ``BACKTEST_ASSUMPTIONS`` so reviewers see it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Top-level documented assumptions, persisted to disk for traceability.
BACKTEST_ASSUMPTIONS: Dict[str, str] = {
    "intrabar_ordering": (
        "If both stop-loss and take-profit are touched in the same bar, the "
        "stop-loss is assumed to fill first."
    ),
    "fills": (
        "Stops and targets fill at exactly the trigger price (no slippage). "
        "Market exits (EOD, RSI mean-revert) fill at the bar's close."
    ),
    "fees": "No commissions, spreads, or financing costs are modelled.",
    "timezones": (
        "All bar timestamps are normalised to UTC before backtest logic runs. "
        "Strategy-specific session windows interpret hours in UTC."
    ),
    "rsi_method": "Wilder smoothing (the original RSI formulation).",
    "pip_size_eurusd": "1 pip = 0.0001.",
    "ny_close_assumed_utc": "21:00 UTC.",
    "london_open_assumed_utc": "08:00 UTC (matches London local in winter).",
    "vol75_payout": "Even-money: win = +stake, loss = -stake. Tie = loss.",
    "size_units": (
        "All trades sized in abstract 'units' unless the strategy specifies "
        "a stake (Strategy C uses real dollar stakes for its martingale)."
    ),
}


@dataclass
class Trade:
    strategy_id: str
    entry_time: Any
    exit_time: Any
    direction: str            # 'long' | 'short'
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float         # PnL / (entry_price * size) for price-based; PnL / stake for binary
    exit_reason: str

    def as_row(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "entry_time": self.entry_time.isoformat() if hasattr(self.entry_time, "isoformat") else str(self.entry_time),
            "exit_time": self.exit_time.isoformat() if hasattr(self.exit_time, "isoformat") else str(self.exit_time),
            "direction": self.direction,
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price),
            "size": float(self.size),
            "pnl": float(self.pnl),
            "return_pct": float(self.return_pct),
            "exit_reason": self.exit_reason,
        }


@dataclass
class BacktestResult:
    strategy_id: str
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)  # [{ts, equity}]
    bars_in_position: int = 0
    bars_total: int = 0
    notes: List[str] = field(default_factory=list)
