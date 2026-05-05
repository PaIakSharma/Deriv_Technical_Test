"""Strategy executors.

Three concrete templates cover the public fixture:
    breakout_session       (Strategy A)
    rsi_mean_reversion     (Strategy B)
    martingale_fixed       (Strategy C)

A spec is dispatched to one of these by ``infer_template``. Each executor
returns a ``BacktestResult``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .backtest import BacktestResult, Trade


# ---------------------------------------------------------------------------
# Template inference
# ---------------------------------------------------------------------------
def infer_template(spec: Dict[str, Any]) -> str:
    text = " ".join(
        [
            str(spec.get("instrument", "")),
            str(spec.get("position_sizing_rule", "")),
            " ".join(c.get("expression", "") for c in spec.get("entry_conditions", [])),
            " ".join(c.get("expression", "") for c in spec.get("exit_conditions", [])),
            " ".join(spec.get("session_filters", []) or []),
            " ".join(spec.get("risk_controls", []) or []),
        ]
    ).lower()

    if "martingale" in text or "double" in text and "stake" in text:
        return "martingale_fixed"
    if "rsi" in text:
        return "rsi_mean_reversion"
    if "breakout" in text or "opening_hour" in text or "opening range" in text:
        return "breakout_session"

    # Heuristic by instrument
    instr = str(spec.get("instrument", "")).lower()
    if "vol" in instr or "volatility" in instr:
        return "martingale_fixed"
    if "qqq" in instr or "spy" in instr or "nasdaq" in instr:
        return "rsi_mean_reversion"
    if "eur" in instr or "gbp" in instr or "fx" in instr:
        return "breakout_session"
    return "breakout_session"  # safe fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _equity_point(ts, equity):
    return {"timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts), "equity": float(equity)}


# ---------------------------------------------------------------------------
# Strategy A — London breakout
# ---------------------------------------------------------------------------
def run_breakout_session(
    spec: Dict[str, Any],
    ohlcv: pd.DataFrame,
    *,
    breakout_pips: float = 5.0,
    pip_size: float = 0.0001,
    london_open_hour_utc: int = 8,
    ny_close_hour_utc: int = 21,
    skip_dow: Optional[set] = None,
) -> BacktestResult:
    if skip_dow is None:
        skip_dow = {"Wednesday"}

    sid = spec["strategy_id"]
    res = BacktestResult(strategy_id=sid)
    res.notes.append(f"breakout_pips={breakout_pips}, pip_size={pip_size}")
    res.notes.append(f"london_open_hour_utc={london_open_hour_utc}, ny_close_hour_utc={ny_close_hour_utc}")
    res.notes.append(f"skip_dow={sorted(skip_dow)}")

    df = ohlcv.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    res.bars_total = len(df)
    in_pos_bars = 0
    equity = 0.0
    res.equity_curve.append(_equity_point(df.index[0], equity))

    threshold = breakout_pips * pip_size

    # Group by calendar date in UTC
    for date, day_bars in df.groupby(df.index.date):
        dow = pd.Timestamp(date).day_name()
        if dow in skip_dow:
            for ts in day_bars.index:
                res.equity_curve.append(_equity_point(ts, equity))
            continue

        opening = day_bars[day_bars.index.hour == london_open_hour_utc]
        if len(opening) == 0:
            for ts in day_bars.index:
                res.equity_curve.append(_equity_point(ts, equity))
            continue
        ref_high = float(opening.iloc[0]["High"])
        ref_low = float(opening.iloc[0]["Low"])

        rest = day_bars[day_bars.index > opening.index[0]]
        rest = rest[rest.index.hour < ny_close_hour_utc]
        if rest.empty:
            for ts in day_bars.index:
                res.equity_curve.append(_equity_point(ts, equity))
            continue

        # Drop the opening bar from the equity curve update — already at the start.
        in_position = False
        direction = entry_price = stop = target = None
        entry_time = None

        for ts, bar in rest.iterrows():
            high, low, close = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
            if not in_position:
                long_trigger = ref_high + threshold
                short_trigger = ref_low - threshold
                hit_long = high >= long_trigger and low <= long_trigger
                hit_short = low <= short_trigger and high >= short_trigger
                # In the same bar, prefer whichever trigger price is closer to the bar open
                if hit_long and hit_short:
                    open_p = float(bar["Open"])
                    if abs(open_p - long_trigger) <= abs(open_p - short_trigger):
                        hit_short = False
                    else:
                        hit_long = False
                if hit_long:
                    direction = "long"
                    entry_price = long_trigger
                    stop = ref_low
                    target = entry_price + 1.5 * (entry_price - stop)
                    in_position = True
                    entry_time = ts
                elif hit_short:
                    direction = "short"
                    entry_price = short_trigger
                    stop = ref_high
                    target = entry_price - 1.5 * (stop - entry_price)
                    in_position = True
                    entry_time = ts
            else:
                in_pos_bars += 1
                hit_stop = (direction == "long" and low <= stop) or (direction == "short" and high >= stop)
                hit_target = (direction == "long" and high >= target) or (direction == "short" and low <= target)
                exit_price = exit_reason = None
                if hit_stop and hit_target:
                    # Conservative: stop first.
                    exit_price = stop
                    exit_reason = "stop_loss_intrabar_conflict"
                elif hit_stop:
                    exit_price = stop
                    exit_reason = "stop_loss"
                elif hit_target:
                    exit_price = target
                    exit_reason = "take_profit"
                if exit_price is not None:
                    pnl = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
                    ret_pct = pnl / entry_price
                    res.trades.append(Trade(
                        strategy_id=sid, entry_time=entry_time, exit_time=ts,
                        direction=direction, entry_price=entry_price, exit_price=exit_price,
                        size=1.0, pnl=pnl, return_pct=ret_pct, exit_reason=exit_reason,
                    ))
                    equity += pnl
                    in_position = False

            res.equity_curve.append(_equity_point(ts, equity))

        # End-of-day forced exit
        if in_position:
            last_bar = rest.iloc[-1]
            ts = rest.index[-1]
            exit_price = float(last_bar["Close"])
            pnl = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
            ret_pct = pnl / entry_price
            res.trades.append(Trade(
                strategy_id=sid, entry_time=entry_time, exit_time=ts,
                direction=direction, entry_price=entry_price, exit_price=exit_price,
                size=1.0, pnl=pnl, return_pct=ret_pct, exit_reason="end_of_day",
            ))
            equity += pnl
            res.equity_curve[-1] = _equity_point(ts, equity)

    res.bars_in_position = in_pos_bars
    return res


# ---------------------------------------------------------------------------
# Strategy B — RSI mean reversion
# ---------------------------------------------------------------------------
def run_rsi_mean_reversion(
    spec: Dict[str, Any],
    ohlcv: pd.DataFrame,
    *,
    rsi_period: int = 14,
    entry_first_threshold: float = 25.0,
    entry_second_threshold: float = 20.0,
    exit_threshold: float = 50.0,
    last_n_minutes_blackout: int = 30,
    bar_minutes: int = 15,
) -> BacktestResult:
    sid = spec["strategy_id"]
    res = BacktestResult(strategy_id=sid)
    res.notes.append(f"rsi({rsi_period}), entry<{entry_first_threshold}/+<{entry_second_threshold}, exit>{exit_threshold}")
    res.notes.append(f"blackout_last_min={last_n_minutes_blackout}")

    df = ohlcv.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df["rsi"] = _wilder_rsi(df["Close"], rsi_period)
    res.bars_total = len(df)
    blackout_bars = max(1, last_n_minutes_blackout // bar_minutes)
    equity = 0.0
    res.equity_curve.append(_equity_point(df.index[0], equity))
    in_pos_bars = 0

    # Group by date — RSI is computed on the full series, exits at session-end per day.
    grouped = list(df.groupby(df.index.date))

    for date, day in grouped:
        # bars eligible for new entries (exclude last blackout window of the day)
        n = len(day)
        last_entry_idx = max(0, n - blackout_bars)

        position_size = 0.0
        avg_entry = 0.0
        added_second = False
        entry_time = None

        for i, (ts, bar) in enumerate(day.iterrows()):
            close = float(bar["Close"])
            rsi = float(bar["rsi"])

            # Manage open position: exit on RSI > 50, or end of day
            if position_size > 0:
                in_pos_bars += 1
                end_of_day = i == n - 1
                exit_now = rsi > exit_threshold or end_of_day
                if exit_now:
                    exit_price = close
                    pnl = (exit_price - avg_entry) * position_size
                    ret_pct = (exit_price - avg_entry) / avg_entry
                    res.trades.append(Trade(
                        strategy_id=sid, entry_time=entry_time, exit_time=ts,
                        direction="long", entry_price=avg_entry, exit_price=exit_price,
                        size=position_size, pnl=pnl, return_pct=ret_pct,
                        exit_reason="end_of_day" if end_of_day else "rsi_above_50",
                    ))
                    equity += pnl
                    position_size = 0.0
                    avg_entry = 0.0
                    added_second = False
                    entry_time = None
                else:
                    # Already long; check if we should add the second tranche.
                    if not added_second and rsi < entry_second_threshold and i < last_entry_idx:
                        new_size = 0.5
                        # Average entry blends:
                        avg_entry = (avg_entry * position_size + close * new_size) / (position_size + new_size)
                        position_size += new_size
                        added_second = True
            else:
                # No position: open if RSI < first threshold and we're not in blackout window
                if rsi < entry_first_threshold and i < last_entry_idx:
                    position_size = 0.5
                    avg_entry = close
                    entry_time = ts
                    added_second = False
                    # If RSI is already < second threshold on the same bar, fill both tranches.
                    if rsi < entry_second_threshold:
                        position_size = 1.0
                        added_second = True

            res.equity_curve.append(_equity_point(ts, equity))

    res.bars_in_position = in_pos_bars
    return res


# ---------------------------------------------------------------------------
# Strategy C — fixed-direction martingale
# ---------------------------------------------------------------------------
def run_martingale_fixed(
    spec: Dict[str, Any],
    ohlcv: pd.DataFrame,
    *,
    initial_stake: float = 1.0,
    max_drawdown_dollars: float = 200.0,
    max_trades: int = 50,
) -> BacktestResult:
    sid = spec["strategy_id"]
    res = BacktestResult(strategy_id=sid)
    res.notes.append(f"initial_stake=${initial_stake}, dd_cap=${max_drawdown_dollars}, max_trades={max_trades}")
    res.notes.append("Even-money payout. Tie counts as loss. One settled trade per bar.")

    df = ohlcv.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    res.bars_total = len(df)
    closes = df["Close"].values
    times = df.index

    equity = 0.0
    stake = initial_stake
    n_trades = 0
    res.equity_curve.append(_equity_point(times[0], equity))

    # Trade is: at bar i, stake on UP. Settles at bar i+1. Win if close[i+1] > close[i].
    for i in range(len(closes) - 1):
        if n_trades >= max_trades:
            res.notes.append(f"Hit max_trades cap at i={i}")
            break
        if equity <= -abs(max_drawdown_dollars):
            res.notes.append(f"Hit drawdown cap (equity={equity:.2f}) at i={i}")
            break

        entry_price = float(closes[i])
        exit_price = float(closes[i + 1])
        win = exit_price > entry_price
        pnl = stake if win else -stake
        equity += pnl
        n_trades += 1

        res.trades.append(Trade(
            strategy_id=sid,
            entry_time=times[i],
            exit_time=times[i + 1],
            direction="long",
            entry_price=entry_price,
            exit_price=exit_price,
            size=stake,
            pnl=pnl,
            return_pct=(pnl / stake) if stake else 0.0,
            exit_reason="binary_win" if win else "binary_loss",
        ))
        res.equity_curve.append(_equity_point(times[i + 1], equity))
        stake = initial_stake if win else stake * 2.0

    # All other bars accumulate equity unchanged for exposure/curve continuity
    last_logged = res.equity_curve[-1]["timestamp"]
    last_equity = res.equity_curve[-1]["equity"]
    for ts in times:
        if ts.isoformat() > last_logged:
            res.equity_curve.append(_equity_point(ts, last_equity))

    # Exposure: every bar where a stake was at risk counts.
    res.bars_in_position = len(res.trades)
    return res


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def run_backtest(spec: Dict[str, Any], ohlcv: pd.DataFrame, **overrides: Any) -> BacktestResult:
    template = overrides.pop("template", None) or infer_template(spec)
    if template == "breakout_session":
        return run_breakout_session(spec, ohlcv, **overrides)
    if template == "rsi_mean_reversion":
        return run_rsi_mean_reversion(spec, ohlcv, **overrides)
    if template == "martingale_fixed":
        return run_martingale_fixed(spec, ohlcv, **overrides)
    raise ValueError(f"Unknown template: {template}")
