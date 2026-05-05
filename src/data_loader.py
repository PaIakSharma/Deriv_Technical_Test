"""OHLCV loader with yfinance + seeded GBM fallback.

Records everything it does to ``data_manifest.json`` so the run is replayable.
If yfinance is unavailable (no network, blocked, or missing dep) the loader
substitutes seeded GBM for the affected symbol and records
``synthetic_fallback: true`` for that entry.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import now_iso

DATA_DIR = Path("data")
MANIFEST_PATH = Path("data_manifest.json")


# ---------------------------------------------------------------------------
# Synthetic GBM
# ---------------------------------------------------------------------------
def gbm_ohlcv(
    *,
    n_bars: int,
    bar_minutes: float,
    sigma_per_year: float,
    drift_per_year: float = 0.0,
    seed: int = 123,
    initial_price: float = 100.0,
    start: Optional[pd.Timestamp] = None,
    intrabar_steps: int = 10,
) -> pd.DataFrame:
    """Generate a deterministic OHLCV series from geometric Brownian motion.

    To populate realistic high/low values within each bar we simulate
    ``intrabar_steps`` sub-bar prices and aggregate.
    """
    rng = np.random.default_rng(seed)
    minutes_per_year = 365 * 24 * 60
    dt_total = bar_minutes / minutes_per_year
    dt_step = dt_total / intrabar_steps

    total_steps = n_bars * intrabar_steps
    z = rng.standard_normal(total_steps)
    log_step = (drift_per_year - 0.5 * sigma_per_year ** 2) * dt_step \
        + sigma_per_year * math.sqrt(dt_step) * z

    log_path = np.empty(total_steps + 1, dtype=np.float64)
    log_path[0] = math.log(initial_price)
    np.cumsum(log_step, out=log_path[1:])
    log_path[1:] += log_path[0]
    prices = np.exp(log_path)

    if start is None:
        start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    index = pd.date_range(start=start, periods=n_bars, freq=f"{int(bar_minutes)}min")

    rows = []
    for i in range(n_bars):
        sub = prices[i * intrabar_steps : (i + 1) * intrabar_steps + 1]
        rows.append(
            {
                "Open": float(sub[0]),
                "High": float(sub.max()),
                "Low": float(sub.min()),
                "Close": float(sub[-1]),
                "Volume": 0.0,
            }
        )
    df = pd.DataFrame(rows, index=index)
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# yfinance
# ---------------------------------------------------------------------------
def _yf_download(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError:
        print(f"[data] yfinance not installed; cannot fetch {ticker}", flush=True)
        return None
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[data] yfinance error for {ticker}: {e}", flush=True)
        return None
    if df is None or df.empty:
        return None

    # yfinance can return MultiIndex columns when a single symbol is requested.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str)
    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(df.columns):
        return None
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        return None

    # Force tz to UTC for downstream consistency.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_eurusd_1h() -> Tuple[pd.DataFrame, Dict]:
    """Strategy A data: EUR/USD 1h.

    Tries yfinance (`EURUSD=X`, period 730d, interval 1h). Falls back to GBM.
    """
    info: Dict = {
        "symbol": "EURUSD=X",
        "interval": "1h",
        "preferred_source": "yfinance",
        "synthetic_fallback": False,
        "fetched_at": now_iso(),
    }
    df = _yf_download("EURUSD=X", period="730d", interval="1h")
    if df is None or len(df) < 200:
        df = gbm_ohlcv(
            n_bars=24 * 30 * 6,  # ~6 months of 1h bars
            bar_minutes=60,
            sigma_per_year=0.07,
            drift_per_year=0.0,
            seed=701,
            initial_price=1.10,
            start=pd.Timestamp("2024-01-01", tz="UTC"),
        )
        info.update(
            {
                "synthetic_fallback": True,
                "process": "geometric_brownian_motion",
                "drift_per_year": 0.0,
                "sigma_per_year": 0.07,
                "seed": 701,
                "initial_price": 1.10,
                "n_bars": int(len(df)),
                "bar_minutes": 60,
            }
        )
    else:
        info.update({"n_bars": int(len(df))})

    info["start"] = str(df.index[0])
    info["end"] = str(df.index[-1])
    info["columns"] = list(df.columns)
    return df, info


def load_qqq_15m() -> Tuple[pd.DataFrame, Dict]:
    """Strategy B data: QQQ 15m. yfinance caps 15m intervals at ~60 days."""
    info: Dict = {
        "symbol": "QQQ",
        "interval": "15m",
        "preferred_source": "yfinance",
        "synthetic_fallback": False,
        "fetched_at": now_iso(),
    }
    df = _yf_download("QQQ", period="55d", interval="15m")
    if df is None or len(df) < 200:
        # ~55 trading days * 26 bars/day
        df = gbm_ohlcv(
            n_bars=55 * 26,
            bar_minutes=15,
            sigma_per_year=0.20,
            drift_per_year=0.0,
            seed=702,
            initial_price=400.0,
            start=pd.Timestamp("2024-01-02 14:30", tz="UTC"),
        )
        info.update(
            {
                "synthetic_fallback": True,
                "process": "geometric_brownian_motion",
                "drift_per_year": 0.0,
                "sigma_per_year": 0.20,
                "seed": 702,
                "initial_price": 400.0,
                "n_bars": int(len(df)),
                "bar_minutes": 15,
            }
        )
    else:
        info.update({"n_bars": int(len(df))})

    info["start"] = str(df.index[0])
    info["end"] = str(df.index[-1])
    info["columns"] = list(df.columns)
    return df, info


def load_vol75_1m(*, n_bars: int = 60 * 24 * 3, seed: int = 123) -> Tuple[pd.DataFrame, Dict]:
    """Strategy C data: Volatility 75 Index proxy.

    Always synthetic — no public free data source for Deriv synthetic indices.
    GBM with drift=0, sigma=0.75/yr, 1-minute bars, 3 days of session.
    """
    df = gbm_ohlcv(
        n_bars=n_bars,
        bar_minutes=1,
        sigma_per_year=0.75,
        drift_per_year=0.0,
        seed=seed,
        initial_price=100.0,
        start=pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
    )
    info = {
        "symbol": "Vol75-synthetic",
        "interval": "1m",
        "preferred_source": "geometric_brownian_motion",
        "synthetic_fallback": True,  # synthetic by design
        "process": "geometric_brownian_motion",
        "drift_per_year": 0.0,
        "sigma_per_year": 0.75,
        "seed": seed,
        "initial_price": 100.0,
        "n_bars": int(n_bars),
        "bar_minutes": 1,
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "columns": list(df.columns),
        "fetched_at": now_iso(),
    }
    return df, info


def cache_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p)


def load_for_strategy(strategy_id: str, name: str) -> Tuple[pd.DataFrame, Dict]:
    """Dispatch by best-guess instrument detection.

    Public-fixture A → EURUSD; B → QQQ; C → Vol75. Unknowns get a generic
    GBM stand-in so the pipeline still runs.
    """
    n_lower = (name or "").lower()
    if "eur" in n_lower or "fx" in n_lower or strategy_id == "A":
        df, info = load_eurusd_1h()
    elif "qqq" in n_lower or "rsi" in n_lower or strategy_id == "B":
        df, info = load_qqq_15m()
    elif "vol" in n_lower or "martingale" in n_lower or strategy_id == "C":
        df, info = load_vol75_1m()
    else:
        # generic stand-in: 1h, sigma 0.20
        df = gbm_ohlcv(
            n_bars=24 * 30 * 3, bar_minutes=60, sigma_per_year=0.20, seed=900 + ord(strategy_id[0])
        )
        info = {
            "symbol": f"GENERIC-{strategy_id}",
            "interval": "1h",
            "preferred_source": "geometric_brownian_motion",
            "synthetic_fallback": True,
            "process": "geometric_brownian_motion",
            "drift_per_year": 0.0,
            "sigma_per_year": 0.20,
            "seed": 900 + ord(strategy_id[0]),
            "initial_price": 100.0,
            "n_bars": int(len(df)),
            "bar_minutes": 60,
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            "columns": list(df.columns),
            "fetched_at": now_iso(),
        }
    return df, info


