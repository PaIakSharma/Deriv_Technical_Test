# Comparative Brief — Risk-Adjusted Ranking

> **Analysis tooling, not financial advice.** All numbers describe historical or simulated behaviour over a finite window. Past performance does not generalise.


## Ranking

| # | Strategy | Sharpe | Profit Factor | Max DD | Trades | Risk Flag | Score |
|---|----------|-------:|--------------:|-------:|-------:|-----------|------:|
| 1 | B | 4.409 | 2.017942155895412 | 9.8975 | 8 | medium | 4.913 |
| 2 | C | 7.148 | 1.4333333333333333 | 31.0000 | 50 | **HIGH** ⚠ | 2.506 |
| 3 | A | 1.449 | 1.1302592985615492 | 0.0587 | 908 | low | 1.731 |

## Strongest and Weakest

- **Strongest (this sample):** B — Sharpe 4.409, PF 2.017942155895412, verdict `fragile`.
- **Weakest (this sample):** A — Sharpe 1.449, risk_level `low`.

## Risk-adjusted reasoning

- **B** — Sharpe 4.409, max DD 9.8975 on 8 trades. Verdict: *fragile*. Regime exposure: RSI mean-reversion entries depend on rangebound or mildly trending behaviour. In a strong downtrend RSI<25 prints repeatedly while price keeps falling, and entries get run over.
- **C** — Sharpe 7.148, max DD 31.0000 on 50 trades. Verdict: *fragile*. Regime exposure: Catastrophically dependent on never observing a long enough loss streak to breach the drawdown cap. Any regime with mean-reverting returns or higher autocorrelation in losses (e.g. trending down) acce
- **A** — Sharpe 1.449, max DD 0.0587 on 908 trades. Verdict: *robust*. Regime exposure: Opening-range breakout requires intraday continuation after the London open. Range-bound days with no follow-through generate stop-outs; gappy news mornings tend to over-trigger and reverse.

## Robustness warning

Sharpe and profit factor are extremely sensitive to the small set of largest moves in any sample. Walk-forward (`walk_forward.json`) and parameter sensitivity (`parameter_sensitivity.json`) are the more honest read of robustness — consult them before treating the ranking above as anything more than a snapshot of one window of one data series.

## Martingale / loss-escalation warning

- **C** uses loss-doubling sizing. This is a high-risk structure: cumulative risk grows as 2^k − 1 after k consecutive losses, so a single tail event wipes out many small winners. The in-sample win rate is **not informative** about ruin probability. Treat in-sample profitability as luck of the seed, not as edge.

## Retail-reader warning

If you are reading this as a retail trader: **this document is not a recommendation.** Strategies that look profitable on a few months of historical or simulated data routinely lose money in live conditions because of slippage, spreads, news gaps, broker max-stake limits, and the simple fact that you cannot replay history. None of the math in this report makes those real-world frictions go away.
