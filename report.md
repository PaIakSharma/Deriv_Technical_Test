# Strategy Backtest Report

> **This is analysis tooling, not financial advice.** All metrics describe historical or simulated behaviour over a finite window. They are not predictions.


## Pipeline summary

Stages enforced: STRATEGIES_LOADED → DATA_FETCHED_OR_SIMULATED → STRATEGIES_FORMALISED → SPECS_VALIDATED → BACKTESTS_EXECUTED → LEDGERS_WRITTEN → METRICS_COMPUTED → STRATEGIES_CRITIQUED → OPTIONAL_ROBUSTNESS_TESTS_COMPLETE → REPORT_GENERATED → VALIDATION_COMPLETE → RESULTS_FINALISED.

LLM stages are limited to **formalisation** (prose → JSON spec) and **critique** (post-hoc commentary). All numerical work is deterministic Python.


## Backtest assumptions

- **intrabar_ordering** — If both stop-loss and take-profit are touched in the same bar, the stop-loss is assumed to fill first.
- **fills** — Stops and targets fill at exactly the trigger price (no slippage). Market exits (EOD, RSI mean-revert) fill at the bar's close.
- **fees** — No commissions, spreads, or financing costs are modelled.
- **timezones** — All bar timestamps are normalised to UTC before backtest logic runs. Strategy-specific session windows interpret hours in UTC.
- **rsi_method** — Wilder smoothing (the original RSI formulation).
- **pip_size_eurusd** — 1 pip = 0.0001.
- **ny_close_assumed_utc** — 21:00 UTC.
- **london_open_assumed_utc** — 08:00 UTC (matches London local in winter).
- **vol75_payout** — Even-money: win = +stake, loss = -stake. Tie = loss.
- **size_units** — All trades sized in abstract 'units' unless the strategy specifies a stake (Strategy C uses real dollar stakes for its martingale).


## Data manifest

| Strategy | Symbol | Interval | Source | Synthetic Fallback | Bars | Start | End |
|---|---|---|---|---|---|---|---|
| A | EURUSD=X | 1h | yfinance | False | 17264 | 2023-07-18 23:00:00+00:00 | 2026-05-05 19:00:00+00:00 |
| B | QQQ | 15m | yfinance | False | 1428 | 2026-02-17 14:30:00+00:00 | 2026-05-05 19:15:00+00:00 |
| C | Vol75-synthetic | 1m | geometric_brownian_motion | True | 4320 | 2024-01-01 00:00:00+00:00 | 2024-01-03 23:59:00+00:00 |


## Strategies & ambiguities

### Strategy A — London Breakout EUR/USD

**Original description (verbatim):** _I trade EURUSD on the London open. Wait for the first hour after 8am London. Mark the high and low of that hour. If price breaks the high by 5 pips, go long with stop at the low and target 1.5x risk. Opposite for short. Skip Wednesdays cause of news. Close everything before NY close._

**Instrument / timeframe:** EURUSD on 1h (data source: `yfinance:EURUSD=X`).

**Explicit ambiguities (6):**
- *'8am London' is timezone-ambiguous: London is UTC+0 in winter (GMT) and UTC+1 in summer (BST). The strategy text does not specify how DST is handled.*
  - assumption used: Use 08:00 UTC year-round as the 'London open hour'. This matches London local time only in winter.
  - impact if wrong: Using 07:00 UTC during BST would shift the opening range by one hour and likely change which bar's high/low anchors the breakout, materially altering entry levels.
- *'NY close' is undefined: equity-market close is 21:00 UTC (winter) / 20:00 UTC (summer); FX 5pm ET rollover is 22:00 UTC / 21:00 UTC.*
  - assumption used: Close any open position by 21:00 UTC.
  - impact if wrong: An earlier or later cutoff changes how often trades exit at-market versus on stop/target, biasing the win-rate estimate.
- *The pip definition for EURUSD is not stated (standard pip = 0.0001 vs fractional/pipette = 0.00001).*
  - assumption used: 1 pip = 0.0001; 5 pips = 0.0005.
  - impact if wrong: If the trader meant pipettes, the breakout buffer is 10x smaller and the system would generate many more trades with much smaller risk units.
- *'Skip Wednesdays cause of news' is a coarse heuristic; the trader does not specify whether it should adapt to actual high-impact news events.*
  - assumption used: Skip every Wednesday unconditionally.
  - impact if wrong: A news-calendar-driven filter would skip fewer days but more selectively, likely improving risk-adjusted returns versus blanket Wednesday avoidance.
- *Position sizing is not described.*
  - assumption used: 1 unit per signal — PnL reported per unit; metrics are sizing-agnostic.
  - impact if wrong: Fixed-risk-per-trade sizing (e.g. 1% of equity at risk) would change drawdown geometry and Sharpe materially.
- *Intrabar ordering when both stop and target are touched in the same bar is not specified.*
  - assumption used: Conservative: assume the stop is hit first.
  - impact if wrong: Optimistic ordering (target first) would inflate win-rate by an amount proportional to the share of bars that touch both extremes.

**Session filters:**
- Trade only intraday bars after the 08:00 London opening hour.
- Skip all of Wednesday (news-day rule from the trader).
- Force-close any open position before 21:00 UTC (assumed 'NY close').

**Risk controls:**
- One position at a time.
- Stop-loss is mandatory and equals the opposite end of the opening range.

### Strategy B — RSI mean reversion on US tech

**Original description (verbatim):** _Watch QQQ on 15min. when RSI(14) drops under 25 buy half position, if it goes under 20 add the rest. exit at RSI 50 or end of day whichever first. dont trade in last 30 min. no shorts._

**Instrument / timeframe:** QQQ on 15m (data source: `yfinance:QQQ`).

**Explicit ambiguities (7):**
- *'RSI(14)' does not specify the smoothing method — Wilder's smoothing vs simple moving average vs exponential.*
  - assumption used: Wilder's RSI on close prices, the standard Welles Wilder original.
  - impact if wrong: SMA-based RSI is more reactive and would produce more sub-25 readings, likely increasing trade count and changing entry prices.
- *'Half position' is undefined. There is no notion of total capital, leverage, or risk-per-trade.*
  - assumption used: Half position = 0.5 units, full = 1.0 units. PnL is reported per unit.
  - impact if wrong: Real risk-based sizing (e.g. fixed % of equity) would change drawdown and Sharpe but not win-rate.
- *'End of day' is undefined: regular hours close 16:00 ET, extended hours run later.*
  - assumption used: Regular session close: last 15-min bar of the regular trading day.
  - impact if wrong: Holding into extended hours would expose the strategy to thin-liquidity gaps not modelled here.
- *The strategy does not say whether to re-enter the same day after an exit.*
  - assumption used: Allow re-entry after a full exit, subject to all other filters.
  - impact if wrong: A 'one trade per day' rule would cap exposure and mechanically lower trade count.
- *There is no stop-loss rule; the position is exposed indefinitely until RSI mean-reverts or EOD hits.*
  - assumption used: No stop-loss; exits only on RSI > 50 or EOD.
  - impact if wrong: Adding a max adverse excursion stop would cap tail-loss but also truncate winning recoveries.
- *Behaviour when RSI re-crosses 25→20→25→20 within the same position.*
  - assumption used: Second tranche fires only once per position; further crosses while in-position do nothing.
  - impact if wrong: A 'pyramid' interpretation would build larger and larger exposure on continued weakness.
- *[human review override]*
  - assumption used: For RSI(14) use Wilder smoothing on regular-session 15m bars only; treat the last 30-minute blackout as 16:00-15:30 ET inclusive.
  - impact if wrong: Recorded by a human reviewer at low-confidence prompt; supersedes the model's interpretation where applicable.

**Session filters:**
- No new entries within the last 30 minutes of the regular session.
- All positions force-closed at the regular session close.
- Long-only: shorts forbidden.

**Risk controls:**
- Maximum one position at a time (with up to two scale-in tranches).
- No stop-loss is specified — exits rely on RSI mean reversion or EOD.

### Strategy C — Synthetic Index Martingale (volatility 75)

**Original description (verbatim):** _On Volatility 75 Index 1min — fixed $1 stake, predict UP. If lose, double stake on next tick. Reset to $1 after a win. Stop trading session if drawdown > $200 OR after 50 trades, whichever first._

**Instrument / timeframe:** Volatility 75 Index (synthetic) on 1m (data source: `synthetic:gbm(sigma=0.75/yr,seed=123)`).

**Explicit ambiguities (5):**
- *Payout assumption is not stated. Real Vol75 binary contracts pay roughly 0.95x stake on a win.*
  - assumption used: Even-money payout: win = +stake, loss = -stake.
  - impact if wrong: A realistic 0.95x payout would erode expectancy further and accelerate ruin.
- *'Tick' is undefined. Real Vol75 ticks every 2 seconds; the description also mentions '1min'.*
  - assumption used: Treat each 1-minute bar as one trade; outcome is sign of next-bar close minus current close.
  - impact if wrong: Higher tick frequency packs more trades per session and brings the cumulative-loss-streak event horizon closer.
- *Drawdown definition: peak-to-trough on equity, or session-cumulative loss?*
  - assumption used: Session-cumulative PnL falling below -$200 triggers the stop.
  - impact if wrong: Peak-to-trough drawdown would trip the stop sooner if the session had any winning streak before the eventual losing run.
- *What counts toward the 50-trade cap — entries placed or trades fully settled?*
  - assumption used: Settled trades count.
  - impact if wrong: Counting placements would change the off-by-one boundary at the cap but not the qualitative outcome.
- *'Predict UP' tie-break: is next_close == current_close a win or a loss?*
  - assumption used: Tie counts as a loss (strict greater-than for a win).
  - impact if wrong: Counting ties as wins would slightly improve the empirical win-rate but doesn't change the ruin geometry.

**Session filters:**
- Single continuous session (no day-of-week or hour filters specified).

**Risk controls:**
- Hard cap: stop session if drawdown > $200.
- Hard cap: stop session after 50 trades.


## Performance metrics (computed deterministically)

| Strategy | Trades | Total PnL | Win Rate | Profit Factor | Max DD | Sharpe | Sortino | Exposure % | Max Loss Streak |
|---|---|---|---|---|---|---|---|---|---|
| A | 908 | 0.0989 | 47.36% | 1.130 | 0.0587 | 1.449 | 1.176 | 20.85% | 9 |
| B | 8 | 10.0750 | 75.00% | 2.018 | 9.8975 | 4.409 | 0.509 | 9.03% | 2 |
| C | 50 | 26.0000 | 52.00% | 1.433 | 31.0000 | 7.148 | 1.334 | 1.16% | 5 |


## Critiques

### Strategy A — risk: **LOW** (verdict: *robust*)

- **Overfitting risk:** Fixed 5-pip breakout buffer, fixed 1.5R target, and the Wednesday skip are unjustified by the data presented. Performance could be sensitive to all three.
- **Regime dependence:** Opening-range breakout requires intraday continuation after the London open. Range-bound days with no follow-through generate stop-outs; gappy news mornings tend to over-trigger and reverse.
- **Assumption sensitivity:** Most sensitive to the timezone and 'end of day' assumptions; secondary sensitivity to intrabar ordering between stop and target.
- **Execution realism:** Backtest assumes perfect fills at trigger prices with no slippage and no spread cost. Real intraday fills, especially around news, would widen effective stops.
- **Ruin risk:** _not applicable: this strategy does not use loss-escalating sizing._
- **Why high win rate may be misleading:** _not applicable: win rate is not used as the primary edge claim._
- **Likely failure modes:**
  - Range days: the price re-enters the opening range and stops the trade out at the opposite extreme.
  - Real news on non-Wednesday days bypasses the day-of-week filter.
  - Wide opening-hour ranges produce stop distances that dominate the 1.5R target.
- **Warnings:**
  - Sample size is limited to the available history window; conclusions do not generalise.
  - No transaction costs are modelled; net returns will be lower in practice.

### Strategy B — risk: **MEDIUM** (verdict: *fragile*)

- **Overfitting risk:** Thresholds 25 / 20 / 50 are the textbook defaults; the rule is not parameter-tuned to this sample, but choice of those exact levels is convention rather than evidence.
- **Regime dependence:** RSI mean-reversion entries depend on rangebound or mildly trending behaviour. In a strong downtrend RSI<25 prints repeatedly while price keeps falling, and entries get run over.
- **Assumption sensitivity:** Most sensitive to the timezone and 'end of day' assumptions; secondary sensitivity to intrabar ordering between stop and target.
- **Execution realism:** Backtest assumes perfect fills at trigger prices with no slippage and no spread cost. Real intraday fills, especially around news, would widen effective stops.
- **Ruin risk:** _not applicable: this strategy does not use loss-escalating sizing._
- **Why high win rate may be misleading:** _not applicable: win rate is not used as the primary edge claim._
- **Likely failure modes:**
  - Persistent downtrends — RSI oversold readings stay oversold while price keeps dropping.
  - Gap-down opens around earnings or macro news — no stop-loss is defined.
  - Whipsaw days where RSI bounces 25 → 50 → 25 generate frequent in-and-out trades.
- **Warnings:**
  - Sample size is limited to the available history window; conclusions do not generalise.
  - No transaction costs are modelled; net returns will be lower in practice.

### Strategy C — risk: **HIGH** (verdict: *fragile*)

> ⚠ **HIGH RISK — flagged by the pipeline.**

> ⚠ **MARTINGALE / LOSS-ESCALATING SIZING.** A high in-sample win rate is consistent with strongly negative skew and ruin risk.

- **Overfitting risk:** Not classical curve-fitting risk because the rule has no parameters tuned on data, but the empirical 'profitability' over short windows is an artefact of the specific loss-streak distribution observed, not a stable edge.
- **Regime dependence:** Catastrophically dependent on never observing a long enough loss streak to breach the drawdown cap. Any regime with mean-reverting returns or higher autocorrelation in losses (e.g. trending down) accelerates ruin.
- **Assumption sensitivity:** Most sensitive to (a) payout assumption — even-money was assumed; real Vol75 binaries pay <1.0x, which makes expected value strictly negative; and (b) tick frequency — at 2s tick rate the trader sees ~30x more events per session, packing the loss-streak event into a much smaller wall-clock window.
- **Execution realism:** Backtest assumes perfect fills at bar close, no slippage, no platform latency, and no platform-side stake limits. Real martingale runs hit broker max-stake caps before the drawdown cap, which converts a paper -$200 cap into an unrecoverable position.
- **Ruin risk:** Stake doubling produces cumulative risk 2^k - 1 after k consecutive losses. To breach the $200 cap requires only 8 losses in a row — at a 50% win rate that has probability 1/256 per any 8-length window, so over even a moderate session the cumulative probability of ruin is non-trivial and approaches 1 as session length grows. Backtest measured max_dd=31.00 on this seed but a different seed could exceed it without changing any rule.
- **Why high win rate may be misleading:** With even-money payouts and 50% win probability the expectancy of each trade is exactly zero. The strategy wins ~50% of trades in the sample (n=50), but each of those wins yields the small reset stake while a single 8-loss run loses 255x that stake. A high win-rate accounting therefore disguises strongly negative skew.
- **Likely failure modes:**
  - 8+ consecutive losses → cumulative -$255 exceeds -$200 cap; session ends in ruin.
  - Broker max-stake cap hits before doubling completes; loss can't be recovered.
  - Sub-1.0 payout makes expectancy negative even with 50% win rate.
  - Path dependency: P(losing streak ≥ k) grows with session length.
- **Warnings:**
  - MARTINGALE / LOSS-DOUBLING — high risk of ruin regardless of in-sample win rate.
  - Geometric stake growth means a single tail event wipes out many small winners.
  - Performance metrics are misleading: Sharpe is dominated by the absence of the tail event in this short sample.


## Walk-forward stability

| Strategy | Window 0 PnL | Window 1 PnL | Window 2 PnL | Stability |
|---|---:|---:|---:|---|
| A | 0.0432 | 0.0353 | 0.0203 | **degrading** |
| B | 12.0899 | -4.4874 | 2.4725 | **unstable** |
| C | 26.0000 | 28.0000 | 29.0000 | **stable** |


## Parameter sensitivity

### Strategy A — sweep over `breakout_pips`

| breakout_pips | total_pnl | sharpe | trades | max_dd |
|---:|---:|---:|---:|---:|
| 2.500 | 0.0992 | 1.519 | 1005 | 0.0552 |
| 3.500 | 0.0858 | 1.283 | 972 | 0.0530 |
| 4.500 | 0.0738 | 1.094 | 924 | 0.0587 |
| 5.500 | 0.0738 | 1.081 | 891 | 0.0481 |
| 6.500 | 0.0674 | 0.973 | 862 | 0.0570 |
| 7.500 | 0.0813 | 1.172 | 818 | 0.0492 |

*Interpretation:* Spread between best and worst PnL across the breakout_pips sweep is 0.0318. This is large relative to the central value and indicates that the original choice was not robust in this sample.
*Stability:* non-monotonic: PnL has at least one local optimum, hinting at parameter-space cliffs.

### Strategy B — sweep over `rsi_entry_threshold`

| rsi_entry_threshold | total_pnl | sharpe | trades | max_dd |
|---:|---:|---:|---:|---:|
| 12.500 | -2.1100 | -4.954 | 1 | 2.1100 |
| 17.500 | -1.5762 | -1.249 | 3 | 5.5012 |
| 22.500 | 6.2101 | 2.921 | 7 | 8.8300 |
| 27.500 | 17.5952 | 5.970 | 13 | 11.4949 |
| 32.500 | 20.9379 | 5.249 | 19 | 15.0944 |
| 37.500 | 17.9915 | 3.985 | 27 | 26.7394 |

*Interpretation:* Spread between best and worst PnL across the rsi_entry_threshold sweep is 23.0479. This is large relative to the central value and indicates that the original choice was not robust in this sample.
*Stability:* non-monotonic: PnL has at least one local optimum, hinting at parameter-space cliffs.


## Adversarial scenarios

### Strategy A

- **trendless_chop** — Low-vol mean-reverting day-after-day; opening ranges trigger breakouts that immediately reverse.
  - n_trades=61, total_pnl=0.0092, max_dd=0.0063, max_loss_streak=4
  - expected: Both long and short breakouts stop out repeatedly; max losing streak grows. | observed: strategy survived (total_pnl=0.0092, n_trades=61, max_dd=0.0063)
- **news_gap_open** — Wednesday-skipped, but a Tuesday morning shock 50 bars in expands range and gaps through stops.
  - n_trades=89, total_pnl=0.0185, max_dd=0.0080, max_loss_streak=5
  - expected: Long breakout trigger hits, then sudden reversal blows through stop with slippage we don't model. | observed: strategy survived (total_pnl=0.0185, n_trades=89, max_dd=0.0080)
- **single_direction_grind** — Low-vol persistent uptrend; opening ranges always break upward — but 1.5R targets miss because of small bar bodies.
  - n_trades=69, total_pnl=0.0322, max_dd=0.0056, max_loss_streak=4
  - expected: Trades enter long but EOD-exit before targets, capping wins. | observed: strategy survived (total_pnl=0.0322, n_trades=69, max_dd=0.0056)

### Strategy B

- **persistent_downtrend** — Strong downtrend keeps RSI under 25; entries stack and ride the trend down.
  - n_trades=6, total_pnl=0.0503, max_dd=8.1219, max_loss_streak=2
  - expected: Repeated full-size entries with no stop-loss; large unrealised drawdown until EOD. | observed: strategy survived (total_pnl=0.0503, n_trades=6, max_dd=8.1219)
- **earnings_cliff** — Quiet drift then a single -8% shock 100 bars in.
  - n_trades=7, total_pnl=1.1169, max_dd=0.7042, max_loss_streak=3
  - expected: If a position is open through the shock, no stop catches it; large biggest_loss. | observed: strategy survived (total_pnl=1.1169, n_trades=7, max_dd=0.7042)
- **rsi_whipsaw** — High vol mean-reverting around the entry threshold.
  - n_trades=3, total_pnl=7.0518, max_dd=0.0000, max_loss_streak=0
  - expected: Many small frictionless trades but no consistent edge; profit factor near 1. | observed: strategy survived (total_pnl=7.0518, n_trades=3, max_dd=0.0000)

### Strategy C

- **long_loss_streak** — Slight negative drift produces clusters of losing predictions.
  - n_trades=50, total_pnl=21.0000, max_dd=63.0000, max_loss_streak=6
  - expected: 8+ consecutive losses breach the $200 drawdown cap; ruin. | observed: strategy survived (total_pnl=21.0000, n_trades=50, max_dd=63.0000)
- **high_vol_random** — Pure random walk at higher sigma; tail of loss streaks gets fatter.
  - n_trades=50, total_pnl=-36.0000, max_dd=63.0000, max_loss_streak=6
  - expected: Probability of an 8-loss streak in 50 trades rises; ruin frequency increases. | observed: net loss of -36.0000 over 50 trades; max losing streak 6, max DD 63.0000
- **downward_shock_then_recover** — Stable random walk with a single concentrated 5-bar down move at bar 20.
  - n_trades=50, total_pnl=22.0000, max_dd=127.0000, max_loss_streak=7
  - expected: Shock window forces 5 consecutive losses, doubling stake to $32; subsequent random losses can finish the ruin. | observed: strategy survived (total_pnl=22.0000, n_trades=50, max_dd=127.0000)


## High-risk strategies (pipeline-flagged)

- **C** — flagged HIGH RISK. MARTINGALE / LOSS-DOUBLING — high risk of ruin regardless of in-sample win rate.


## Retail-reader warning

If you are not a quantitative analyst with risk-management infrastructure: do not trade strategies based on output like this. In-sample metrics over a few months of data, computed against historical or simulated paths with no slippage, no broker stake limits, and no spread, are NOT representative of live trading. Martingale-style strategies in particular are mathematically prone to ruin regardless of any in-sample win rate.
