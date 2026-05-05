# Strategy Formalisation & Backtest Pipeline

A staged, replayable pipeline that converts informal retail-trader strategy
descriptions into strict executable specifications, runs deterministic
backtests against real or simulated OHLCV data, computes risk-adjusted
performance metrics in code, and uses an LLM to write a robustness critique
*after* the numbers are already known.

> **This is analysis tooling, not financial advice.** The metrics produced
> here describe historical or simulated behaviour and say nothing about future
> performance. Martingale-style strategies are flagged as high-risk because
> they are mathematically prone to ruin regardless of in-sample win rates.

## Design

```
INIT
 -> STRATEGIES_LOADED
 -> DATA_FETCHED_OR_SIMULATED
 -> STRATEGIES_FORMALISED          (Stage 1 LLM call, one per strategy)
 -> SPECS_VALIDATED
 -> BACKTESTS_EXECUTED             (deterministic Python only)
 -> LEDGERS_WRITTEN
 -> METRICS_COMPUTED               (deterministic Python only)
 -> STRATEGIES_CRITIQUED           (Stage 2 LLM call, one per strategy)
 -> OPTIONAL_ROBUSTNESS_TESTS_COMPLETE
 -> REPORT_GENERATED
 -> VALIDATION_COMPLETE
 -> RESULTS_FINALISED
```

Each transition is enforced by `src/state.py`. Trying to skip a stage raises
`PipelineStateError`, so by construction the report cannot be produced before
metrics exist, and metrics cannot be produced before ledgers exist.

The two LLM stages do **only** natural-language work:

| Stage           | LLM job                                          | LLM is *not* allowed to ... |
| --------------- | ------------------------------------------------ | --------------------------- |
| Formalisation   | Translate prose → JSON spec, surface ambiguities | Compute returns, predict performance |
| Critique        | Discuss robustness, regime risk, failure modes   | Compute Sharpe, win rate, drawdown — those are pre-computed and passed in |

All numerical work (entry/exit simulation, returns, Sharpe, Sortino, profit
factor, drawdown, exposure) lives in pure deterministic Python in
`src/backtest.py`, `src/strategies.py` and `src/metrics.py`.

## Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. configure (copy .env.example, then edit)
cp .env.example .env
# put your OpenRouter key in OPENROUTER_API_KEY, OR set USE_MOCK_LLM=1

# 3. run the staged pipeline
python pipeline.py

# 4. check that every required invariant holds
python validate.py
```

`make all` runs the pipeline followed by validation.

### LLM provider

This project calls **OpenRouter** through its OpenAI-compatible REST API.
Default model is `anthropic/claude-3.5-sonnet`; override with `LLM_MODEL`.

If `OPENROUTER_API_KEY` is missing the pipeline transparently falls back to a
deterministic *mock* LLM that produces hand-crafted (but plausible and
schema-valid) responses for the public fixture. Both modes write to
`llm_calls.jsonl` with the same record shape, so the validator cannot tell
them apart structurally — only the `provider` field is different.

The mock mode exists so the pipeline runs end-to-end on a clean checkout
with no network. For an evaluator who wants real LLM output, set the API key.

### Data

| Strategy | Source                                  | Fallback                          |
| -------- | --------------------------------------- | --------------------------------- |
| A        | `EURUSD=X` 1h bars from `yfinance`      | seeded GBM (sigma 0.07/yr)        |
| B        | `QQQ` 15m bars from `yfinance`          | seeded GBM (sigma 0.20/yr)        |
| C        | seeded GBM (sigma 0.75/yr), 1m bars     | (always synthetic)                |

If `yfinance` cannot reach the network, the relevant series is replaced with
seeded GBM and `synthetic_fallback: true` is recorded for that entry in
`data_manifest.json`. The pipeline still completes deterministically.

## Artifacts

After a successful run the working directory contains:

```
strategies.json              # input
data_manifest.json           # data sources, seeds, GBM params
specs/{A,B,C}.json           # Stage 1 output (one per strategy)
ledgers/{A,B,C}.csv          # per-trade ledgers from the backtest
metrics.json                 # deterministic performance metrics
critiques.json               # Stage 2 LLM output
walk_forward.json            # 3-window stability check
parameter_sensitivity.json   # ±50% sweeps + LLM interpretation
adversarial_scenarios.json   # LLM-proposed stress paths + their backtests
comparative_brief.md         # ranked one-page brief
report.md                    # full human-readable report
llm_calls.jsonl              # one record per LLM call (formalisation + critique + extras)
```

## Validation

`validate.py` enforces every requirement listed in the spec, including:

* required artifacts exist and parse as valid JSON;
* every strategy has a formal spec with at least 3 substantive ambiguities;
* ledger totals reconcile with the summary `metrics.json` (≤ $0.01 / 0.01% PnL drift);
* Strategy C is flagged high-risk in `critiques.json` and `report.md`;
* `llm_calls.jsonl` contains separate records for the formalisation and
  critique of every strategy.

It exits non-zero on any failure.

## Replacing the fixture

The evaluator may swap `strategies.json` for an equivalent set. The pipeline
does not hard-code strategy IDs, count, ordering, or instrument names — it
loops over whatever it finds. The execution-template dispatcher inspects the
formalised spec to choose a backtest implementation; if none of the known
templates match, it falls back to a generic interpreter and records the
template choice in the spec for transparency.

## What this pipeline is *not*

* It is not a live-trading system. There is no broker integration, no fills
  realism, no market impact, no slippage model beyond the documented
  intrabar ordering assumption.
* It is not a strategy-discovery engine. It evaluates strategies that humans
  describe; it does not search for new ones.
* It does not give financial advice. It only describes historical /
  simulated behaviour with explicit caveats.
