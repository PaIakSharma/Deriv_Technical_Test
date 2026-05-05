"""Run the full staged pipeline.

    INIT
     -> STRATEGIES_LOADED
     -> DATA_FETCHED_OR_SIMULATED
     -> STRATEGIES_FORMALISED          (Stage 1 LLM call)
     -> SPECS_VALIDATED
     -> BACKTESTS_EXECUTED             (deterministic Python only)
     -> LEDGERS_WRITTEN
     -> METRICS_COMPUTED               (deterministic Python only)
     -> STRATEGIES_CRITIQUED           (Stage 2 LLM call)
     -> OPTIONAL_ROBUSTNESS_TESTS_COMPLETE
     -> REPORT_GENERATED
     -> VALIDATION_COMPLETE
     -> RESULTS_FINALISED

Stage transitions are enforced by ``src.state.Pipeline``; trying to skip
fails loudly.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.adversarial import run_adversarial
from src.backtest import BACKTEST_ASSUMPTIONS
from src.comparative import build_comparative_brief
from src.critique import critique_one
from src.data_loader import cache_csv, load_for_strategy
from src.formaliser import SpecValidationError, formalise_one, validate_spec
from src.human_review import reset_log as reset_human_review_log
from src.llm_client import LLMClient
from src.metrics import compute_metrics, reconcile_ledger
from src.report import build_report
from src.sensitivity import run_sensitivity
from src.state import Pipeline, Stage
from src.strategies import run_backtest
from src.utils import read_json, write_json
from src.walk_forward import walk_forward_one


def _print_stage(label: str) -> None:
    print(f"\n=== {label} ===", flush=True)


def main() -> int:
    pipeline = Pipeline()
    llm = LLMClient(log_path="llm_calls.jsonl")
    llm.reset_log()  # one fresh log per run
    reset_human_review_log()

    # ---------------- STAGE 1: load strategies
    _print_stage("STRATEGIES_LOADED")
    strategies: List[Dict[str, Any]] = read_json("strategies.json")
    if not isinstance(strategies, list) or not strategies:
        print("[fatal] strategies.json must be a non-empty list.", flush=True)
        return 2
    pipeline.advance(Stage.STRATEGIES_LOADED)
    print(f"  loaded {len(strategies)} strategies: {[s['id'] for s in strategies]}", flush=True)

    # ---------------- STAGE 2: data
    _print_stage("DATA_FETCHED_OR_SIMULATED")
    data_by_id: Dict[str, pd.DataFrame] = {}
    manifest: Dict[str, Any] = {}
    Path("data").mkdir(parents=True, exist_ok=True)
    for s in strategies:
        sid = s["id"]
        df, info = load_for_strategy(sid, s.get("name", ""))
        data_by_id[sid] = df
        manifest[sid] = info
        cache_csv(df, Path("data") / f"{sid}.csv")
        print(f"  [{sid}] {info.get('symbol')} {info.get('interval')} bars={info.get('n_bars')} "
              f"synthetic_fallback={info.get('synthetic_fallback')}", flush=True)
    write_json("data_manifest.json", manifest)
    pipeline.advance(Stage.DATA_FETCHED_OR_SIMULATED)

    # ---------------- STAGE 3: formalise (Stage 1 LLM call x N)
    _print_stage("STRATEGIES_FORMALISED")
    specs: Dict[str, Dict[str, Any]] = {}
    for s in strategies:
        sid = s["id"]
        print(f"  [{sid}] calling LLM for formalisation...", flush=True)
        try:
            spec = formalise_one(strategy=s, llm=llm)
            specs[sid] = spec
        except Exception as e:  # noqa: BLE001
            print(f"[fatal] formalisation failed for {sid}: {e}", flush=True)
            traceback.print_exc()
            return 3
    pipeline.advance(Stage.STRATEGIES_FORMALISED)

    # ---------------- STAGE 4: validate specs
    _print_stage("SPECS_VALIDATED")
    for sid, spec in specs.items():
        problems = validate_spec(spec)
        if problems:
            print(f"[fatal] spec {sid} failed validation:", flush=True)
            for p in problems:
                print(f"   - {p}", flush=True)
            raise SpecValidationError(f"spec {sid} invalid")
        print(f"  [{sid}] spec OK ({len(spec.get('explicit_ambiguities', []))} ambiguities)", flush=True)
    pipeline.advance(Stage.SPECS_VALIDATED)

    # ---------------- STAGE 5: backtests
    _print_stage("BACKTESTS_EXECUTED")
    results = {}
    for sid, spec in specs.items():
        df = data_by_id[sid]
        result = run_backtest(spec, df)
        results[sid] = result
        print(f"  [{sid}] {len(result.trades)} trades on {result.bars_total} bars", flush=True)
    pipeline.advance(Stage.BACKTESTS_EXECUTED)

    # ---------------- STAGE 6: ledgers
    _print_stage("LEDGERS_WRITTEN")
    Path("ledgers").mkdir(parents=True, exist_ok=True)
    for sid, result in results.items():
        rows = [t.as_row() for t in result.trades]
        ledger_path = Path("ledgers") / f"{sid}.csv"
        if rows:
            pd.DataFrame(rows, columns=[
                "strategy_id", "entry_time", "exit_time", "direction",
                "entry_price", "exit_price", "size", "pnl", "return_pct", "exit_reason",
            ]).to_csv(ledger_path, index=False)
        else:
            ledger_path.write_text(
                "strategy_id,entry_time,exit_time,direction,entry_price,exit_price,size,pnl,return_pct,exit_reason\n",
                encoding="utf-8",
            )
        print(f"  [{sid}] ledger -> {ledger_path}", flush=True)
    pipeline.advance(Stage.LEDGERS_WRITTEN)

    # ---------------- STAGE 7: metrics
    _print_stage("METRICS_COMPUTED")
    metrics_list: List[Dict[str, Any]] = []
    for sid, result in results.items():
        m = compute_metrics(result)
        metrics_list.append(m)
        print(f"  [{sid}] sharpe={m['sharpe_annualised']:.3f} pf={m['profit_factor']} "
              f"max_dd={m['max_drawdown_pnl_units']:.4f} n_trades={m['n_trades']}", flush=True)
    write_json("metrics.json", metrics_list)
    # Reconcile ledger -> metrics now (also re-checked by validate.py)
    for sid in specs:
        problems = reconcile_ledger(f"ledgers/{sid}.csv", next(m for m in metrics_list if m["strategy_id"] == sid))
        if problems:
            print(f"[fatal] reconciliation failed for {sid}: {problems}", flush=True)
            return 4
    pipeline.advance(Stage.METRICS_COMPUTED)

    # ---------------- STAGE 8: critique (Stage 2 LLM call x N)
    _print_stage("STRATEGIES_CRITIQUED")
    critiques: List[Dict[str, Any]] = []
    for sid in specs:
        result = results[sid]
        ledger_df = pd.read_csv(f"ledgers/{sid}.csv") if Path(f"ledgers/{sid}.csv").exists() else pd.DataFrame()
        m = next(x for x in metrics_list if x["strategy_id"] == sid)
        print(f"  [{sid}] calling LLM for critique...", flush=True)
        c = critique_one(
            spec=specs[sid],
            metrics=m,
            equity_curve=result.equity_curve,
            ledger_df=ledger_df,
            llm=llm,
        )
        critiques.append(c)
    write_json("critiques.json", critiques)
    pipeline.advance(Stage.STRATEGIES_CRITIQUED)

    # ---------------- STAGE 9: optional robustness tests
    _print_stage("OPTIONAL_ROBUSTNESS_TESTS_COMPLETE")
    # 5. walk-forward
    walk_forward_results = []
    for sid, spec in specs.items():
        wf = walk_forward_one(spec, data_by_id[sid])
        walk_forward_results.append(wf)
        print(f"  [{sid}] walk_forward stability={wf['stability']}", flush=True)
    write_json("walk_forward.json", walk_forward_results)

    # 6. parameter sensitivity (only for templates that have a sweep)
    sensitivity_results = []
    for sid, spec in specs.items():
        sr = run_sensitivity(spec=spec, ohlcv=data_by_id[sid], llm=llm)
        sensitivity_results.append(sr)
        if sr.get("skipped"):
            print(f"  [{sid}] sensitivity skipped: {sr.get('reason')}", flush=True)
        else:
            print(f"  [{sid}] sensitivity sweep over {sr['parameter']} ({len(sr['table'])} pts)", flush=True)
    write_json("parameter_sensitivity.json", sensitivity_results)

    # 7. adversarial
    adversarial_results = []
    for sid, spec in specs.items():
        ar = run_adversarial(spec=spec, base_ohlcv=data_by_id[sid], llm=llm)
        adversarial_results.append(ar)
        print(f"  [{sid}] adversarial scenarios = {len(ar.get('scenarios', []))}", flush=True)
    write_json("adversarial_scenarios.json", adversarial_results)

    # 8. comparative brief (deterministic; uses metrics+critiques on disk)
    build_comparative_brief()
    print("  comparative_brief.md written.", flush=True)

    pipeline.advance(Stage.OPTIONAL_ROBUSTNESS_TESTS_COMPLETE)

    # ---------------- STAGE 10: report
    _print_stage("REPORT_GENERATED")
    build_report()
    print("  report.md written.", flush=True)
    pipeline.advance(Stage.REPORT_GENERATED)

    # ---------------- STAGE 11: validation (delegate to validate.py)
    _print_stage("VALIDATION_COMPLETE")
    import importlib
    sys.path.insert(0, str(Path(__file__).parent))
    validate_mod = importlib.import_module("validate")
    rc, problems = validate_mod.run_validation()
    if rc != 0:
        print(f"[validate] failures ({len(problems)}):", flush=True)
        for p in problems:
            print(f"  - {p}", flush=True)
        return rc
    pipeline.advance(Stage.VALIDATION_COMPLETE)

    # ---------------- STAGE 12: finalise
    _print_stage("RESULTS_FINALISED")
    pipeline.advance(Stage.RESULTS_FINALISED)
    print(f"  pipeline reached {pipeline} via {len(pipeline.history)} transitions.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
