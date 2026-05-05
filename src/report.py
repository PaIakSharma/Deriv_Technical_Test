"""Final human-readable ``report.md`` generator.

Pulls everything together: specs (with ambiguities), backtest assumptions,
metrics, critiques, optional walk-forward / sensitivity / adversarial /
comparative results.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backtest import BACKTEST_ASSUMPTIONS
from .utils import read_json


def _safe_read(path: str | Path) -> Optional[Any]:
    try:
        return read_json(path)
    except FileNotFoundError:
        return None


def _format_metrics_table(metrics: List[Dict[str, Any]]) -> str:
    if not metrics:
        return "_no metrics available_\n"
    headers = [
        "Strategy", "Trades", "Total PnL", "Win Rate", "Profit Factor",
        "Max DD", "Sharpe", "Sortino", "Exposure %", "Max Loss Streak",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for m in metrics:
        lines.append("| " + " | ".join([
            m["strategy_id"],
            str(m["n_trades"]),
            f"{m['total_pnl']:.4f}",
            f"{m['win_rate']:.2%}",
            str(m["profit_factor"]) if m["profit_factor"] == "inf" else f"{m['profit_factor']:.3f}",
            f"{m['max_drawdown_pnl_units']:.4f}",
            f"{m['sharpe_annualised']:.3f}",
            f"{m['sortino_annualised']:.3f}",
            f"{m['exposure_pct']:.2%}",
            str(m["largest_losing_streak"]),
        ]) + " |")
    return "\n".join(lines) + "\n"


def build_report(output_path: str | Path = "report.md") -> str:
    strategies = _safe_read("strategies.json") or []
    manifest = _safe_read("data_manifest.json") or {}
    metrics = _safe_read("metrics.json") or []
    critiques = _safe_read("critiques.json") or []
    walk_forward = _safe_read("walk_forward.json")
    sensitivity = _safe_read("parameter_sensitivity.json")
    adversarial = _safe_read("adversarial_scenarios.json")

    crit_by_id = {c["strategy_id"]: c for c in critiques}
    metrics_by_id = {m["strategy_id"]: m for m in metrics}

    lines: List[str] = []
    lines.append("# Strategy Backtest Report\n")
    lines.append("> **This is analysis tooling, not financial advice.** All metrics describe historical or simulated behaviour over a finite window. They are not predictions.\n")
    lines.append("\n## Pipeline summary\n")
    lines.append(
        "Stages enforced: STRATEGIES_LOADED → DATA_FETCHED_OR_SIMULATED → STRATEGIES_FORMALISED "
        "→ SPECS_VALIDATED → BACKTESTS_EXECUTED → LEDGERS_WRITTEN → METRICS_COMPUTED → "
        "STRATEGIES_CRITIQUED → OPTIONAL_ROBUSTNESS_TESTS_COMPLETE → REPORT_GENERATED → "
        "VALIDATION_COMPLETE → RESULTS_FINALISED."
    )
    lines.append("\nLLM stages are limited to **formalisation** (prose → JSON spec) and **critique** (post-hoc commentary). All numerical work is deterministic Python.\n")

    lines.append("\n## Backtest assumptions\n")
    for k, v in BACKTEST_ASSUMPTIONS.items():
        lines.append(f"- **{k}** — {v}")
    lines.append("")

    lines.append("\n## Data manifest\n")
    if manifest:
        lines.append("| Strategy | Symbol | Interval | Source | Synthetic Fallback | Bars | Start | End |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for sid, info in manifest.items():
            lines.append(
                f"| {sid} | {info.get('symbol', '?')} | {info.get('interval', '?')} | "
                f"{info.get('preferred_source', '?')} | {info.get('synthetic_fallback', False)} | "
                f"{info.get('n_bars', '?')} | {info.get('start', '?')} | {info.get('end', '?')} |"
            )
        lines.append("")
    else:
        lines.append("_no manifest_\n")

    lines.append("\n## Strategies & ambiguities\n")
    for s in strategies:
        sid = s["id"]
        spec = _safe_read(Path("specs") / f"{sid}.json") or {}
        lines.append(f"### Strategy {sid} — {s.get('name', '')}\n")
        lines.append(f"**Original description (verbatim):** _{s['description']}_\n")
        lines.append(
            f"**Instrument / timeframe:** {spec.get('instrument', '?')} on "
            f"{spec.get('timeframe', '?')} (data source: `{spec.get('data_source', '?')}`).\n"
        )
        ambs = spec.get("explicit_ambiguities", []) or []
        if ambs:
            lines.append(f"**Explicit ambiguities ({len(ambs)}):**")
            for a in ambs:
                lines.append(f"- *{a.get('ambiguity', '')}*")
                lines.append(f"  - assumption used: {a.get('assumption_used_for_backtest', '')}")
                lines.append(f"  - impact if wrong: {a.get('impact_if_different', '')}")
        sf = spec.get("session_filters") or []
        if sf:
            lines.append("\n**Session filters:**")
            for f in sf:
                lines.append(f"- {f}")
        rc = spec.get("risk_controls") or []
        if rc:
            lines.append("\n**Risk controls:**")
            for r in rc:
                lines.append(f"- {r}")
        lines.append("")

    lines.append("\n## Performance metrics (computed deterministically)\n")
    lines.append(_format_metrics_table(metrics))

    lines.append("\n## Critiques\n")
    for c in critiques:
        sid = c["strategy_id"]
        lines.append(f"### Strategy {sid} — risk: **{c.get('risk_level', 'unknown').upper()}** "
                     f"(verdict: *{c.get('robustness_verdict', 'n/a')}*)\n")
        if c.get("is_high_risk"):
            lines.append("> ⚠ **HIGH RISK — flagged by the pipeline.**\n")
        if c.get("is_martingale_or_loss_escalating"):
            lines.append("> ⚠ **MARTINGALE / LOSS-ESCALATING SIZING.** A high in-sample win rate is consistent with strongly negative skew and ruin risk.\n")
        for key, label in [
            ("overfitting_risk", "Overfitting risk"),
            ("regime_dependence", "Regime dependence"),
            ("assumption_sensitivity", "Assumption sensitivity"),
            ("execution_realism", "Execution realism"),
            ("ruin_risk_discussion", "Ruin risk"),
            ("high_win_rate_misleading_explanation", "Why high win rate may be misleading"),
        ]:
            v = c.get(key)
            if v and not (isinstance(v, str) and v.lower().startswith("not applicable") and key in ("ruin_risk_discussion", "high_win_rate_misleading_explanation")):
                lines.append(f"- **{label}:** {v}")
            elif v and isinstance(v, str) and v.lower().startswith("not applicable"):
                lines.append(f"- **{label}:** _{v}_")
        failures = c.get("likely_failure_modes") or []
        if failures:
            lines.append("- **Likely failure modes:**")
            for fmode in failures:
                lines.append(f"  - {fmode}")
        warnings = c.get("warnings") or []
        if warnings:
            lines.append("- **Warnings:**")
            for w in warnings:
                lines.append(f"  - {w}")
        lines.append("")

    if walk_forward:
        lines.append("\n## Walk-forward stability\n")
        lines.append("| Strategy | Window 0 PnL | Window 1 PnL | Window 2 PnL | Stability |")
        lines.append("|---|---:|---:|---:|---|")
        for entry in walk_forward:
            wm = entry.get("windows", [])
            pnls = [(w.get("metrics") or {}).get("total_pnl") for w in wm]
            lines.append(
                f"| {entry['strategy_id']} | "
                + " | ".join(f"{p:.4f}" if isinstance(p, (int, float)) else "n/a" for p in pnls)
                + f" | **{entry.get('stability', '?')}** |"
            )
        lines.append("")

    if sensitivity:
        lines.append("\n## Parameter sensitivity\n")
        for entry in sensitivity:
            if entry.get("skipped"):
                continue
            sid = entry["strategy_id"]
            param = entry["parameter"]
            interp = entry.get("interpretation", {})
            lines.append(f"### Strategy {sid} — sweep over `{param}`\n")
            lines.append(f"| {param} | total_pnl | sharpe | trades | max_dd |")
            lines.append("|---:|---:|---:|---:|---:|")
            for row in entry.get("table", []):
                lines.append(
                    f"| {row['value']:.3f} | {row['total_pnl']:.4f} | {row['sharpe']:.3f} | "
                    f"{row['n_trades']} | {row['max_dd']:.4f} |"
                )
            lines.append(
                f"\n*Interpretation:* {interp.get('interpretation', 'n/a')}\n"
                f"*Stability:* {interp.get('stability_summary', 'n/a')}\n"
            )

    if adversarial:
        lines.append("\n## Adversarial scenarios\n")
        for entry in adversarial:
            sid = entry["strategy_id"]
            lines.append(f"### Strategy {sid}\n")
            for s in entry.get("scenarios", []):
                bm = s.get("backtest_result", {})
                lines.append(f"- **{s['name']}** — {s.get('description', '')}")
                lines.append(
                    f"  - n_trades={bm.get('n_trades', 0)}, total_pnl={bm.get('total_pnl', 0):.4f}, "
                    f"max_dd={bm.get('max_drawdown_pnl_units', 0):.4f}, "
                    f"max_loss_streak={bm.get('largest_losing_streak', 0)}"
                )
                lines.append(f"  - {s.get('failure_mode_observed', '')}")
            lines.append("")

    # Required: explicit high-risk note for martingale strategies in the report.
    high_risk = [c for c in critiques if c.get("is_high_risk") or c.get("risk_level") == "high"]
    if high_risk:
        lines.append("\n## High-risk strategies (pipeline-flagged)\n")
        for c in high_risk:
            lines.append(f"- **{c['strategy_id']}** — flagged HIGH RISK. {c.get('warnings', [''])[0] if c.get('warnings') else ''}")
        lines.append("")

    lines.append("\n## Retail-reader warning\n")
    lines.append(
        "If you are not a quantitative analyst with risk-management infrastructure: do not "
        "trade strategies based on output like this. In-sample metrics over a few months "
        "of data, computed against historical or simulated paths with no slippage, no "
        "broker stake limits, and no spread, are NOT representative of live trading. "
        "Martingale-style strategies in particular are mathematically prone to ruin "
        "regardless of any in-sample win rate."
    )
    lines.append("")

    body = "\n".join(lines)
    Path(output_path).write_text(body, encoding="utf-8")
    return body
