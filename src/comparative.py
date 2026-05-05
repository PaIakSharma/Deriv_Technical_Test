"""One-page comparative brief.

Ranks strategies on a risk-adjusted basis, names strongest and weakest,
flags martingale-style strategies as high-risk, and includes the mandatory
retail-reader and robustness warnings.

Computed deterministically — no LLM call here. The LLM critiques have already
been written into ``critiques.json`` upstream and are surfaced verbatim.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .utils import read_json


def _risk_adjusted_score(metric: Dict[str, Any], critique: Dict[str, Any]) -> float:
    """A simple deterministic scoring function for ranking purposes only.

    Uses Sharpe as the primary axis, profit factor as a tiebreaker, with a
    hard penalty for high-risk-flagged strategies.
    """
    sharpe = float(metric.get("sharpe_annualised", 0.0) or 0.0)
    pf = metric.get("profit_factor", 0.0)
    if pf == "inf":
        pf_val = 5.0  # cap
    else:
        try:
            pf_val = float(pf)
        except Exception:  # noqa: BLE001
            pf_val = 0.0
    pf_val = max(0.0, min(pf_val, 5.0))

    score = sharpe + 0.25 * pf_val

    if critique.get("is_high_risk") or critique.get("risk_level") == "high":
        score -= 5.0  # flagged-high-risk strategies cannot rank above safe ones
    return score


def build_comparative_brief(
    *,
    metrics_path: str | Path = "metrics.json",
    critiques_path: str | Path = "critiques.json",
    output_path: str | Path = "comparative_brief.md",
) -> str:
    metrics = read_json(metrics_path)
    critiques = read_json(critiques_path)
    crit_by_id = {c["strategy_id"]: c for c in critiques}

    rows: List[Dict[str, Any]] = []
    for m in metrics:
        sid = m["strategy_id"]
        c = crit_by_id.get(sid, {})
        rows.append({"sid": sid, "metrics": m, "critique": c, "score": _risk_adjusted_score(m, c)})

    rows.sort(key=lambda r: r["score"], reverse=True)
    if not rows:
        body = "No strategies to compare.\n"
    else:
        strongest = rows[0]
        weakest = rows[-1]
        any_martingale = any(
            r["critique"].get("is_martingale_or_loss_escalating") for r in rows
        )

        lines: List[str] = []
        lines.append("# Comparative Brief — Risk-Adjusted Ranking\n")
        lines.append("> **Analysis tooling, not financial advice.** All numbers describe historical or simulated behaviour over a finite window. Past performance does not generalise.\n")
        lines.append("\n## Ranking\n")
        lines.append("| # | Strategy | Sharpe | Profit Factor | Max DD | Trades | Risk Flag | Score |")
        lines.append("|---|----------|-------:|--------------:|-------:|-------:|-----------|------:|")
        for i, r in enumerate(rows, start=1):
            m = r["metrics"]
            c = r["critique"]
            risk = c.get("risk_level", "unknown")
            if c.get("is_high_risk"):
                risk = f"**{risk.upper()}** ⚠"
            lines.append(
                f"| {i} | {r['sid']} | {m.get('sharpe_annualised', 0):.3f} | "
                f"{m.get('profit_factor', 0)} | {m.get('max_drawdown_pnl_units', 0):.4f} | "
                f"{m.get('n_trades', 0)} | {risk} | {r['score']:.3f} |"
            )

        lines.append("\n## Strongest and Weakest\n")
        lines.append(
            f"- **Strongest (this sample):** {strongest['sid']} — "
            f"Sharpe {strongest['metrics'].get('sharpe_annualised', 0):.3f}, "
            f"PF {strongest['metrics'].get('profit_factor', 0)}, "
            f"verdict `{strongest['critique'].get('robustness_verdict', 'n/a')}`."
        )
        lines.append(
            f"- **Weakest (this sample):** {weakest['sid']} — "
            f"Sharpe {weakest['metrics'].get('sharpe_annualised', 0):.3f}, "
            f"risk_level `{weakest['critique'].get('risk_level', 'n/a')}`."
        )

        lines.append("\n## Risk-adjusted reasoning\n")
        for r in rows:
            m, c = r["metrics"], r["critique"]
            verdict = c.get("robustness_verdict", "n/a")
            lines.append(
                f"- **{r['sid']}** — Sharpe {m.get('sharpe_annualised', 0):.3f}, "
                f"max DD {m.get('max_drawdown_pnl_units', 0):.4f} on "
                f"{m.get('n_trades', 0)} trades. Verdict: *{verdict}*. "
                f"Regime exposure: {c.get('regime_dependence', 'n/a')[:200]}"
            )

        lines.append("\n## Robustness warning\n")
        lines.append(
            "Sharpe and profit factor are extremely sensitive to the small set of largest "
            "moves in any sample. Walk-forward (`walk_forward.json`) and parameter "
            "sensitivity (`parameter_sensitivity.json`) are the more honest read of "
            "robustness — consult them before treating the ranking above as anything more "
            "than a snapshot of one window of one data series."
        )

        if any_martingale:
            lines.append("\n## Martingale / loss-escalation warning\n")
            for r in rows:
                if r["critique"].get("is_martingale_or_loss_escalating"):
                    lines.append(
                        f"- **{r['sid']}** uses loss-doubling sizing. This is a high-risk "
                        f"structure: cumulative risk grows as 2^k − 1 after k consecutive "
                        f"losses, so a single tail event wipes out many small winners. The "
                        f"in-sample win rate is **not informative** about ruin probability. "
                        f"Treat in-sample profitability as luck of the seed, not as edge."
                    )

        lines.append("\n## Retail-reader warning\n")
        lines.append(
            "If you are reading this as a retail trader: **this document is not a "
            "recommendation.** Strategies that look profitable on a few months of "
            "historical or simulated data routinely lose money in live conditions because "
            "of slippage, spreads, news gaps, broker max-stake limits, and the simple fact "
            "that you cannot replay history. None of the math in this report makes those "
            "real-world frictions go away."
        )
        body = "\n".join(lines) + "\n"

    Path(output_path).write_text(body, encoding="utf-8")
    return body
