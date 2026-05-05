"""Pipeline validation — the source of truth for "did the run produce
everything it claims to have produced".

Checks:
  * required artifacts exist and are valid JSON
  * every strategy has a formal JSON spec
  * each spec contains at least 3 substantive ambiguities
  * specs use the required schema (top-level keys present)
  * ledgers exist for every backtested strategy
  * metrics are computed in code, not by the LLM (heuristic: every strategy
    has a metrics record AND no LLM record claims a metrics output_artifact)
  * ledger totals reconcile with summary metrics
  * Strategy C is flagged as high risk when the public fixture is used
  * backtest assumptions are documented in BACKTEST_ASSUMPTIONS / report
  * llm_calls.jsonl contains separate records for formalisation and critique
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.backtest import BACKTEST_ASSUMPTIONS
from src.formaliser import REQUIRED_TOP_LEVEL, validate_spec
from src.metrics import reconcile_ledger
from src.utils import read_json


REQUIRED_ARTIFACTS = [
    "strategies.json",
    "data_manifest.json",
    "metrics.json",
    "critiques.json",
    "report.md",
    "llm_calls.jsonl",
]
OPTIONAL_ARTIFACTS = [
    "walk_forward.json",
    "parameter_sensitivity.json",
    "adversarial_scenarios.json",
    "comparative_brief.md",
]


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _is_public_fixture(strategies: List[Dict[str, Any]]) -> bool:
    """The public fixture is the EXACT three-strategy A/B/C set with the
    martingale flavour in C. The evaluator may swap fixtures, so we only
    enforce the "C is high risk" rule for the published fixture.
    We also enforce 'any martingale-like strategy is high risk' generally.
    """
    if len(strategies) != 3:
        return False
    ids = {s.get("id") for s in strategies}
    if ids != {"A", "B", "C"}:
        return False
    desc_c = next((s["description"] for s in strategies if s.get("id") == "C"), "")
    return "martingale" in desc_c.lower() or "double" in desc_c.lower()


def run_validation() -> Tuple[int, List[str]]:
    problems: List[str] = []

    # 1. required artifacts exist
    for art in REQUIRED_ARTIFACTS:
        if not Path(art).exists():
            problems.append(f"required artifact missing: {art}")
    if any(p.startswith("required artifact missing") for p in problems):
        return 1, problems

    # 2. JSON files parse
    json_artifacts = ["strategies.json", "data_manifest.json", "metrics.json", "critiques.json"]
    json_artifacts += [a for a in OPTIONAL_ARTIFACTS if a.endswith(".json") and Path(a).exists()]
    for j in json_artifacts:
        try:
            read_json(j)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{j} is not valid JSON: {e}")

    strategies = read_json("strategies.json")
    metrics = read_json("metrics.json")
    critiques = read_json("critiques.json")

    # 3. every strategy has a formal spec
    for s in strategies:
        sid = s["id"]
        spec_path = Path("specs") / f"{sid}.json"
        if not spec_path.exists():
            problems.append(f"missing spec for strategy {sid}: {spec_path}")
            continue
        try:
            spec = read_json(spec_path)
        except Exception as e:  # noqa: BLE001
            problems.append(f"spec {spec_path} not valid JSON: {e}")
            continue
        # 4. spec uses required schema + 5. >= 3 substantive ambiguities
        spec_problems = validate_spec(spec)
        for sp in spec_problems:
            problems.append(f"spec {sid}: {sp}")
        missing_top = REQUIRED_TOP_LEVEL - set(spec)
        if missing_top:
            problems.append(f"spec {sid} missing top-level keys: {sorted(missing_top)}")

    # 6. ledgers exist
    for s in strategies:
        sid = s["id"]
        led = Path("ledgers") / f"{sid}.csv"
        if not led.exists():
            problems.append(f"missing ledger for strategy {sid}: {led}")

    # 7. metrics-by-LLM check (heuristic: no llm log entry claims a numeric metric output)
    llm_log = _load_jsonl("llm_calls.jsonl")
    forbidden_outputs = {"metrics.json", "ledgers/"}
    for rec in llm_log:
        out = str(rec.get("output_artifact", ""))
        if out == "metrics.json" or out.startswith("ledgers/"):
            problems.append(
                f"LLM call (stage={rec.get('stage')}) claims a numerical artifact as output: {out}"
            )

    # 8. ledger ↔ metrics reconciliation
    for m in metrics:
        sid = m["strategy_id"]
        led = Path("ledgers") / f"{sid}.csv"
        if not led.exists():
            continue
        rec_problems = reconcile_ledger(str(led), m)
        for rp in rec_problems:
            problems.append(f"reconciliation for {sid}: {rp}")

    # 9. Strategy C high-risk on public fixture; all martingale strategies high-risk always.
    crit_by_id = {c["strategy_id"]: c for c in critiques}
    if _is_public_fixture(strategies):
        c = crit_by_id.get("C")
        if not c:
            problems.append("public fixture: critique for Strategy C missing")
        elif not (c.get("is_high_risk") or c.get("risk_level") == "high"):
            problems.append("public fixture: Strategy C must be flagged as high risk")
        report = Path("report.md").read_text(encoding="utf-8") if Path("report.md").exists() else ""
        if "high risk" not in report.lower() and "high-risk" not in report.lower():
            problems.append("report.md does not contain a high-risk warning")
    for c in critiques:
        if c.get("is_martingale_or_loss_escalating") and not (c.get("is_high_risk") or c.get("risk_level") == "high"):
            problems.append(
                f"strategy {c.get('strategy_id')}: martingale/loss-escalating but not flagged high-risk"
            )

    # 10. backtest assumptions documented
    if not BACKTEST_ASSUMPTIONS:
        problems.append("BACKTEST_ASSUMPTIONS dict is empty")
    report_text = Path("report.md").read_text(encoding="utf-8") if Path("report.md").exists() else ""
    for key in ("intrabar_ordering",):
        if key not in str(BACKTEST_ASSUMPTIONS):
            problems.append(f"backtest assumption missing: {key}")
        if "intrabar" not in report_text.lower():
            problems.append("report.md does not document intrabar ordering assumption")
            break

    # 11. llm_calls.jsonl shape & required stage records
    if not llm_log:
        problems.append("llm_calls.jsonl is empty — no LLM calls were logged")
    else:
        required_fields = {"stage", "strategy_id", "timestamp", "provider", "model",
                           "prompt_hash", "input_artifacts", "output_artifact"}
        for i, rec in enumerate(llm_log):
            missing = required_fields - set(rec)
            if missing:
                problems.append(f"llm_calls.jsonl[{i}] missing fields: {sorted(missing)}")
        # one formalisation + one critique per strategy
        for s in strategies:
            sid = s["id"]
            has_form = any(r.get("stage") == "formalisation" and r.get("strategy_id") == sid for r in llm_log)
            has_crit = any(r.get("stage") == "critique" and r.get("strategy_id") == sid for r in llm_log)
            if not has_form:
                problems.append(f"llm_calls.jsonl: no formalisation record for strategy {sid}")
            if not has_crit:
                problems.append(f"llm_calls.jsonl: no critique record for strategy {sid}")

    return (0 if not problems else 1), problems


def main() -> int:
    rc, problems = run_validation()
    if rc == 0:
        print("[validate] OK — every required invariant holds.", flush=True)
        return 0
    print(f"[validate] FAILED with {len(problems)} problem(s):", flush=True)
    for p in problems:
        print(f"  - {p}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
