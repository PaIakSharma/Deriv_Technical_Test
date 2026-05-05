"""
Deriv AI Support Triage Pipeline
Usage:
  python run.py                  # interactive operator corrections
  python run.py --no-interactive # auto-accept (CI / replay mode)
"""
import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from pipeline.state import PipelineState
from pipeline import (
    stage1_classify,
    stage2_risk,
    stage3_draft,
    stage4_compliance,
    operator,
    fewshot,
    reclassify as reclassify_module,
    compare,
    analytics,
    routing,
)


def _load_jsonl(path):
    records = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Deriv AI Support Triage Pipeline")
    parser.add_argument("--no-interactive", action="store_true", help="Auto-accept all corrections")
    args = parser.parse_args()

    state = PipelineState()
    print(f"\n{'='*60}")
    print("Deriv AI Support Triage Pipeline")
    print(f"{'='*60}\n")

    # ── STAGE 0: Load inputs ────────────────────────────────────────
    print("[INIT] Loading support messages...")
    with open("support_messages.json", encoding="utf-8") as f:
        messages = json.load(f)
    state.advance("INPUTS_LOADED")
    print(f"  Loaded {len(messages)} messages\n")

    # ── Load prior corrections (longitudinal store) ─────────────────
    prior_corrections = _load_jsonl("corrections.jsonl")
    correction_store_status = {
        "prior_corrections_loaded": bool(prior_corrections),
        "prior_correction_count": len(prior_corrections),
        "source": "corrections.jsonl" if prior_corrections else None,
    }
    _write_json("correction_store_status.json", correction_store_status)

    if prior_corrections:
        state.advance("PRIOR_CORRECTIONS_LOADED")
        print(f"  Loaded {len(prior_corrections)} prior corrections from corrections.jsonl\n")

    # ── STAGE 1: Classification ─────────────────────────────────────
    print("[STAGE 1] Classifying all messages...")
    prior_few_shot = fewshot.build_block(prior_corrections)
    classifications = stage1_classify.classify(messages, few_shot_block=prior_few_shot)
    _write_json("initial_classifications.json", classifications)
    state.advance("INITIAL_CLASSIFICATION_COMPLETE")
    print(f"  Classified {len(classifications)} messages\n")

    # ── STAGE 2: Risk scoring ───────────────────────────────────────
    print("[STAGE 2] Scoring risk...")
    risk_scores = stage2_risk.score(messages, classifications)
    _write_json("risk_scores.json", risk_scores)
    state.advance("RISK_SCORING_COMPLETE")

    risk_counts = {}
    for r in risk_scores:
        risk_counts[r["risk_level"]] = risk_counts.get(r["risk_level"], 0) + 1
    print(f"  Risk breakdown: {risk_counts}\n")

    # ── STAGE 3: Drafting ───────────────────────────────────────────
    print("[STAGE 3] Drafting responses...")
    drafts = stage3_draft.draft(messages, risk_scores)
    _write_json("draft_responses.json", drafts)
    state.advance("RESPONSES_DRAFTED")
    batched = sum(1 for d in drafts if d["drafting_mode"] == "batched")
    individual = sum(1 for d in drafts if d["drafting_mode"] == "individual")
    print(f"  Batched: {batched}, Individual: {individual}\n")

    # ── STAGE 4: Compliance check ───────────────────────────────────
    print("[STAGE 4] Checking response compliance...")
    compliance_results = stage4_compliance.check(drafts)
    _write_json("response_compliance.json", compliance_results)
    state.advance("COMPLIANCE_CHECK_COMPLETE")
    failures = sum(1 for c in compliance_results if not c["passed"])
    print(f"  Compliance: {failures} failures out of {len(compliance_results)}\n")

    # ── STAGE 5: Operator corrections ──────────────────────────────
    print("[STAGE 5] Operator correction interface...")
    new_corrections = operator.collect(
        messages, classifications, risk_scores,
        no_interactive=args.no_interactive,
    )
    # Append new corrections to persistent store
    with open("corrections.jsonl", "a", encoding="utf-8") as f:
        for c in new_corrections:
            f.write(json.dumps(c) + "\n")
    state.advance("OPERATOR_CORRECTIONS_COLLECTED")
    accepted = sum(1 for c in new_corrections if c["action"] == "accepted")
    corrected = sum(1 for c in new_corrections if c["action"] == "corrected")
    skipped = sum(1 for c in new_corrections if c["action"] == "skipped")
    print(f"\n  Corrections: {accepted} accepted, {corrected} corrected, {skipped} skipped\n")

    # Load all corrections for few-shot
    all_corrections = _load_jsonl("corrections.jsonl")

    # ── STAGE 6: Few-shot reclassification ──────────────────────────
    print("[STAGE 6] Building few-shot block and reclassifying...")
    few_shot_block = fewshot.build_block(all_corrections)
    state.advance("FEW_SHOT_BLOCK_BUILT")

    reclassified = reclassify_module.reclassify(messages, few_shot_block)
    _write_json("reclassified_outputs.json", reclassified)
    state.advance("RECLASSIFICATION_COMPLETE")
    print(f"  Reclassified {len(reclassified)} messages\n")

    # ── STAGE 7: Before/after comparison ───────────────────────────
    print("[STAGE 7] Computing before/after comparison...")
    triage = compare.build_comparison(
        messages, classifications, reclassified,
        all_corrections, risk_scores, drafts, compliance_results,
    )
    _write_json("triage_output.json", triage)
    state.advance("BEFORE_AFTER_COMPARISON_COMPLETE")
    if triage["delta"] is not None:
        print(f"  Accuracy delta: {triage['delta']:+.2%}\n")
    else:
        print("  No labeled messages — accuracy delta not computed\n")

    # ── STAGE 8: Analytics ──────────────────────────────────────────
    print("[STAGE 8] Generating analytics...")
    summary = analytics.generate(classifications, risk_scores, all_corrections, triage)
    _write_json("analytics_summary.json", summary)
    state.advance("ANALYTICS_GENERATED")
    analytics.print_summary(summary)

    # ── STRETCH: Routing decisions ──────────────────────────────────
    print("\n[ROUTING] Generating routing decisions...")
    routing_decisions = routing.route_with_categories(messages, risk_scores, classifications)
    _write_json("routing_decisions.json", routing_decisions)

    # ── Final state ─────────────────────────────────────────────────
    state.advance("VALIDATION_COMPLETE")
    state.advance("RESULTS_FINALISED")
    print(f"\n{'='*60}")
    print("Pipeline complete. Run 'python validate.py' to verify all artifacts.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
