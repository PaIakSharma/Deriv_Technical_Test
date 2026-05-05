"""
Validation script — checks all required artifacts are present and correct.
Usage: python validate.py
Exit 0 = all checks passed. Exit 1 = failures found.
"""
import json
import os
import re
import sys

from pipeline.vocab import ALLOWED_CATEGORIES, ALLOWED_RISKS, ALLOWED_VIOLATIONS, ALLOWED_ACTIONS, COMPLIANCE_RISKY_TERMS

REQUIRED_FILES = [
    "support_messages.json",
    "initial_classifications.json",
    "risk_scores.json",
    "draft_responses.json",
    "corrections.jsonl",
    "reclassified_outputs.json",
    "triage_output.json",
    "response_compliance.json",
    "analytics_summary.json",
    "routing_decisions.json",
    "correction_store_status.json",
    "llm_calls.jsonl",
]

failures = []
warnings = []


def fail(msg):
    failures.append(msg)
    print(f"  FAIL: {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"  WARN: {msg}")


def ok(msg):
    print(f"  OK:   {msg}")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                fail(f"{path} line {i}: invalid JSON — {e}")
    return records


def check_files_exist():
    print("\n── Artifact existence ──")
    for path in REQUIRED_FILES:
        if os.path.exists(path):
            ok(path)
        else:
            fail(f"Missing: {path}")


def check_messages():
    print("\n── Input messages ──")
    messages = load_json("support_messages.json")
    if not isinstance(messages, list) or not messages:
        fail("support_messages.json must be a non-empty list")
        return set()
    ok(f"support_messages.json: {len(messages)} messages")
    return {m["id"] for m in messages}


def check_classifications(msg_ids):
    print("\n── Initial classifications ──")
    cls = load_json("initial_classifications.json")
    cls_ids = {c["id"] for c in cls}

    for mid in msg_ids:
        if mid not in cls_ids:
            fail(f"Message {mid} missing from initial_classifications.json")

    for c in cls:
        cat = c.get("category", "")
        if cat not in ALLOWED_CATEGORIES:
            fail(f"[id={c['id']}] Invalid category: {cat!r}")
        conf = c.get("confidence", -1)
        if not (0.0 <= conf <= 1.0):
            fail(f"[id={c['id']}] Confidence out of range: {conf}")
        if conf < 0.7 and not c.get("needs_human_review"):
            fail(f"[id={c['id']}] confidence={conf:.2f} < 0.7 but needs_human_review is not true")

    ok(f"All {len(cls)} classifications valid")
    return cls


def check_risk_scores(msg_ids, classifications):
    print("\n── Risk scores ──")
    risk = load_json("risk_scores.json")
    risk_ids = {r["id"] for r in risk}

    llm_eligible = {
        c["id"] for c in classifications
        if c["category"] == "escalation" or c.get("needs_human_review")
    }
    llm_scored = {r["id"] for r in risk if r.get("scored_by") == "llm"}

    for mid in llm_eligible:
        if mid not in llm_scored:
            warn(f"Message {mid} eligible for LLM risk scoring but not marked scored_by=llm")

    for r in risk:
        if r.get("risk_level") not in ALLOWED_RISKS:
            fail(f"[id={r['id']}] Invalid risk_level: {r.get('risk_level')!r}")

    ok(f"All {len(risk)} risk scores valid")
    return risk


def check_drafts(msg_ids, risk_scores):
    print("\n── Draft responses ──")
    drafts = load_json("draft_responses.json")
    risk_by_id = {r["id"]: r["risk_level"] for r in risk_scores}

    for d in drafts:
        mid = d["id"]
        risk = risk_by_id.get(mid, "low")
        mode = d.get("drafting_mode", "")

        if risk in ("high", "critical") and mode != "individual":
            fail(f"[id={mid}] risk={risk} but drafting_mode={mode!r} (expected 'individual')")
        if risk in ("low", "medium") and mode != "batched":
            fail(f"[id={mid}] risk={risk} but drafting_mode={mode!r} (expected 'batched')")

    ok(f"All {len(drafts)} drafts valid")


def check_corrections():
    print("\n── Corrections ──")
    if not os.path.exists("corrections.jsonl"):
        fail("corrections.jsonl missing")
        return []
    records = load_jsonl("corrections.jsonl")
    for i, r in enumerate(records):
        if r.get("action") not in ALLOWED_ACTIONS:
            fail(f"corrections.jsonl record {i}: invalid action {r.get('action')!r}")
    ok(f"{len(records)} correction records valid")
    return records


def check_reclassified(msg_ids):
    print("\n── Reclassified outputs ──")
    reclass = load_json("reclassified_outputs.json")
    reclass_ids = {c["id"] for c in reclass}
    for mid in msg_ids:
        if mid not in reclass_ids:
            fail(f"Message {mid} missing from reclassified_outputs.json")
    for c in reclass:
        if c.get("category") not in ALLOWED_CATEGORIES:
            fail(f"[id={c['id']}] Invalid reclassified category: {c.get('category')!r}")
    ok(f"All {len(reclass)} reclassified records valid")


def check_triage():
    print("\n── Triage output ──")
    triage = load_json("triage_output.json")
    for field in ("accuracy_before", "accuracy_after", "delta"):
        if field not in triage:
            fail(f"triage_output.json missing field: {field}")
        else:
            ok(f"triage_output.json has {field} = {triage[field]}")


def check_compliance():
    print("\n── Compliance ──")
    comp = load_json("response_compliance.json")
    for c in comp:
        for v in c.get("violations", []):
            if v not in ALLOWED_VIOLATIONS:
                fail(f"[id={c['id']}] Invalid violation: {v!r}")
    # Sanity: keyword scan assertion
    drafts = load_json("draft_responses.json")
    comp_by_id = {c["id"]: c for c in comp}
    for d in drafts:
        text = d.get("draft_response", "")
        for pattern in COMPLIANCE_RISKY_TERMS:
            if re.search(pattern, text, re.IGNORECASE):
                record = comp_by_id.get(d["id"], {})
                if record.get("passed"):
                    warn(f"[id={d['id']}] Draft contains risky term '{pattern}' but compliance passed=True")
    ok(f"All {len(comp)} compliance records valid")


def check_llm_calls():
    print("\n── LLM call log ──")
    records = load_jsonl("llm_calls.jsonl")
    stages_found = {r.get("stage") for r in records}

    always_required = {
        "stage1_classify",
        "stage2_risk",
        "stage3_draft_batched",
        "stage4_compliance",
        "stage6_reclassify",
    }
    for s in always_required:
        if s not in stages_found:
            fail(f"llm_calls.jsonl: missing stage record for '{s}'")
        else:
            ok(f"Found LLM call: {s}")

    # Only require individual drafts if high/critical messages exist
    if os.path.exists("risk_scores.json"):
        risk = load_json("risk_scores.json")
        high_crit = [r for r in risk if r.get("risk_level") in ("high", "critical")]
        if high_crit and "stage3_draft_individual" not in stages_found:
            fail("llm_calls.jsonl: high/critical messages exist but no stage3_draft_individual record")
        elif high_crit:
            ok(f"Found LLM call: stage3_draft_individual")

    # Reclassification must have few_shot_examples_included=True
    reclass_calls = [r for r in records if r.get("stage") == "stage6_reclassify"]
    if not reclass_calls:
        fail("llm_calls.jsonl: no stage6_reclassify record found")
    elif not reclass_calls[-1].get("few_shot_examples_included"):
        fail("llm_calls.jsonl: stage6_reclassify call does not have few_shot_examples_included=true")
    else:
        ok("stage6_reclassify has few_shot_examples_included=true")


def main():
    print("=" * 60)
    print("Artifact Validation")
    print("=" * 60)

    check_files_exist()

    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        print(f"\nCannot continue — {len(missing)} required file(s) missing.")
        sys.exit(1)

    msg_ids = check_messages()
    classifications = check_classifications(msg_ids)
    risk_scores = check_risk_scores(msg_ids, classifications)
    check_drafts(msg_ids, risk_scores)
    check_corrections()
    check_reclassified(msg_ids)
    check_triage()
    check_compliance()
    check_llm_calls()

    print(f"\n{'='*60}")
    if failures:
        print(f"RESULT: {len(failures)} failure(s), {len(warnings)} warning(s)")
        for f in failures:
            print(f"  ✗ {f}")
        sys.exit(1)
    else:
        print(f"RESULT: All checks passed ({len(warnings)} warning(s))")
        if warnings:
            for w in warnings:
                print(f"  ⚠ {w}")
        sys.exit(0)


if __name__ == "__main__":
    main()
