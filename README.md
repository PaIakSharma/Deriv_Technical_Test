# Deriv AI Support Triage Pipeline

Multi-stage AI pipeline that classifies support messages, scores risk, drafts responses, checks compliance, collects operator corrections, and reclassifies using few-shot examples.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY
```

## Run

```bash
# Interactive (operator corrects classifications at Stage 5)
python run.py

# Non-interactive / CI / replay
python run.py --no-interactive
```

## Validate

```bash
python validate.py
# or
make validate
```

## Pipeline stages

| Stage | Description |
|---|---|
| INPUTS_LOADED | Read support_messages.json |
| PRIOR_CORRECTIONS_LOADED | Load corrections.jsonl if it exists (longitudinal store) |
| INITIAL_CLASSIFICATION_COMPLETE | LLM classifies all messages into 6 categories |
| RISK_SCORING_COMPLETE | LLM scores escalation/flagged messages; deterministic fallback for the rest |
| RESPONSES_DRAFTED | Batched call for low/medium risk; individual calls for high/critical |
| COMPLIANCE_CHECK_COMPLETE | LLM + keyword checks for promises, timelines, liability |
| OPERATOR_CORRECTIONS_COLLECTED | Interactive CLI — accept / correct / skip each classification |
| FEW_SHOT_BLOCK_BUILT | Build examples from operator-validated corrections |
| RECLASSIFICATION_COMPLETE | Re-run Stage 1 with few-shot examples injected |
| BEFORE_AFTER_COMPARISON_COMPLETE | Compute accuracy delta over operator-labeled set |
| ANALYTICS_GENERATED | Summary stats, category distribution, risk breakdown |
| RESULTS_FINALISED | All artifacts written |

## Output artifacts

| File | Description |
|---|---|
| `initial_classifications.json` | Stage 1 output |
| `risk_scores.json` | Stage 2 output |
| `draft_responses.json` | Stage 3 output |
| `response_compliance.json` | Stage 4 output |
| `corrections.jsonl` | Append-only operator correction log |
| `reclassified_outputs.json` | Stage 6 output |
| `triage_output.json` | Full before/after comparison + accuracy delta |
| `analytics_summary.json` | Summary statistics |
| `routing_decisions.json` | Team routing + SLA per message |
| `correction_store_status.json` | Whether prior corrections were loaded |
| `llm_calls.jsonl` | One log record per LLM call |

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required. Your OpenRouter API key |
| `MODEL` | `google/gemini-2-flash-preview` | Model to use via OpenRouter |
