run:
	python run.py

run-auto:
	python run.py --no-interactive

validate:
	python validate.py

clean:
	del /f /q initial_classifications.json risk_scores.json draft_responses.json \
		response_compliance.json reclassified_outputs.json triage_output.json \
		analytics_summary.json routing_decisions.json correction_store_status.json \
		llm_calls.jsonl corrections.jsonl 2>nul || true

.PHONY: run run-auto validate clean
