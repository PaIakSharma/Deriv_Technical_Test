PYTHON ?= python

.PHONY: all install run validate clean

all: run validate

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) pipeline.py

validate:
	$(PYTHON) validate.py

clean:
	rm -rf specs ledgers data
	rm -f data_manifest.json metrics.json critiques.json walk_forward.json \
	      parameter_sensitivity.json adversarial_scenarios.json \
	      comparative_brief.md report.md llm_calls.jsonl human_reviews.jsonl
