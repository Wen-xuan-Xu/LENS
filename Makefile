# LENS — developer entry points. Everything here runs against the synthetic
# fake data under data/fake/ (real study data is not distributable; see README).

PYTHON ?= python
DATA_ROOT ?= data/fake

.PHONY: help
help:
	@echo "LENS make targets:"
	@echo "  make fake-data            regenerate synthetic raw sensors + feature rows + datasets"
	@echo "  make smoke-test           run the full pipeline end-to-end on fake data"
	@echo "  make smoke-test-feature   raw sensors -> feature_rows.csv"
	@echo "  make smoke-test-pipeline  EMA -> templates -> (mock) enrichment -> training jsonl"
	@echo "  make smoke-test-eval      (mock) inference -> NLP metrics"
	@echo "  make test                 pytest tests/"
	@echo "  make lint                 ruff check"
	@echo "  make train-stage1 / train-stage2   delegates to third_party/ChatTS-Training (needs GPUs)"

# --- fake data ---------------------------------------------------------------
.PHONY: fake-data
fake-data:
	$(PYTHON) $(DATA_ROOT)/generate.py --out $(DATA_ROOT) --all

# --- smoke tests -------------------------------------------------------------
.PHONY: smoke-test-feature
smoke-test-feature: fake-data
	$(PYTHON) -m lens.feature_engineering.build_feature_rows \
		--data-root $(DATA_ROOT) --output $(DATA_ROOT)/feature_rows.csv
	$(PYTHON) -m lens.feature_engineering.add_unlock_and_narratives \
		--data-root $(DATA_ROOT) \
		--in $(DATA_ROOT)/feature_rows.csv --out $(DATA_ROOT)/filtered_feature_rows.csv

.PHONY: smoke-test-pipeline
smoke-test-pipeline: smoke-test-feature
	$(PYTHON) -m lens.data_pipeline.run_pipeline --config configs/pipeline/smoke.yaml --mock-llm
	$(PYTHON) -m lens.data_pipeline.dataset_build.build_dataset --config configs/pipeline/dataset_build_smoke.yaml
	$(PYTHON) -m lens.data_pipeline.dataset_build.convert_hf_to_jsonl --root $(DATA_ROOT)/arrow --out $(DATA_ROOT)
	$(PYTHON) -m lens.data_pipeline.fix_ts_tokens $(DATA_ROOT)/narrative_dataset $(DATA_ROOT)/qa_dataset

.PHONY: smoke-test-eval
smoke-test-eval:
	$(PYTHON) -m lens.eval.metrics.compute_nlp_metrics --pred examples/one_sample_each.jsonl --ref examples/one_sample_each.jsonl --self-test

.PHONY: smoke-test
smoke-test: smoke-test-feature smoke-test-pipeline smoke-test-eval
	@echo "SMOKE TEST OK"

# --- quality -----------------------------------------------------------------
.PHONY: test
test:
	pytest -q tests/

.PHONY: lint
lint:
	ruff check lens tests data/fake/generate.py

# --- training (requires GPUs + the submodule) --------------------------------
.PHONY: train-stage1
train-stage1:
	bash lens/training/scripts/train_stage1.sh

.PHONY: train-stage2
train-stage2:
	bash lens/training/scripts/train_stage2.sh
