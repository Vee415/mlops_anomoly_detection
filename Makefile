.PHONY: setup train test lint serve clean pull push pipeline

PYTHON ?= python

setup:
	conda env create -f environment.yml
	dvc pull

train:
	$(PYTHON) src/train.py

pipeline:
	dvc repro

test:
	$(PYTHON) -m pytest tests/unit/ -v --tb=short

test-integration:
	$(PYTHON) -m pytest tests/integration/ -v --tb=short

test-all:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	ruff check src/

typecheck:
	mypy src/

serve:
	uvicorn src.serve:app --host 0.0.0.0 --port 8000

pull:
	dvc pull

push:
	dvc push

clean:
	dvc gc -f
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

tune:
	$(PYTHON) src/tune.py --n-trials 50

monitor:
	$(PYTHON) src/monitor.py

evaluate:
	$(PYTHON) src/evaluate.py

metrics:
	@cat reports/metrics.json

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-build:
	docker compose build

docker-serve:
	docker compose up serve