.PHONY: train train-lstm sweep evaluate test lint format export docker-build docker-push deploy clean

# ── Training ────────────────────────────────────────────────────────────────
train:
	python -m src.train --config configs/experiment_01.yaml

train-lstm:
	python -m src.train --config configs/experiment_01.yaml --model basic_lstm

sweep:
	wandb sweep configs/sweep.yaml

# ── Evaluation ──────────────────────────────────────────────────────────────
evaluate:
	python -m src.evaluate --config configs/experiment_01.yaml --checkpoint output_models/best_actor.pth

# ── Tests ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Lint / format ───────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/

# ── Deploy (scaffold — not wired up yet) ────────────────────────────────────
export:
	python deploy/export.py

docker-build:
	docker build -t dynamic_treatment:latest -f deploy/Dockerfile .

docker-push:
	# TODO: set ECR_REPO in .env
	docker tag dynamic_treatment:latest $(ECR_REPO):latest
	docker push $(ECR_REPO):latest

deploy: docker-build docker-push

# ── Misc ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
