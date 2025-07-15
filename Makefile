.PHONY: help install install-dev test lint format type-check security clean build docs docker

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install the package with development dependencies
	pip install -e ".[dev,docs]"

test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast: ## Run tests without coverage
	pytest tests/ -v -x

lint: ## Run linting
	flake8 src tests
	black --check src tests
	isort --check-only src tests

format: ## Format code
	black src tests
	isort src tests

type-check: ## Run type checking
	mypy src

security: ## Run security checks
	bandit -r src
	safety check

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: ## Build the package
	python -m build

docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

docker-build: ## Build Docker image
	docker build -t polymer-prediction .

docker-run: ## Run Docker container
	docker run -it --rm -v $(PWD):/home/app polymer-prediction

docker-jupyter: ## Run Jupyter in Docker
	docker-compose up jupyter

docker-tensorboard: ## Run TensorBoard in Docker
	docker-compose up tensorboard

setup-env: ## Set up development environment
	python -m venv venv
	./venv/Scripts/activate && pip install -e ".[dev,docs]"
	./venv/Scripts/activate && pre-commit install

train: ## Train the model with default configuration
	python -m polymer_prediction.main

train-config: ## Train with specific config (usage: make train-config CONFIG=experiment/custom)
	python -m polymer_prediction.main --config-name=$(CONFIG)

hyperparameter-sweep: ## Run hyperparameter sweep
	python -m polymer_prediction.main --multirun model.hidden_channels=64,128,256 training.learning_rate=1e-4,1e-3,1e-2