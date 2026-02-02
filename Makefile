.PHONY: install test lint run-demo run-api docker-build docker-run clean

install:
	pip install -e .

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/

run-demo:
	python run.py demo

run-api:
	python run.py api

run-eval:
	python run.py eval

docker-build:
	docker build -t customer-support-chatbot .

docker-run:
	docker-compose up

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage coverage.xml
