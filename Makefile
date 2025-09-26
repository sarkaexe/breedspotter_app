.PHONY: dev test lint format run train docker

dev:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .[dev]

test:
	pytest

lint:
	ruff check .

format:
	ruff check . --fix

run:
	streamlit run src/breedspotter/app.py

train:
	python -m breedspotter.train

docker:
	docker build -t breedspotter:latest .
