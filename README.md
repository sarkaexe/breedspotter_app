# BreedSpotter 🐶

Klasyfikator ras psów w PyTorch z GUI w Streamlit + pełny pipeline trenowania.

## Demo lokalne
```bash
python -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install -e .[dev]

# Dane w data/train/<breed>/* oraz data/val/<breed>/*
make train
make run
```

## Struktura
- src/breedspotter/ — kod pakietu
- tests/ — testy
- MODEL_CARD.md — opis modelu

## Docker
```bash
docker build -t breedspotter .
docker run -p 8501:8501 -v $(pwd):/app breedspotter
```

MIT License
