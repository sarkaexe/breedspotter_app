FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ruff.toml ./
COPY src ./src

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir ".[dev]"

EXPOSE 8501
CMD ["streamlit", "run", "src/breedspotter/app.py", "--server.headless=true", "--server.port=8501"]
