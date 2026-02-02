FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY run.py .
COPY pyproject.toml .

RUN pip install -e .

EXPOSE 8000

CMD ["python", "run.py", "api"]
