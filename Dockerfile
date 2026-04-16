# === Stage 1 : install dependencies ===
FROM python:3.12-slim AS builder
WORKDIR /app

# Install system dependencies for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# === Stage 2 : runtime ===
FROM python:3.12-slim
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project files
COPY conf/base/ conf/base/
COPY src/ src/
COPY pyproject.toml .
COPY README.md .

# Default command : run the full pipeline
CMD ["kedro", "run"]
