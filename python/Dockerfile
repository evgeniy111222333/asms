# Python API Server Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source
COPY acms/ ./acms/

# Create data directories
RUN mkdir -p /data/acms/{logs,parquet,models}

EXPOSE 8000

CMD ["uvicorn", "acms.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
