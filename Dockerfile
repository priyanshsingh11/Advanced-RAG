# --- Build Stage ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gfortran \
    libopenblas-dev \
    pkg-config \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a virtualenv or just globally in the builder
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# --- Final Stage ---
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (e.g., for PDF processing or scientific libs)
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and scripts
COPY ./app ./app
COPY ./scripts ./scripts
COPY .env .env

# Pre-download AI models
# We set a cache directory to ensure they are stored within the image
ENV HF_HOME=/app/models/huggingface
ENV FASTEMBED_CACHE_PATH=/app/models/fastembed
RUN mkdir -p /app/models/huggingface /app/models/fastembed /app/data /app/storage
RUN python scripts/download_models.py

# Set environment variables for the models to use the cached paths
ENV HF_HOME=/app/models/huggingface
ENV FASTEMBED_CACHE_PATH=/app/models/fastembed

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
