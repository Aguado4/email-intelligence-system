# Dockerfile for Classifier Service
# Multi-stage build for smaller image size

# ============= Builder Stage =============
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# ============= Runtime Stage =============
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Set Python to run in unbuffered mode (logs appear immediately)
ENV PYTHONUNBUFFERED=1

# Expose port (we'll add FastAPI later, but good practice)
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from config.settings import get_settings; get_settings()" || exit 1

# Run the test script for now (we'll change this to a service later)
CMD ["python", "test_classifier.py"]