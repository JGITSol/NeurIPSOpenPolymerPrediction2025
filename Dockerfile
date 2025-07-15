FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy requirements first for better caching
COPY --chown=app:app pyproject.toml ./
COPY --chown=app:app README.md ./

# Install Python dependencies
RUN pip install --user -e ".[dev]"

# Copy source code
COPY --chown=app:app . .

# Install the package
RUN pip install --user -e .

# Expose port for Jupyter or API
EXPOSE 8888

# Default command
CMD ["python", "-m", "polymer_prediction.main", "--help"]