# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV (modern Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv is on PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency file and install
COPY requirements.txt .

# Use UV instead of pip to install all dependencies
RUN uv pip install --system -r requirements.txt

# Copy everything else
COPY . .

# Keep container running (no exit)
CMD ["tail", "-f", "/dev/null"]