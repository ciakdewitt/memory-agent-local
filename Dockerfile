# Use Python 3.9 as base image (slim version for smaller size)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for persistent storage
RUN mkdir -p .memory

# Expose port for potential API
EXPOSE 8000

# Set the entrypoint command
# Default to CLI mode, can be overridden with different command
ENTRYPOINT ["python", "main.py"]

# Default command when no arguments are provided
CMD ["--list-threads"]