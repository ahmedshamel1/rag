FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies (curl used by healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install Python dependencies
COPY requirements.txt .

# Install Python dependencies with cache mount for faster builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy backend code into the image
COPY . .

# Ensure database directory exists (named volume will be mounted here at runtime)
RUN mkdir -p /app/database /app/assets

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["uvicorn", "app_server:app", "--host", "0.0.0.0", "--port", "8000"]
