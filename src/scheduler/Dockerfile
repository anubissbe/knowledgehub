FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    httpx==0.25.2 \
    apscheduler==3.10.4 \
    aiohttp==3.8.6 \
    asyncio

# Copy application code
COPY src/scheduler/ /app/
COPY src/shared/ /app/shared/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL=http://api:3000
ENV SCHEDULER_ENABLED=true
ENV REFRESH_SCHEDULE="0 2 * * 0"
ENV REFRESH_BATCH_SIZE=2
ENV REFRESH_DELAY_SECONDS=300

# Expose HTTP management port
EXPOSE 8080

# Health check via HTTP endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the scheduler
CMD ["python", "-m", "main"]