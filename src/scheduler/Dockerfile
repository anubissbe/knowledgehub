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

# No health check endpoint for scheduler

# Run the scheduler
CMD ["python", "-m", "main"]