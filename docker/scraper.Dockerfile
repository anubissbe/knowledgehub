FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN pip install playwright && playwright install --with-deps chromium

# Copy source code
COPY src/scraper ./scraper
COPY src/shared ./shared

# Set Python path
ENV PYTHONPATH=/app

# Run the scraper
CMD ["python", "-m", "scraper.worker"]