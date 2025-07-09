FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/rag_processor ./rag_processor
COPY src/shared ./shared

# Copy health check script
COPY docker/rag_health_check.sh /tmp/health_check.sh
RUN chmod +x /tmp/health_check.sh

# Set Python path
ENV PYTHONPATH=/app

# Run the RAG processor
CMD ["python", "-m", "rag_processor.main"]