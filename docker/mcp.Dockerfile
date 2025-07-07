FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/mcp_server ./src/mcp_server
COPY src/shared ./src/shared

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 3002

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Health check removed - using docker-compose health check instead

# Run the MCP service
CMD ["python", "-m", "src.mcp_server.main"]