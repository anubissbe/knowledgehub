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
COPY src/mcp ./mcp
COPY src/shared ./shared

# Set Python path
ENV PYTHONPATH=/app

# Run the MCP service
CMD ["python", "-m", "mcp.server"]