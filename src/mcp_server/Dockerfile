FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose the MCP server port
EXPOSE 3002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 3002)); s.close()"

# Run the MCP server
CMD ["python", "-m", "mcp_server.main"]