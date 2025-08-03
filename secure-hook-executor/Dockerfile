# Hardened Docker container for Claude Code hook execution
FROM python:3.11-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    openssl-dev

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only essential packages
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Final stage - minimal runtime
FROM python:3.11-alpine

# Install runtime dependencies only
RUN apk add --no-cache \
    libffi \
    openssl \
    curl

# Create non-root user
RUN adduser -D -s /bin/sh -u 1000 sandbox

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set up working directory
WORKDIR /app

# Copy hook script
COPY --chown=sandbox:sandbox hook-script.py /app/

# Security hardening
RUN chmod 500 /app/hook-script.py && \
    rm -rf /tmp/* /var/cache/apk/*

# Switch to non-root user
USER sandbox

# Set read-only root filesystem
# Additional security will be applied at runtime

ENTRYPOINT ["python", "/app/hook-script.py"]