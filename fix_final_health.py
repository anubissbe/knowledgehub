#\!/usr/bin/env python3

import re

# Read the current docker-compose.yml
with open('docker-compose.yml', 'r') as f:
    content = f.read()

# Remove health checks for Qdrant and Zep since they don't have the necessary tools
# but are working fine (services are accessible from host)

# Remove Qdrant health check
content = re.sub(
    r'    healthcheck:\s*\n      test: \["CMD-SHELL", "nc -z localhost 6333 \|\| exit 1"\]\s*\n      interval: 10s\s*\n      timeout: 5s\s*\n      retries: 5\s*\n',
    '',
    content,
    flags=re.MULTILINE
)

# Remove Zep health check 
content = re.sub(
    r'    healthcheck:\s*\n      test: \["CMD-SHELL", "nc -z localhost 8000 \|\| exit 1"\]\s*\n      interval: 30s\s*\n      timeout: 10s\s*\n      retries: 3\s*\n',
    '',
    content,
    flags=re.MULTILINE
)

# Write the updated content
with open('docker-compose.yml', 'w') as f:
    f.write(content)

print("✅ Removed problematic health checks")
