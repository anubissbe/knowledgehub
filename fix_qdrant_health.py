#\!/usr/bin/env python3

import re

# Read the current docker-compose.yml
with open('docker-compose.yml', 'r') as f:
    content = f.read()

# Fix Qdrant health check (use simpler approach)
content = re.sub(
    r'test: \["CMD-SHELL", "timeout 10s bash -c .*?echo > /dev/tcp/localhost/6333.*?"\]',
    'test: ["CMD-SHELL", "nc -z localhost 6333 || exit 1"]',
    content
)

# Fix Zep health check too
content = re.sub(
    r'test: \["CMD-SHELL", "timeout 10s bash -c .*?echo > /dev/tcp/localhost/8000.*?"\]',
    'test: ["CMD-SHELL", "nc -z localhost 8000 || exit 1"]',
    content
)

# Write the updated content
with open('docker-compose.yml', 'w') as f:
    f.write(content)

print("✅ Health checks simplified")
