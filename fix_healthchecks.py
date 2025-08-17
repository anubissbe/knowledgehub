#\!/usr/bin/env python3

import re

# Read the current docker-compose.yml
with open('docker-compose.yml', 'r') as f:
    content = f.read()

# Fix MinIO health check (has mc command available)
content = re.sub(
    r'test: \["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"\]',
    'test: ["CMD", "mc", "ready", "local"]',
    content
)

# Fix Qdrant health check (use TCP connection test)
content = re.sub(
    r'test: \["CMD", "curl", "-f", "http://localhost:6333/"\]',
    r'test: ["CMD-SHELL", "timeout 10s bash -c \'echo > /dev/tcp/localhost/6333\' || exit 1"]',
    content
)

# Fix Zep health check (use TCP connection test)
content = re.sub(
    r'test: \["CMD", "curl", "-f", "http://localhost:8000/healthz"\]',
    r'test: ["CMD-SHELL", "timeout 10s bash -c \'echo > /dev/tcp/localhost/8000\' || exit 1"]',
    content
)

# Fix AI service health check (use Python)
content = re.sub(
    r'test: \["CMD", "curl", "-f", "http://localhost:8000/health"\]',
    r'test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen(\'http://localhost:8000/health\', timeout=5)"]',
    content
)

# Add webui health check
content = re.sub(
    r'(  webui:\s+build: \.\/frontend\s+restart: unless-stopped\s+environment:\s+- VITE_API_URL=http://localhost:3000\s+- VITE_WS_URL=ws://localhost:3000\s+ports:\s+- "3100:80"\s+depends_on:\s+- api)',
    r'\1\n    healthcheck:\n      test: ["CMD-SHELL", "test -f /usr/share/nginx/html/index.html"]\n      interval: 30s\n      timeout: 10s\n      retries: 3',
    content,
    flags=re.MULTILINE | re.DOTALL
)

# Add nginx health check and fix depends_on
content = re.sub(
    r'(  # Nginx Reverse Proxy \(Optional\)\s+nginx:\s+image: nginx:alpine\s+restart: unless-stopped\s+ports:\s+- "80:80"\s+- "443:443"\s+volumes:\s+- \.\/nginx\.conf:\/etc\/nginx\/nginx\.conf:ro\s+- \.\/ssl:\/etc\/nginx\/ssl:ro\s+depends_on:\s+- api\s+- webui)',
    r'''\1
    healthcheck:
      test: ["CMD-SHELL", "nginx -t && ps aux | grep '[n]ginx: master process' || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3''',
    content,
    flags=re.MULTILINE | re.DOTALL
)

# Write the updated content
with open('docker-compose.yml', 'w') as f:
    f.write(content)

print("✅ Health checks fixed successfully")
