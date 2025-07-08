# Security Dependency Scanning Documentation

## Overview

KnowledgeHub implements comprehensive security dependency scanning to identify and manage vulnerabilities in third-party dependencies across Python, JavaScript, and Docker containers. This system provides automated scanning, reporting, and remediation guidance.

## Features

### 🔍 Multi-Language Support
- **Python**: Dependencies in requirements.txt and installed packages
- **JavaScript**: NPM and Yarn dependencies
- **Docker**: Base images and container vulnerabilities
- **License Compliance**: Automated license checking

### 🚀 Automated Scanning
- **CI/CD Integration**: GitHub Actions workflow
- **Scheduled Scans**: Daily vulnerability checks
- **PR Scanning**: Automatic security checks on pull requests
- **Local Development**: Command-line scanning tools

### 📊 Comprehensive Reporting
- **JSON Reports**: Machine-readable vulnerability data
- **HTML Reports**: Human-friendly dashboards
- **SARIF Format**: GitHub Security tab integration
- **Summary Reports**: Executive-level overviews

## Architecture

### Scanning Tools

| Tool | Purpose | Languages | Integration |
|------|---------|-----------|-------------|
| **Safety** | Python vulnerability database | Python | CLI, CI/CD |
| **pip-audit** | Python package auditing | Python | CLI, CI/CD |
| **Bandit** | Security linter | Python | CLI, CI/CD |
| **npm audit** | JavaScript vulnerabilities | JavaScript | CLI, CI/CD |
| **yarn audit** | JavaScript vulnerabilities | JavaScript | CLI, CI/CD |
| **Trivy** | Container scanning | Docker, IaC | CLI, CI/CD |
| **Snyk** | Comprehensive scanning | All | CI/CD |
| **OWASP Dependency Check** | Known vulnerabilities | All | CI/CD |

### Workflow Architecture

```
┌─────────────────────────────────────────────┐
│           Trigger Sources                    │
│  • Push to main/develop                     │
│  • Pull Request                             │
│  • Daily Schedule (2 AM UTC)                │
│  • Manual Trigger                           │
└─────────────────────┬───────────────────────┘
                      │
┌─────────────────────▼───────────────────────┐
│         Security Scan Workflow               │
│  ┌─────────────┐  ┌──────────────┐         │
│  │   Python    │  │  JavaScript  │         │
│  │  Security   │  │   Security   │         │
│  └──────┬──────┘  └──────┬───────┘         │
│         │                 │                  │
│  ┌──────▼─────────────────▼──────┐         │
│  │      Docker Security          │         │
│  └──────────────┬────────────────┘         │
│                 │                           │
│  ┌──────────────▼────────────────┐         │
│  │    License Compliance         │         │
│  └──────────────┬────────────────┘         │
│                 │                           │
│  ┌──────────────▼────────────────┐         │
│  │     Security Summary          │         │
│  └───────────────────────────────┘         │
└─────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────┐
│              Outputs                         │
│  • GitHub Security Alerts                   │
│  • PR Comments                              │
│  • Action Artifacts                         │
│  • Summary Reports                          │
└─────────────────────────────────────────────┘
```

## Local Development

### Installation

1. **Install Security Tools**
```bash
# Install Python security tools
pip install -r requirements-security.txt

# Or use the setup script
./scripts/run_security_scan.sh --install
```

2. **Install Additional Tools**
```bash
# Install Trivy for container scanning
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Install Node.js tools
npm install -g npm-audit yarn audit
```

### Running Scans

#### Quick Scan (All)
```bash
./scripts/run_security_scan.sh --all
```

#### Python Security Scan
```bash
./scripts/run_security_scan.sh --python

# Or individually:
safety check
pip-audit
bandit -r src/
```

#### JavaScript Security Scan
```bash
./scripts/run_security_scan.sh --javascript

# Or manually:
npm audit
yarn audit
```

#### Docker Security Scan
```bash
./scripts/run_security_scan.sh --docker

# Or manually:
trivy fs .
trivy image python:3.11-slim
```

#### Secret Scanning
```bash
./scripts/run_security_scan.sh --secrets

# Or manually:
detect-secrets scan
```

### Using the Python Scanner

```bash
# Run comprehensive scan
python scripts/security_scan.py

# Scan specific directory
python scripts/security_scan.py --path /path/to/project

# Set severity threshold
python scripts/security_scan.py --severity-threshold high

# Generate specific output format
python scripts/security_scan.py --output-format json
```

## CI/CD Integration

### GitHub Actions Workflow

The security scanning is automatically triggered by:

1. **Push Events**: On main and develop branches
2. **Pull Requests**: Automatic scanning and commenting
3. **Schedule**: Daily at 2 AM UTC
4. **Manual**: Via GitHub Actions UI

### Workflow Jobs

#### 1. Python Security (`python-security`)
```yaml
- Safety vulnerability check
- pip-audit dependency audit
- Bandit security linting
- Upload reports as artifacts
```

#### 2. JavaScript Security (`javascript-security`)
```yaml
- npm audit for all package.json files
- yarn audit for yarn.lock files
- Recursive scanning
- Upload reports as artifacts
```

#### 3. Docker Security (`docker-security`)
```yaml
- Trivy filesystem scan
- Dockerfile base image scanning
- SARIF report generation
- GitHub Security integration
```

#### 4. OWASP Dependency Check (`dependency-check`)
```yaml
- Comprehensive vulnerability database
- All language support
- CVE correlation
- HTML and JSON reports
```

#### 5. License Compliance (`license-check`)
```yaml
- Python license analysis
- JavaScript license checking
- Problematic license detection
- Compliance reporting
```

#### 6. Security Summary (`security-summary`)
```yaml
- Consolidate all reports
- Generate summary
- PR commenting
- Issue creation for critical findings
```

## Report Formats

### JSON Report Structure
```json
{
  "scan_date": "2025-07-08T10:00:00Z",
  "project": "KnowledgeHub",
  "vulnerabilities": {
    "critical": [...],
    "high": [...],
    "medium": [...],
    "low": [...]
  },
  "summary": {
    "total_vulnerabilities": 15,
    "critical": 0,
    "high": 2,
    "medium": 5,
    "low": 8
  },
  "recommendations": [...]
}
```

### HTML Report
- Visual dashboard
- Sortable vulnerability tables
- Severity color coding
- Remediation guidance
- Export capabilities

### SARIF Report
- GitHub Security tab integration
- Code scanning alerts
- Inline annotations
- Automated PR checks

## Vulnerability Management

### Severity Levels

| Level | CVSS Score | Response Time | Action |
|-------|------------|---------------|--------|
| **Critical** | 9.0-10.0 | Immediate | Block deployment, immediate patch |
| **High** | 7.0-8.9 | 24 hours | Prioritize fix, consider workaround |
| **Medium** | 4.0-6.9 | 1 week | Schedule update, monitor |
| **Low** | 0.1-3.9 | 1 month | Track, update in next release |

### Remediation Process

1. **Identify**: Automated scanning detects vulnerability
2. **Assess**: Determine impact and exploitability
3. **Prioritize**: Based on severity and exposure
4. **Remediate**: Update, patch, or apply workaround
5. **Verify**: Re-scan to confirm fix
6. **Document**: Update security log

### Common Remediation Actions

#### Python Dependencies
```bash
# Update specific package
pip install --upgrade package-name==safe-version

# Update all packages
pip install --upgrade -r requirements.txt

# Use pip-tools for better management
pip-compile --upgrade requirements.in
```

#### JavaScript Dependencies
```bash
# NPM update
npm update package-name
npm audit fix

# Yarn update
yarn upgrade package-name
yarn upgrade-interactive
```

#### Docker Base Images
```dockerfile
# Update base image
FROM python:3.11-slim-bookworm  # Use specific, updated tags

# Multi-stage build for smaller attack surface
FROM python:3.11-slim as builder
# ... build steps ...
FROM python:3.11-slim-bookworm
COPY --from=builder /app /app
```

## Configuration

### Scanner Configuration

#### `security-config.yml`
```yaml
scanning:
  python:
    enabled: true
    tools:
      - safety
      - pip-audit
      - bandit
    severity_threshold: medium
  
  javascript:
    enabled: true
    tools:
      - npm-audit
      - yarn-audit
    auto_fix: false
  
  docker:
    enabled: true
    scan_base_images: true
    scan_filesystem: true
    severity_threshold: high
  
  secrets:
    enabled: true
    exclude_patterns:
      - "*.test"
      - "*.example"

reporting:
  formats:
    - json
    - html
    - sarif
  
  upload_artifacts: true
  
  notifications:
    slack:
      enabled: false
      webhook_url: ${SLACK_WEBHOOK}
      critical_only: true
    
    email:
      enabled: false
      recipients:
        - security@knowledgehub.com
```

### Exclusions

#### `.safety-policy.json`
```json
{
  "security": {
    "ignore-vulnerabilities": [
      {
        "vulnerability-id": "12345",
        "reason": "False positive - not applicable to our usage",
        "expires": "2025-12-31"
      }
    ]
  }
}
```

#### `.bandit`
```ini
[bandit]
exclude = /test,/scripts
skips = B101,B601
```

## Best Practices

### 1. Regular Scanning
- Run scans on every commit
- Daily scheduled scans
- Manual scans before releases
- Post-deployment verification

### 2. Dependency Management
```toml
# Use dependency pinning
flask==2.3.3  # Exact version
requests>=2.31.0,<3.0.0  # Version range

# Regular updates
# Schedule monthly dependency updates
# Test thoroughly after updates
```

### 3. Container Security
```dockerfile
# Use minimal base images
FROM python:3.11-slim-bookworm

# Run as non-root
RUN useradd -m -u 1000 appuser
USER appuser

# Copy only necessary files
COPY --chown=appuser:appuser requirements.txt .
COPY --chown=appuser:appuser src/ ./src/
```

### 4. License Compliance
- Maintain approved license list
- Review new dependencies
- Document license exceptions
- Regular compliance audits

### 5. Security Champions
- Assign security champions per team
- Regular security training
- Share vulnerability reports
- Celebrate security improvements

## Troubleshooting

### Common Issues

#### Scanner Installation Failed
```bash
# Use virtual environment
python -m venv venv-security
source venv-security/bin/activate
pip install -r requirements-security.txt
```

#### Permission Denied
```bash
# Fix script permissions
chmod +x scripts/run_security_scan.sh
chmod +x scripts/security_scan.py
```

#### Scan Timeout
```bash
# Increase timeout for large projects
export SCAN_TIMEOUT=3600  # 1 hour
./scripts/run_security_scan.sh --all
```

#### False Positives
```python
# Add inline suppressions
# nosec B101  # Suppress specific Bandit warning
assert True  # Safe assertion
```

### Debug Mode
```bash
# Enable verbose output
export DEBUG=1
./scripts/run_security_scan.sh --all

# Check individual tool versions
safety --version
pip-audit --version
bandit --version
trivy --version
```

## Metrics and Monitoring

### Key Metrics
- **Mean Time to Detect (MTTD)**: < 24 hours
- **Mean Time to Remediate (MTTR)**: < 7 days
- **Vulnerability Density**: < 1 per 1000 LOC
- **Patch Compliance**: > 95%
- **License Compliance**: 100%

### Dashboard Integration
```python
# Prometheus metrics
vulnerability_total = Gauge('security_vulnerabilities_total', 'Total vulnerabilities by severity', ['severity'])
scan_duration = Histogram('security_scan_duration_seconds', 'Security scan duration')
last_scan_timestamp = Gauge('security_last_scan_timestamp', 'Timestamp of last security scan')
```

### Alerting Rules
```yaml
groups:
  - name: security_alerts
    rules:
      - alert: CriticalVulnerabilityDetected
        expr: security_vulnerabilities_total{severity="critical"} > 0
        for: 5m
        annotations:
          summary: "Critical vulnerability detected"
          description: "{{ $value }} critical vulnerabilities found"
      
      - alert: SecurityScanFailed
        expr: time() - security_last_scan_timestamp > 86400
        annotations:
          summary: "Security scan not run in 24 hours"
```

## Integration Examples

### Pre-commit Hook
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pycqa/bandit
    rev: 1.7.6
    hooks:
      - id: bandit
        args: ['-r', 'src/']
  
  - repo: https://github.com/pyupio/safety
    rev: v3.0.1
    hooks:
      - id: safety
```

### Makefile Integration
```makefile
.PHONY: security-scan
security-scan:
	@echo "Running security scans..."
	./scripts/run_security_scan.sh --all

.PHONY: security-fix
security-fix:
	@echo "Attempting to fix vulnerabilities..."
	npm audit fix
	pip install --upgrade -r requirements.txt
```

### Docker Compose
```yaml
services:
  security-scanner:
    build:
      context: .
      dockerfile: Dockerfile.security
    volumes:
      - .:/app
      - ./security-reports:/reports
    command: /app/scripts/run_security_scan.sh --all
```

## Future Enhancements

### Planned Features
1. **AI-Powered Analysis**: ML-based vulnerability prediction
2. **Automated Remediation**: Auto-update for patch versions
3. **Supply Chain Security**: SBOM generation and tracking
4. **Runtime Protection**: Dynamic vulnerability detection
5. **Threat Intelligence**: Real-time threat feeds integration

### Roadmap
- **Q3 2025**: Implement SBOM generation
- **Q4 2025**: Add runtime security monitoring
- **Q1 2026**: ML-based vulnerability prediction
- **Q2 2026**: Automated remediation system

---

## Quick Reference

### Run All Scans
```bash
./scripts/run_security_scan.sh --all
```

### Check Python Only
```bash
safety check
pip-audit
bandit -r src/
```

### Check JavaScript Only
```bash
npm audit
yarn audit
```

### Check Containers
```bash
trivy fs .
trivy image <image-name>
```

### Generate Report
```bash
python scripts/security_scan.py
```

---

**Document Version**: 1.0.0  
**Last Updated**: July 8, 2025  
**Maintained By**: Security Team  
**Review Frequency**: Monthly

**Remember**: Security scanning is only as good as its follow-up actions. Scan regularly, remediate promptly, and keep dependencies updated.