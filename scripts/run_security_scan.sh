#!/bin/bash
# KnowledgeHub Local Security Scanner
# Run comprehensive security scans locally

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
REPORT_DIR="$PROJECT_ROOT/security-reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create report directory
mkdir -p "$REPORT_DIR"

echo -e "${BLUE}🔒 KnowledgeHub Security Scanner${NC}"
echo "======================================"
echo "Project: $PROJECT_ROOT"
echo "Reports: $REPORT_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Python security tools
install_security_tools() {
    echo -e "${YELLOW}📦 Installing security tools...${NC}"
    pip install -r requirements-security.txt || {
        echo -e "${YELLOW}Creating virtual environment for security tools...${NC}"
        python3 -m venv .venv-security
        source .venv-security/bin/activate
        pip install -r requirements-security.txt
    }
}

# Function to run Python security scans
run_python_scans() {
    echo -e "\n${BLUE}🐍 Python Security Scans${NC}"
    echo "========================"
    
    # Safety check
    if command_exists safety; then
        echo -e "\n${YELLOW}Running Safety check...${NC}"
        safety check --json --output "$REPORT_DIR/safety-report-$TIMESTAMP.json" || true
        safety check || echo -e "${RED}Safety found vulnerabilities${NC}"
    else
        echo -e "${RED}Safety not installed${NC}"
    fi
    
    # Pip-audit
    if command_exists pip-audit; then
        echo -e "\n${YELLOW}Running pip-audit...${NC}"
        pip-audit --format json --output "$REPORT_DIR/pip-audit-report-$TIMESTAMP.json" || true
        pip-audit || echo -e "${RED}Pip-audit found vulnerabilities${NC}"
    else
        echo -e "${RED}Pip-audit not installed${NC}"
    fi
    
    # Bandit
    if command_exists bandit; then
        echo -e "\n${YELLOW}Running Bandit security linter...${NC}"
        bandit -r src/ -f json -o "$REPORT_DIR/bandit-report-$TIMESTAMP.json" || true
        bandit -r src/ -ll || echo -e "${RED}Bandit found security issues${NC}"
    else
        echo -e "${RED}Bandit not installed${NC}"
    fi
    
    # License check
    if command_exists pip-licenses; then
        echo -e "\n${YELLOW}Checking Python licenses...${NC}"
        pip-licenses --format=json --output-file="$REPORT_DIR/python-licenses-$TIMESTAMP.json"
        pip-licenses --format=markdown
        
        # Check for problematic licenses
        problematic=$(pip-licenses --format=json | python3 -c "
import json, sys
data = json.load(sys.stdin)
issues = [f\"{pkg['Name']} ({pkg['License']})\" for pkg in data if any(lic in pkg.get('License', '') for lic in ['GPL', 'AGPL', 'LGPL'])]
if issues:
    print('\\n'.join(issues))
")
        
        if [ ! -z "$problematic" ]; then
            echo -e "${YELLOW}⚠️  Potentially problematic licenses:${NC}"
            echo "$problematic"
        fi
    fi
}

# Function to run JavaScript security scans
run_javascript_scans() {
    echo -e "\n${BLUE}📦 JavaScript Security Scans${NC}"
    echo "============================"
    
    # Find package.json files
    find "$PROJECT_ROOT" -name "package.json" -not -path "*/node_modules/*" | while read -r package_file; do
        dir=$(dirname "$package_file")
        echo -e "\n${YELLOW}Scanning $dir${NC}"
        
        cd "$dir"
        
        # NPM audit
        if [ -f "package-lock.json" ]; then
            echo "Running npm audit..."
            npm audit --json > "$REPORT_DIR/npm-audit-$(basename $dir)-$TIMESTAMP.json" || true
            npm audit || echo -e "${RED}npm audit found vulnerabilities${NC}"
        fi
        
        # Yarn audit
        if [ -f "yarn.lock" ]; then
            echo "Running yarn audit..."
            yarn audit --json > "$REPORT_DIR/yarn-audit-$(basename $dir)-$TIMESTAMP.json" || true
            yarn audit || echo -e "${RED}yarn audit found vulnerabilities${NC}"
        fi
        
        cd - > /dev/null
    done
}

# Function to run Docker security scans
run_docker_scans() {
    echo -e "\n${BLUE}🐳 Docker Security Scans${NC}"
    echo "========================"
    
    # Install trivy if not present
    if ! command_exists trivy; then
        echo -e "${YELLOW}Installing Trivy...${NC}"
        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
    fi
    
    if command_exists trivy; then
        # Scan filesystem
        echo -e "\n${YELLOW}Running Trivy filesystem scan...${NC}"
        trivy fs . --format json --output "$REPORT_DIR/trivy-fs-report-$TIMESTAMP.json" || true
        trivy fs . --severity HIGH,CRITICAL || echo -e "${RED}Trivy found vulnerabilities${NC}"
        
        # Scan Dockerfiles
        find "$PROJECT_ROOT" -name "Dockerfile*" -not -path "*/node_modules/*" | while read -r dockerfile; do
            echo -e "\n${YELLOW}Scanning $dockerfile${NC}"
            
            # Extract and scan base images
            grep "^FROM" "$dockerfile" | awk '{print $2}' | while read -r image; do
                echo "Scanning image: $image"
                trivy image "$image" --format json --output "$REPORT_DIR/trivy-image-$(echo $image | tr '/:' '_')-$TIMESTAMP.json" || true
                trivy image "$image" --severity HIGH,CRITICAL || echo -e "${RED}Found vulnerabilities in $image${NC}"
            done
        done
    else
        echo -e "${RED}Trivy not installed${NC}"
    fi
}

# Function to run secret scanning
run_secret_scans() {
    echo -e "\n${BLUE}🔑 Secret Scanning${NC}"
    echo "=================="
    
    if command_exists detect-secrets; then
        echo -e "\n${YELLOW}Running detect-secrets...${NC}"
        detect-secrets scan --baseline "$REPORT_DIR/secrets-baseline-$TIMESTAMP.json"
        detect-secrets audit "$REPORT_DIR/secrets-baseline-$TIMESTAMP.json" || echo -e "${GREEN}No secrets detected${NC}"
    else
        echo -e "${RED}detect-secrets not installed${NC}"
    fi
}

# Function to generate summary report
generate_summary() {
    echo -e "\n${BLUE}📊 Generating Summary Report${NC}"
    echo "============================"
    
    python3 "$PROJECT_ROOT/scripts/security_scan.py" --path "$PROJECT_ROOT" || {
        echo -e "${YELLOW}Running simplified summary...${NC}"
        
        SUMMARY_FILE="$REPORT_DIR/security-summary-$TIMESTAMP.md"
        
        cat > "$SUMMARY_FILE" << EOF
# Security Scan Summary
Date: $(date)
Project: KnowledgeHub

## Scan Results

### Python Security
$(ls -la "$REPORT_DIR"/*safety* 2>/dev/null | wc -l) Safety reports
$(ls -la "$REPORT_DIR"/*pip-audit* 2>/dev/null | wc -l) Pip-audit reports
$(ls -la "$REPORT_DIR"/*bandit* 2>/dev/null | wc -l) Bandit reports

### JavaScript Security
$(ls -la "$REPORT_DIR"/*npm-audit* 2>/dev/null | wc -l) NPM audit reports
$(ls -la "$REPORT_DIR"/*yarn-audit* 2>/dev/null | wc -l) Yarn audit reports

### Container Security
$(ls -la "$REPORT_DIR"/*trivy* 2>/dev/null | wc -l) Trivy reports

### License Compliance
$(ls -la "$REPORT_DIR"/*licenses* 2>/dev/null | wc -l) License reports

## Reports Location
All detailed reports are available in: $REPORT_DIR

## Next Steps
1. Review critical and high severity vulnerabilities
2. Update vulnerable dependencies
3. Apply security patches
4. Re-run scans after fixes
EOF
        
        echo -e "${GREEN}Summary saved to: $SUMMARY_FILE${NC}"
    }
}

# Main execution
main() {
    cd "$PROJECT_ROOT"
    
    # Check for virtual environment
    if [ -f ".venv-security/bin/activate" ]; then
        source .venv-security/bin/activate
    fi
    
    # Install tools if needed
    if [ "$1" == "--install" ]; then
        install_security_tools
    fi
    
    # Run scans based on arguments
    if [ $# -eq 0 ] || [ "$1" == "--all" ]; then
        # Run all scans
        run_python_scans
        run_javascript_scans
        run_docker_scans
        run_secret_scans
        generate_summary
    else
        # Run specific scans
        for arg in "$@"; do
            case $arg in
                --python)
                    run_python_scans
                    ;;
                --javascript|--js)
                    run_javascript_scans
                    ;;
                --docker)
                    run_docker_scans
                    ;;
                --secrets)
                    run_secret_scans
                    ;;
                --summary)
                    generate_summary
                    ;;
                *)
                    echo -e "${RED}Unknown option: $arg${NC}"
                    echo "Usage: $0 [--all|--python|--javascript|--docker|--secrets|--summary|--install]"
                    exit 1
                    ;;
            esac
        done
    fi
    
    echo -e "\n${GREEN}✅ Security scan complete!${NC}"
    echo -e "Reports saved to: ${BLUE}$REPORT_DIR${NC}"
    
    # Check if any critical issues were found
    if grep -r "CRITICAL\|critical" "$REPORT_DIR"/*$TIMESTAMP* 2>/dev/null | grep -v "CRITICAL:0"; then
        echo -e "\n${RED}⚠️  CRITICAL vulnerabilities found! Review reports immediately.${NC}"
        exit 1
    fi
}

# Run main function
main "$@"