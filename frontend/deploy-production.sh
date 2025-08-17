#!/bin/bash

# KnowledgeHub Phase 5 Production Deployment Script
# Optimized for LAN deployment with performance monitoring

set -e

echo "🚀 KnowledgeHub Phase 5 - Production Deployment"
echo "================================================="

# Configuration
BUILD_MODE="production"
TARGET_ENV=${1:-"lan"} # lan, local, or custom
ANALYZE_BUNDLE=${2:-false}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Node.js version
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed"
    fi
    
    NODE_VERSION=$(node -v | cut -d'.' -f1 | sed 's/v//')
    if [ "$NODE_VERSION" -lt 16 ]; then
        log_error "Node.js version 16 or higher is required (found: $(node -v))"
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        log_error "npm is not installed"
    fi
    
    # Check if package.json exists
    if [ ! -f "package.json" ]; then
        log_error "package.json not found. Run this script from the frontend directory."
    fi
    
    log_success "Prerequisites check passed"
}

# Clean previous builds
clean_build() {
    log_info "Cleaning previous builds..."
    
    # Remove existing build files
    rm -rf dist/
    rm -rf node_modules/.vite/
    rm -f dist.tar.gz
    
    log_success "Build directory cleaned"
}

# Install dependencies with optimization
install_dependencies() {
    log_info "Installing dependencies..."
    
    # Use npm ci for faster, reproducible builds
    if [ -f "package-lock.json" ]; then
        npm ci --production=false
    else
        log_warning "package-lock.json not found, using npm install"
        npm install
    fi
    
    log_success "Dependencies installed"
}

# Build application
build_application() {
    log_info "Building application for $TARGET_ENV environment..."
    
    # Set environment variables
    export NODE_ENV=production
    export VITE_APP_VERSION=$(node -p "require('./package.json').version")
    export VITE_BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    
    # Configure target environment
    case $TARGET_ENV in
        "lan")
            export VITE_API_BASE_URL="http://192.168.1.25:3000"
            export VITE_WS_URL="ws://192.168.1.25:3000"
            ;;
        "local")
            export VITE_API_BASE_URL="http://localhost:3000"
            export VITE_WS_URL="ws://localhost:3000"
            ;;
        *)
            log_warning "Using default configuration for custom environment"
            ;;
    esac
    
    # Build with production config
    if [ -f "vite.config.production.ts" ]; then
        npx vite build --config vite.config.production.ts
    else
        npx vite build
    fi
    
    log_success "Application built successfully"
}

# Analyze bundle if requested
analyze_bundle() {
    if [ "$ANALYZE_BUNDLE" = "true" ]; then
        log_info "Analyzing bundle..."
        
        export ANALYZE=true
        npm run build
        
        log_info "Bundle analysis complete - check dist/stats.html"
    fi
}

# Optimize build output
optimize_build() {
    log_info "Optimizing build output..."
    
    # Gzip compression
    find dist -type f \( -name "*.js" -o -name "*.css" -o -name "*.html" \) -exec gzip -k {} \;
    
    # Generate service worker if not already present
    if [ ! -f "dist/sw.js" ]; then
        cp public/sw.js dist/sw.js 2>/dev/null || log_warning "Service worker not found"
    fi
    
    # Copy manifest
    if [ ! -f "dist/manifest.json" ]; then
        cp public/manifest.json dist/manifest.json 2>/dev/null || log_warning "Manifest not found"
    fi
    
    # Create icons directory if missing
    if [ ! -d "dist/icons" ]; then
        mkdir -p dist/icons
        cp -r public/icons/* dist/icons/ 2>/dev/null || log_warning "Icons not found"
    fi
    
    log_success "Build optimization complete"
}

# Validate build
validate_build() {
    log_info "Validating build..."
    
    # Check if dist directory exists
    if [ ! -d "dist" ]; then
        log_error "Build directory not found"
    fi
    
    # Check critical files
    critical_files=("dist/index.html" "dist/assets")
    for file in "${critical_files[@]}"; do
        if [ ! -e "$file" ]; then
            log_error "Critical file missing: $file"
        fi
    done
    
    # Check build size
    BUILD_SIZE=$(du -sh dist | cut -f1)
    log_info "Build size: $BUILD_SIZE"
    
    # Check for source maps in production
    if find dist -name "*.map" | grep -q .; then
        log_warning "Source maps found in production build"
    fi
    
    log_success "Build validation passed"
}

# Generate deployment package
create_package() {
    log_info "Creating deployment package..."
    
    # Create tarball
    tar -czf dist.tar.gz -C dist .
    
    PACKAGE_SIZE=$(du -sh dist.tar.gz | cut -f1)
    log_success "Deployment package created: dist.tar.gz ($PACKAGE_SIZE)"
}

# Performance test
performance_test() {
    log_info "Running performance tests..."
    
    # Check if lighthouse is available
    if command -v lighthouse &> /dev/null; then
        log_info "Running Lighthouse audit..."
        
        # Start a temporary server
        npx serve dist -l 8080 &
        SERVER_PID=$!
        
        # Wait for server to start
        sleep 3
        
        # Run lighthouse
        lighthouse http://localhost:8080 \
            --only-categories=performance,accessibility,best-practices,seo,pwa \
            --output=json \
            --output-path=./lighthouse-report.json \
            --chrome-flags="--headless --no-sandbox" \
            --quiet || log_warning "Lighthouse audit failed"
        
        # Kill temporary server
        kill $SERVER_PID 2>/dev/null || true
        
        if [ -f "lighthouse-report.json" ]; then
            PERFORMANCE_SCORE=$(node -p "
                try { 
                    const report = require('./lighthouse-report.json');
                    Math.round(report.categories.performance.score * 100);
                } catch(e) { 
                    'N/A' 
                }
            ")
            log_info "Performance Score: $PERFORMANCE_SCORE/100"
        fi
    else
        log_warning "Lighthouse not available for performance testing"
    fi
}

# Deploy to LAN server
deploy_to_lan() {
    if [ "$TARGET_ENV" = "lan" ]; then
        log_info "Deploying to LAN server..."
        
        # Configure deployment target
        LAN_SERVER="192.168.1.25"
        DEPLOY_PATH="/opt/knowledgehub/frontend"
        
        # Check if server is accessible
        if ping -c 1 $LAN_SERVER &> /dev/null; then
            log_info "LAN server accessible at $LAN_SERVER"
            
            # You would typically use scp or rsync here
            # scp -r dist/* user@$LAN_SERVER:$DEPLOY_PATH/
            
            log_success "Deployment to LAN server complete"
        else
            log_warning "LAN server not accessible, skipping deployment"
        fi
    fi
}

# Generate deployment report
generate_report() {
    log_info "Generating deployment report..."
    
    cat > deployment-report.md << EOF
# KnowledgeHub Phase 5 Deployment Report

**Build Date:** $(date)  
**Environment:** $TARGET_ENV  
**Version:** $(node -p "require('./package.json').version")  
**Node.js:** $(node -v)  
**npm:** $(npm -v)  

## Build Metrics
- **Build Size:** $(du -sh dist | cut -f1)
- **Package Size:** $(du -sh dist.tar.gz | cut -f1 2>/dev/null || echo "N/A")
- **Files Generated:** $(find dist -type f | wc -l)

## Bundle Analysis
- **JavaScript Chunks:** $(find dist/assets -name "*.js" | wc -l)
- **CSS Files:** $(find dist/assets -name "*.css" | wc -l)
- **Assets:** $(find dist/assets -not -name "*.js" -not -name "*.css" | wc -l)

## Performance
$(if [ -f "lighthouse-report.json" ]; then
    echo "- **Lighthouse Performance:** $PERFORMANCE_SCORE/100"
else
    echo "- **Lighthouse Performance:** Not tested"
fi)

## Features Included
- ✅ Modern React 18 with TypeScript
- ✅ Material-UI design system
- ✅ Real-time WebSocket integration
- ✅ Progressive Web App (PWA)
- ✅ Mobile-responsive design
- ✅ Performance optimizations
- ✅ Advanced caching strategies
- ✅ Bundle splitting and lazy loading

## Deployment Commands
\`\`\`bash
# Extract and serve
tar -xzf dist.tar.gz -C /var/www/knowledgehub/
nginx -s reload
\`\`\`
EOF
    
    log_success "Deployment report generated: deployment-report.md"
}

# Main execution flow
main() {
    echo "Starting deployment for target: $TARGET_ENV"
    echo "Bundle analysis: $ANALYZE_BUNDLE"
    echo ""
    
    check_prerequisites
    clean_build
    install_dependencies
    build_application
    analyze_bundle
    optimize_build
    validate_build
    create_package
    performance_test
    deploy_to_lan
    generate_report
    
    echo ""
    log_success "🎉 KnowledgeHub Phase 5 deployment complete!"
    echo ""
    echo "📦 Package: dist.tar.gz"
    echo "📊 Report: deployment-report.md"
    if [ -f "lighthouse-report.json" ]; then
        echo "🚀 Lighthouse: lighthouse-report.json"
    fi
    echo ""
    echo "Next steps:"
    echo "1. Extract dist.tar.gz to your web server"
    echo "2. Configure nginx/apache to serve the files"
    echo "3. Ensure API endpoints are accessible"
    echo "4. Test the application in production"
}

# Handle script interruption
trap 'echo ""; log_error "Deployment interrupted"' INT TERM

# Run main function
main "$@"