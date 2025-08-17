#!/bin/bash
"""
KnowledgeHub Integration Testing - Quick Start Script
===================================================

This script provides easy execution of the comprehensive integration test suite
for the KnowledgeHub hybrid RAG system transformation.

Usage:
  ./run_integration_tests.sh [test_type]

Test Types:
  all          - Run all test suites (default)
  quick        - Run essential tests only
  integration  - Run integration tests only
  performance  - Run performance tests only
  workflows    - Run workflow validation only
  migration    - Run migration validation only
  orchestrated - Run full orchestrated test suite
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TEST_DIR="/opt/projects/knowledgehub"
LOG_DIR="$TEST_DIR/test_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Ensure we're in the right directory
cd "$TEST_DIR"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check dependencies
check_dependencies() {
    print_status $BLUE "🔍 Checking dependencies..."
    
    # Check Python 3
    if ! command -v python3 &> /dev/null; then
        print_status $RED "❌ Python 3 is required but not installed"
        exit 1
    fi
    
    # Check required Python packages
    local packages=("psycopg2" "redis" "requests" "aiohttp" "psutil")
    for package in "${packages[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            print_status $YELLOW "⚠️  Installing missing package: $package"
            pip3 install "$package" || {
                print_status $RED "❌ Failed to install $package"
                exit 1
            }
        fi
    done
    
    print_status $GREEN "✅ Dependencies check passed"
}

# Function to check system health
check_system_health() {
    print_status $BLUE "🏥 Checking system health..."
    
    # Check if Docker services are running
    if ! docker-compose ps | grep -q "Up"; then
        print_status $YELLOW "⚠️  Some Docker services may not be running"
        print_status $BLUE "🔄 Attempting to start services..."
        docker-compose up -d
        sleep 10
    fi
    
    # Check key service endpoints
    local endpoints=(
        "http://localhost:3000/health"
        "http://localhost:3100"
        "http://localhost:8002/health"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f -s "$endpoint" > /dev/null 2>&1; then
            print_status $GREEN "✅ $endpoint is responsive"
        else
            print_status $YELLOW "⚠️  $endpoint is not responsive"
        fi
    done
}

# Function to make scripts executable
setup_test_scripts() {
    print_status $BLUE "🔧 Setting up test scripts..."
    
    # Make test scripts executable
    local scripts=(
        "comprehensive_integration_test_suite.py"
        "performance_load_testing.py"
        "agent_workflow_validation.py"
        "migration_validation_comprehensive.py"
        "integration_test_orchestrator.py"
    )
    
    for script in "${scripts[@]}"; do
        if [ -f "$script" ]; then
            chmod +x "$script"
            print_status $GREEN "✅ Made $script executable"
        else
            print_status $RED "❌ Script not found: $script"
            exit 1
        fi
    done
}

# Function to run a single test suite
run_test_suite() {
    local test_name=$1
    local script_name=$2
    local log_file="$LOG_DIR/${test_name}_${TIMESTAMP}.log"
    
    print_status $BLUE "🚀 Running $test_name..."
    
    if python3 "$script_name" 2>&1 | tee "$log_file"; then
        print_status $GREEN "✅ $test_name completed successfully"
        return 0
    else
        print_status $RED "❌ $test_name failed"
        print_status $YELLOW "📄 Log file: $log_file"
        return 1
    fi
}

# Function to run quick tests
run_quick_tests() {
    print_status $BLUE "⚡ Running quick essential tests..."
    
    local failed_tests=0
    
    # Health check only from integration tests
    if ! python3 -c "
from comprehensive_integration_test_suite import IntegrationTestSuite
suite = IntegrationTestSuite()
suite.test_service_health_checks()
suite.test_database_connections()
print('Quick health check completed')
"; then
        ((failed_tests++))
    fi
    
    # Basic workflow test
    if ! python3 -c "
from agent_workflow_validation import AgentWorkflowValidator
validator = AgentWorkflowValidator()
validator.test_workflow_definitions_exist()
validator.test_agent_definitions_exist()
print('Quick workflow check completed')
"; then
        ((failed_tests++))
    fi
    
    if [ $failed_tests -eq 0 ]; then
        print_status $GREEN "✅ Quick tests passed"
        return 0
    else
        print_status $RED "❌ $failed_tests quick tests failed"
        return 1
    fi
}

# Function to run individual test categories
run_integration_tests() {
    run_test_suite "Integration Tests" "comprehensive_integration_test_suite.py"
}

run_performance_tests() {
    run_test_suite "Performance Tests" "performance_load_testing.py"
}

run_workflow_tests() {
    run_test_suite "Workflow Validation" "agent_workflow_validation.py"
}

run_migration_tests() {
    run_test_suite "Migration Validation" "migration_validation_comprehensive.py"
}

# Function to run orchestrated tests
run_orchestrated_tests() {
    print_status $BLUE "🎯 Running orchestrated test suite..."
    
    local log_file="$LOG_DIR/orchestrated_tests_${TIMESTAMP}.log"
    
    if python3 integration_test_orchestrator.py 2>&1 | tee "$log_file"; then
        print_status $GREEN "✅ Orchestrated tests completed successfully"
        
        # Find and display the report file
        local report_file=$(ls -t unified_integration_report_*.json 2>/dev/null | head -1)
        if [ -n "$report_file" ]; then
            print_status $BLUE "📊 Report generated: $report_file"
        fi
        
        return 0
    else
        print_status $RED "❌ Orchestrated tests failed"
        print_status $YELLOW "📄 Log file: $log_file"
        return 1
    fi
}

# Function to run all tests
run_all_tests() {
    print_status $BLUE "🎯 Running all test suites..."
    
    local failed_tests=0
    
    # Run each test suite
    run_integration_tests || ((failed_tests++))
    run_migration_tests || ((failed_tests++))
    run_workflow_tests || ((failed_tests++))
    run_performance_tests || ((failed_tests++))
    
    if [ $failed_tests -eq 0 ]; then
        print_status $GREEN "🎉 All test suites passed!"
        return 0
    else
        print_status $RED "❌ $failed_tests test suites failed"
        return 1
    fi
}

# Function to display help
show_help() {
    cat << EOF
KnowledgeHub Integration Testing - Quick Start Script

Usage: $0 [test_type]

Test Types:
  all          - Run all test suites (default)
  quick        - Run essential tests only (fastest)
  integration  - Run integration tests only
  performance  - Run performance tests only
  workflows    - Run workflow validation only
  migration    - Run migration validation only
  orchestrated - Run full orchestrated test suite with unified reporting

Examples:
  $0                    # Run all tests
  $0 quick             # Run quick health checks
  $0 orchestrated      # Run full orchestrated suite
  $0 integration       # Run integration tests only

Logs are saved to: $LOG_DIR/
EOF
}

# Function to generate summary report
generate_summary() {
    print_status $BLUE "📊 Generating test summary..."
    
    local report_files=(
        $(ls -t *_test_results_*.json 2>/dev/null || true)
        $(ls -t *_validation_*.json 2>/dev/null || true)
        $(ls -t *_report_*.json 2>/dev/null || true)
    )
    
    if [ ${#report_files[@]} -gt 0 ]; then
        print_status $GREEN "📄 Generated report files:"
        for file in "${report_files[@]:0:5}"; do  # Show up to 5 most recent
            echo "   - $file"
        done
    fi
    
    # Show log files
    local log_files=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -3 || true)
    if [ -n "$log_files" ]; then
        print_status $BLUE "📋 Recent log files:"
        echo "$log_files" | while read log_file; do
            echo "   - $log_file"
        done
    fi
}

# Main execution
main() {
    local test_type=${1:-"all"}
    
    print_status $BLUE "🚀 KnowledgeHub Integration Testing Started"
    print_status $BLUE "⏰ Timestamp: $TIMESTAMP"
    print_status $BLUE "🎯 Test Type: $test_type"
    echo
    
    # Setup
    check_dependencies
    check_system_health
    setup_test_scripts
    echo
    
    # Execute tests based on type
    case $test_type in
        "help"|"-h"|"--help")
            show_help
            exit 0
            ;;
        "quick")
            run_quick_tests
            ;;
        "integration")
            run_integration_tests
            ;;
        "performance")
            run_performance_tests
            ;;
        "workflows")
            run_workflow_tests
            ;;
        "migration")
            run_migration_tests
            ;;
        "orchestrated")
            run_orchestrated_tests
            ;;
        "all")
            run_all_tests
            ;;
        *)
            print_status $RED "❌ Unknown test type: $test_type"
            show_help
            exit 1
            ;;
    esac
    
    local test_result=$?
    echo
    
    # Generate summary
    generate_summary
    echo
    
    # Final status
    if [ $test_result -eq 0 ]; then
        print_status $GREEN "🎉 Testing completed successfully!"
        print_status $BLUE "📊 System transformation validation: PASSED"
    else
        print_status $RED "❌ Testing completed with failures"
        print_status $YELLOW "🔧 Review logs and fix issues before deployment"
    fi
    
    exit $test_result
}

# Execute main function with all arguments
main "$@"