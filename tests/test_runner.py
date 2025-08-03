#!/usr/bin/env python3
"""
Test runner for KnowledgeHub comprehensive testing.

Provides different test execution modes and reporting capabilities.
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from datetime import datetime


class KnowledgeHubTestRunner:
    """Comprehensive test runner for KnowledgeHub."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_root = Path(__file__).parent
        
    def run_smoke_tests(self) -> int:
        """Run quick smoke tests for basic functionality."""
        print("🚀 Running KnowledgeHub Smoke Tests...")
        
        smoke_args = [
            "-m", "smoke",
            "--tb=short",
            "--durations=5",
            "-v",
            str(self.test_root)
        ]
        
        return pytest.main(smoke_args)
    
    def run_unit_tests(self, coverage: bool = True) -> int:
        """Run all unit tests."""
        print("🧪 Running KnowledgeHub Unit Tests...")
        
        unit_args = [
            "-m", "unit",
            "--tb=short",
            "--durations=10",
            "-v"
        ]
        
        if coverage:
            unit_args.extend([
                "--cov=api",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/unit",
                "--cov-fail-under=85"
            ])
        
        unit_args.append(str(self.test_root / "unit"))
        return pytest.main(unit_args)
    
    def run_integration_tests(self) -> int:
        """Run integration tests."""
        print("🔗 Running KnowledgeHub Integration Tests...")
        
        integration_args = [
            "-m", "integration",
            "--tb=short",
            "--durations=10",
            "-v",
            str(self.test_root / "integration")
        ]
        
        return pytest.main(integration_args)
    
    def run_performance_tests(self) -> int:
        """Run performance and load tests."""
        print("⚡ Running KnowledgeHub Performance Tests...")
        
        performance_args = [
            "-m", "performance or load",
            "--tb=short",
            "--durations=0",  # Show all durations for performance analysis
            "-v",
            str(self.test_root / "performance")
        ]
        
        return pytest.main(performance_args)
    
    def run_e2e_tests(self) -> int:
        """Run end-to-end tests."""
        print("🎯 Running KnowledgeHub End-to-End Tests...")
        
        e2e_args = [
            "-m", "e2e",
            "--tb=long",  # More detail for e2e failures
            "--durations=10",
            "-v",
            str(self.test_root / "e2e")
        ]
        
        return pytest.main(e2e_args)
    
    def run_security_tests(self) -> int:
        """Run security-focused tests."""
        print("🔒 Running KnowledgeHub Security Tests...")
        
        security_args = [
            "-m", "security",
            "--tb=short",
            "-v",
            str(self.test_root)
        ]
        
        return pytest.main(security_args)
    
    def run_all_tests(self, skip_slow: bool = False) -> Dict[str, int]:
        """Run complete test suite."""
        print("🚀 Running Complete KnowledgeHub Test Suite...")
        
        results = {}
        test_suites = [
            ("smoke", self.run_smoke_tests),
            ("unit", lambda: self.run_unit_tests(coverage=True)),
            ("integration", self.run_integration_tests),
            ("e2e", self.run_e2e_tests),
        ]
        
        if not skip_slow:
            test_suites.extend([
                ("performance", self.run_performance_tests),
                ("security", self.run_security_tests)
            ])
        
        for suite_name, suite_runner in test_suites:
            print(f"\n{'='*60}")
            print(f"Running {suite_name.upper()} tests...")
            print(f"{'='*60}")
            
            start_time = time.time()
            result = suite_runner()
            end_time = time.time()
            
            results[suite_name] = {
                "exit_code": result,
                "duration": end_time - start_time,
                "status": "PASSED" if result == 0 else "FAILED"
            }
            
            print(f"\n{suite_name.upper()} tests {results[suite_name]['status']} "
                  f"in {results[suite_name]['duration']:.2f}s")
        
        return results
    
    def run_ci_tests(self) -> int:
        """Run tests suitable for CI/CD environment."""
        print("🤖 Running KnowledgeHub CI Tests...")
        
        ci_args = [
            "--tb=short",
            "--durations=10",
            "-v",
            "--cov=api",
            "--cov-report=xml",
            "--cov-report=term",
            "--cov-fail-under=90",
            "-m", "not slow and not load",  # Skip slow tests in CI
            str(self.test_root)
        ]
        
        return pytest.main(ci_args)
    
    def run_specific_test(self, test_path: str, markers: Optional[str] = None) -> int:
        """Run specific test file or directory."""
        print(f"🎯 Running specific test: {test_path}")
        
        args = ["-v", "--tb=short"]
        
        if markers:
            args.extend(["-m", markers])
        
        args.append(test_path)
        
        return pytest.main(args)
    
    def generate_test_report(self, results: Dict[str, Dict]) -> None:
        """Generate comprehensive test report."""
        report_path = self.project_root / "test_report.md"
        
        total_duration = sum(result["duration"] for result in results.values())
        passed_suites = [name for name, result in results.items() if result["status"] == "PASSED"]
        failed_suites = [name for name, result in results.items() if result["status"] == "FAILED"]
        
        report_content = f"""# KnowledgeHub Test Report
        
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary
- **Total Duration**: {total_duration:.2f} seconds
- **Test Suites Run**: {len(results)}
- **Passed**: {len(passed_suites)}
- **Failed**: {len(failed_suites)}

## Test Suite Results

| Suite | Status | Duration | Exit Code |
|-------|--------|----------|-----------|
"""
        
        for suite_name, result in results.items():
            status_emoji = "✅" if result["status"] == "PASSED" else "❌"
            report_content += f"| {suite_name.title()} | {status_emoji} {result['status']} | {result['duration']:.2f}s | {result['exit_code']} |\n"
        
        if failed_suites:
            report_content += f"\n## Failed Suites\n"
            for suite in failed_suites:
                report_content += f"- ❌ {suite.title()}\n"
        
        report_content += f"""
## Coverage Report

Coverage reports are generated in the following locations:
- HTML Report: `htmlcov/index.html`
- XML Report: `coverage.xml`
- Terminal output available in test logs

## Performance Metrics

Performance test results provide insights into:
- Memory operation throughput
- Session management performance
- AI service response times
- Concurrent operation handling
- Database query performance

## Recommendations

{"✅ All tests passed! System is ready for deployment." if not failed_suites else "❌ Some tests failed. Review failed suites before deployment."}

## Next Steps

1. Review any failed tests
2. Check coverage reports for untested code
3. Analyze performance metrics for bottlenecks
4. Update documentation if needed
"""
        
        with open(report_path, "w") as f:
            f.write(report_content)
        
        print(f"\n📊 Test report generated: {report_path}")
    
    def validate_environment(self) -> bool:
        """Validate test environment setup."""
        print("🔍 Validating test environment...")
        
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 8):
            issues.append("Python 3.8+ required")
        
        # Check required packages
        required_packages = [
            "pytest", "pytest-asyncio", "pytest-cov", "httpx", "psutil"
        ]
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                issues.append(f"Missing package: {package}")
        
        # Check test database
        test_db_url = os.getenv("TEST_DATABASE_URL")
        if not test_db_url:
            issues.append("TEST_DATABASE_URL environment variable not set")
        
        # Check project structure
        required_dirs = ["api", "tests/unit", "tests/integration", "tests/e2e"]
        for dir_path in required_dirs:
            if not (self.project_root / dir_path).exists():
                issues.append(f"Missing directory: {dir_path}")
        
        if issues:
            print("❌ Environment validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print("✅ Environment validation passed")
        return True


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(
        description="KnowledgeHub Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py --smoke              # Quick smoke tests
  python test_runner.py --unit               # Unit tests only
  python test_runner.py --integration        # Integration tests only
  python test_runner.py --performance        # Performance tests only
  python test_runner.py --e2e                # End-to-end tests only
  python test_runner.py --all                # All tests
  python test_runner.py --all --skip-slow    # All tests except slow ones
  python test_runner.py --ci                 # CI-suitable tests
  python test_runner.py --specific tests/unit/test_memory_service.py
  python test_runner.py --validate           # Validate environment only
        """
    )
    
    # Test suite options
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests")
    parser.add_argument("--security", action="store_true", help="Run security tests")
    parser.add_argument("--all", action="store_true", help="Run all test suites")
    parser.add_argument("--ci", action="store_true", help="Run CI/CD suitable tests")
    
    # Specific test options
    parser.add_argument("--specific", help="Run specific test file or directory")
    parser.add_argument("--markers", help="Pytest markers for specific tests")
    
    # Configuration options
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow-running tests")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    parser.add_argument("--validate", action="store_true", help="Validate environment only")
    parser.add_argument("--report", action="store_true", help="Generate detailed test report")
    
    args = parser.parse_args()
    
    runner = KnowledgeHubTestRunner()
    
    # Validate environment first
    if args.validate:
        return 0 if runner.validate_environment() else 1
    
    if not runner.validate_environment():
        print("❌ Environment validation failed. Use --validate for details.")
        return 1
    
    # Determine which tests to run
    if args.specific:
        return runner.run_specific_test(args.specific, args.markers)
    elif args.smoke:
        return runner.run_smoke_tests()
    elif args.unit:
        return runner.run_unit_tests(coverage=not args.no_coverage)
    elif args.integration:
        return runner.run_integration_tests()
    elif args.performance:
        return runner.run_performance_tests()
    elif args.e2e:
        return runner.run_e2e_tests()
    elif args.security:
        return runner.run_security_tests()
    elif args.ci:
        return runner.run_ci_tests()
    elif args.all:
        results = runner.run_all_tests(skip_slow=args.skip_slow)
        
        if args.report:
            runner.generate_test_report(results)
        
        # Print summary
        print(f"\n{'='*60}")
        print("TEST SUITE SUMMARY")
        print(f"{'='*60}")
        
        for suite_name, result in results.items():
            status = "✅" if result["status"] == "PASSED" else "❌"
            print(f"{status} {suite_name.upper()}: {result['status']} ({result['duration']:.2f}s)")
        
        # Return non-zero if any suite failed
        return 0 if all(r["exit_code"] == 0 for r in results.values()) else 1
    else:
        # Default to smoke tests
        print("No specific test suite specified. Running smoke tests...")
        return runner.run_smoke_tests()


if __name__ == "__main__":
    sys.exit(main())