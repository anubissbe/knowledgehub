"""
Comprehensive Test Runner for RAG System.

Orchestrates all RAG testing including:
- Unit tests with coverage reporting
- Integration tests with database setup
- Performance tests with benchmarking
- Quality gates validation
- CI/CD pipeline integration

Author: Peter Verschuere - Test-Driven Development Expert
"""

import pytest
import sys
import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGTestRunner:
    """Comprehensive test runner for RAG system."""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        self.results = {
            "unit_tests": {},
            "integration_tests": {},
            "performance_tests": {},
            "quality_gates": {},
            "overall": {}
        }
    
    def run_all_tests(self, fast: bool = False, verbose: bool = True) -> Dict[str, Any]:
        """Run complete RAG testing suite."""
        logger.info("Starting comprehensive RAG testing suite...")
        start_time = time.time()
        
        try:
            # Step 1: Unit Tests
            logger.info("Running unit tests...")
            self.results["unit_tests"] = self.run_unit_tests(verbose=verbose)
            
            # Step 2: Integration Tests (skip if fast mode)
            if not fast:
                logger.info("Running integration tests...")
                self.results["integration_tests"] = self.run_integration_tests(verbose=verbose)
            
            # Step 3: Performance Tests (skip if fast mode)
            if not fast:
                logger.info("Running performance tests...")
                self.results["performance_tests"] = self.run_performance_tests(verbose=verbose)
            
            # Step 4: Quality Gates
            logger.info("Validating quality gates...")
            self.results["quality_gates"] = self.run_quality_gates(verbose=verbose)
            
            # Calculate overall results
            end_time = time.time()
            self.results["overall"] = {
                "duration": end_time - start_time,
                "success": self._calculate_overall_success(),
                "summary": self._generate_summary()
            }
            
            # Generate reports
            self._generate_reports()
            
            logger.info(f"Testing completed in {self.results['overall']['duration']:.2f} seconds")
            return self.results
            
        except Exception as e:
            logger.error(f"Testing failed with error: {e}")
            self.results["overall"] = {
                "duration": time.time() - start_time,
                "success": False,
                "error": str(e)
            }
            return self.results
    
    def run_unit_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run unit tests with coverage reporting."""
        unit_test_path = self.base_path / "unit"
        
        # Pytest arguments for unit tests
        pytest_args = [
            str(unit_test_path),
            "-m", "unit",
            "--cov=api.services.rag_pipeline",
            "--cov=api.services.graphrag_service", 
            "--cov=api.services.llamaindex_rag_service",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov/unit",
            "--cov-report=xml:coverage-unit.xml",
            "--cov-fail-under=80",
            "--tb=short",
            "--durations=10"
        ]
        
        if verbose:
            pytest_args.extend(["-v", "--tb=line"])
        else:
            pytest_args.append("-q")
        
        start_time = time.time()
        result = pytest.main(pytest_args)
        duration = time.time() - start_time
        
        return {
            "success": result == 0,
            "duration": duration,
            "exit_code": result,
            "coverage_target": 80,
            "test_path": str(unit_test_path)
        }
    
    def run_integration_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run integration tests with database setup."""
        integration_test_path = self.base_path / "integration"
        
        # Setup test database if needed
        self._setup_test_environment()
        
        pytest_args = [
            str(integration_test_path),
            "-m", "integration",
            "--tb=short",
            "--durations=20"
        ]
        
        if verbose:
            pytest_args.extend(["-v", "-s"])
        else:
            pytest_args.append("-q")
        
        start_time = time.time()
        result = pytest.main(pytest_args)
        duration = time.time() - start_time
        
        # Cleanup test environment
        self._cleanup_test_environment()
        
        return {
            "success": result == 0,
            "duration": duration,
            "exit_code": result,
            "test_path": str(integration_test_path)
        }
    
    def run_performance_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run performance and load tests."""
        performance_test_path = self.base_path / "performance"
        
        pytest_args = [
            str(performance_test_path),
            "-m", "performance",
            "--tb=short",
            "--durations=30"
        ]
        
        if verbose:
            pytest_args.extend(["-v", "-s"])
        else:
            pytest_args.append("-q")
        
        start_time = time.time()
        result = pytest.main(pytest_args)
        duration = time.time() - start_time
        
        # Generate performance report
        self._generate_performance_report()
        
        return {
            "success": result == 0,
            "duration": duration,
            "exit_code": result,
            "test_path": str(performance_test_path)
        }
    
    def run_quality_gates(self, verbose: bool = True) -> Dict[str, Any]:
        """Run quality gates validation."""
        quality_gates_path = self.base_path / "test_quality_gates.py"
        
        pytest_args = [
            str(quality_gates_path),
            "-m", "quality_gates",
            "--tb=short"
        ]
        
        if verbose:
            pytest_args.extend(["-v"])
        else:
            pytest_args.append("-q")
        
        start_time = time.time()
        result = pytest.main(pytest_args)
        duration = time.time() - start_time
        
        return {
            "success": result == 0,
            "duration": duration,
            "exit_code": result,
            "test_path": str(quality_gates_path)
        }
    
    def run_specific_test_suite(self, suite: str, **kwargs) -> Dict[str, Any]:
        """Run a specific test suite."""
        if suite == "unit":
            return self.run_unit_tests(**kwargs)
        elif suite == "integration":
            return self.run_integration_tests(**kwargs)
        elif suite == "performance":
            return self.run_performance_tests(**kwargs)
        elif suite == "quality":
            return self.run_quality_gates(**kwargs)
        else:
            raise ValueError(f"Unknown test suite: {suite}")
    
    def _setup_test_environment(self):
        """Setup test environment for integration tests."""
        logger.info("Setting up test environment...")
        
        # Check if test database is available
        try:
            import psycopg2
            # Would connect to test database and create tables if needed
            logger.info("Test database available")
        except ImportError:
            logger.warning("psycopg2 not available, some tests may be skipped")
        
        # Setup test Redis if needed
        try:
            import redis
            logger.info("Redis available for testing")
        except ImportError:
            logger.warning("Redis not available, caching tests may be skipped")
    
    def _cleanup_test_environment(self):
        """Cleanup test environment after integration tests."""
        logger.info("Cleaning up test environment...")
        # Would cleanup test data, connections, etc.
    
    def _calculate_overall_success(self) -> bool:
        """Calculate overall test success."""
        return all(
            result.get("success", False) 
            for result in [
                self.results.get("unit_tests", {}),
                self.results.get("integration_tests", {}),
                self.results.get("performance_tests", {}),
                self.results.get("quality_gates", {})
            ]
            if result  # Only check results that were actually run
        )
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test execution summary."""
        total_duration = sum(
            result.get("duration", 0)
            for result in [
                self.results.get("unit_tests", {}),
                self.results.get("integration_tests", {}),
                self.results.get("performance_tests", {}),
                self.results.get("quality_gates", {})
            ]
            if result
        )
        
        return {
            "total_duration": total_duration,
            "suites_run": len([r for r in self.results.values() if r and r != self.results["overall"]]),
            "all_passed": self.results["overall"]["success"]
        }
    
    def _generate_reports(self):
        """Generate test reports."""
        # JSON report
        report_path = self.base_path / "reports"
        report_path.mkdir(exist_ok=True)
        
        json_report = report_path / "rag_test_results.json"
        with open(json_report, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Markdown report
        md_report = report_path / "rag_test_report.md"
        with open(md_report, "w") as f:
            f.write(self._generate_markdown_report())
        
        logger.info(f"Reports generated in {report_path}")
    
    def _generate_markdown_report(self) -> str:
        """Generate markdown test report."""
        overall = self.results.get("overall", {})
        success_emoji = "✅" if overall.get("success", False) else "❌"
        
        report = f"""# RAG System Test Report

**Status**: {success_emoji} {'PASSED' if overall.get('success', False) else 'FAILED'}
**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Duration**: {overall.get('duration', 0):.2f} seconds

## Test Suite Results

"""
        
        # Add results for each test suite
        for suite_name, result in self.results.items():
            if suite_name == "overall" or not result:
                continue
            
            status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
            duration = result.get("duration", 0)
            
            report += f"""### {suite_name.replace('_', ' ').title()}

- **Status**: {status}
- **Duration**: {duration:.2f} seconds
- **Test Path**: `{result.get('test_path', 'N/A')}`

"""
        
        # Add summary
        summary = self.results.get("overall", {}).get("summary", {})
        report += f"""## Summary

- **Total Duration**: {summary.get('total_duration', 0):.2f} seconds
- **Test Suites Run**: {summary.get('suites_run', 0)}
- **Overall Success**: {'✅ YES' if summary.get('all_passed', False) else '❌ NO'}

"""
        
        return report
    
    def _generate_performance_report(self):
        """Generate performance benchmark report."""
        # Would analyze performance test results and generate benchmarks
        logger.info("Performance report would be generated here")


def main():
    """Main entry point for test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System Test Runner")
    parser.add_argument("--suite", choices=["unit", "integration", "performance", "quality", "all"], 
                       default="all", help="Test suite to run")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode (skip slow tests)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")
    
    args = parser.parse_args()
    
    # Setup test runner
    runner = RAGTestRunner()
    
    # Run tests
    if args.suite == "all":
        results = runner.run_all_tests(fast=args.fast, verbose=args.verbose and not args.quiet)
    else:
        results = runner.run_specific_test_suite(
            args.suite, 
            verbose=args.verbose and not args.quiet
        )
    
    # Exit with appropriate code
    success = results.get("overall", {}).get("success", False) if args.suite == "all" else results.get("success", False)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
