#!/usr/bin/env python3
"""
KnowledgeHub Integration Test Orchestrator
=========================================

Master orchestrator for running all integration test suites and producing
a unified comprehensive report for the hybrid RAG system transformation.

Test Suites Included:
1. Comprehensive Integration Testing
2. Performance and Load Testing  
3. Agent Workflow Validation
4. Migration Validation
5. System Health Monitoring

Produces unified dashboard and recommendations.
"""

import asyncio
import subprocess
import json
import time
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import concurrent.futures
import tempfile

# Test orchestrator configuration
ORCHESTRATOR_CONFIG = {
    'test_suites': [
        {
            'name': 'Comprehensive Integration Tests',
            'script': 'comprehensive_integration_test_suite.py',
            'category': 'integration',
            'timeout_minutes': 15,
            'critical': True
        },
        {
            'name': 'Performance and Load Tests',
            'script': 'performance_load_testing.py',
            'category': 'performance',
            'timeout_minutes': 20,
            'critical': True
        },
        {
            'name': 'Agent Workflow Validation',
            'script': 'agent_workflow_validation.py',
            'category': 'workflows',
            'timeout_minutes': 10,
            'critical': True
        },
        {
            'name': 'Migration Validation',
            'script': 'migration_validation_comprehensive.py',
            'category': 'migration',
            'timeout_minutes': 8,
            'critical': True
        }
    ],
    'parallel_execution': True,
    'max_concurrent_suites': 2,
    'health_check_interval': 30,
    'report_formats': ['json', 'html', 'summary']
}

@dataclass
class TestSuiteResult:
    """Test suite execution result"""
    suite_name: str
    category: str
    status: str  # 'passed', 'failed', 'timeout', 'error'
    duration_seconds: int
    exit_code: int
    output: str
    error_output: str
    report_file: Optional[str] = None
    report_data: Optional[Dict[str, Any]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class IntegrationTestOrchestrator:
    """Master orchestrator for all integration test suites"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.suite_results: List[TestSuiteResult] = []
        self.start_time = None
        self.execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _run_test_suite(self, suite_config: Dict[str, Any]) -> TestSuiteResult:
        """Run individual test suite"""
        suite_name = suite_config['name']
        script_name = suite_config['script']
        timeout_seconds = suite_config['timeout_minutes'] * 60
        
        self.logger.info(f"🚀 Starting {suite_name}...")
        
        start_time = time.time()
        
        try:
            # Run test suite
            result = subprocess.run(
                ['python3', script_name],
                cwd='/opt/projects/knowledgehub',
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            
            duration = int(time.time() - start_time)
            
            # Determine status based on exit code
            if result.returncode == 0:
                status = 'passed'
            else:
                status = 'failed'
            
            # Try to find and load report file
            report_file = None
            report_data = None
            
            # Look for JSON report files
            import glob
            pattern = f"*{suite_config['category']}*{self.execution_id[:8]}*.json"
            json_files = glob.glob(f"/opt/projects/knowledgehub/{pattern}")
            
            if not json_files:
                # Look for any recent JSON files
                recent_patterns = [
                    f"*{suite_config['category']}*report*.json",
                    f"*{script_name.replace('.py', '')}*.json"
                ]
                for pattern in recent_patterns:
                    json_files = glob.glob(f"/opt/projects/knowledgehub/{pattern}")
                    if json_files:
                        break
            
            if json_files:
                # Get most recent file
                report_file = max(json_files, key=os.path.getctime)
                try:
                    with open(report_file, 'r') as f:
                        report_data = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Could not load report data from {report_file}: {e}")
            
            result_obj = TestSuiteResult(
                suite_name=suite_name,
                category=suite_config['category'],
                status=status,
                duration_seconds=duration,
                exit_code=result.returncode,
                output=result.stdout,
                error_output=result.stderr,
                report_file=report_file,
                report_data=report_data
            )
            
            # Log result
            status_emoji = "✅" if status == 'passed' else "❌"
            self.logger.info(f"{status_emoji} {suite_name} completed ({duration}s) - {status}")
            
            return result_obj
            
        except subprocess.TimeoutExpired:
            duration = int(time.time() - start_time)
            self.logger.error(f"⏰ {suite_name} timed out after {duration}s")
            
            return TestSuiteResult(
                suite_name=suite_name,
                category=suite_config['category'],
                status='timeout',
                duration_seconds=duration,
                exit_code=-1,
                output="",
                error_output=f"Test suite timed out after {timeout_seconds} seconds",
                report_file=None,
                report_data=None
            )
            
        except Exception as e:
            duration = int(time.time() - start_time)
            self.logger.error(f"💥 {suite_name} failed with exception: {e}")
            
            return TestSuiteResult(
                suite_name=suite_name,
                category=suite_config['category'],
                status='error',
                duration_seconds=duration,
                exit_code=-2,
                output="",
                error_output=str(e),
                report_file=None,
                report_data=None
            )
    
    def _run_parallel_tests(self) -> List[TestSuiteResult]:
        """Run test suites in parallel"""
        test_suites = ORCHESTRATOR_CONFIG['test_suites']
        max_workers = ORCHESTRATOR_CONFIG['max_concurrent_suites']
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all test suites
            future_to_suite = {
                executor.submit(self._run_test_suite, suite): suite
                for suite in test_suites
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_suite):
                suite = future_to_suite[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Test suite {suite['name']} generated an exception: {e}")
                    # Create error result
                    error_result = TestSuiteResult(
                        suite_name=suite['name'],
                        category=suite['category'],
                        status='error',
                        duration_seconds=0,
                        exit_code=-3,
                        output="",
                        error_output=str(e)
                    )
                    results.append(error_result)
        
        return results
    
    def _run_sequential_tests(self) -> List[TestSuiteResult]:
        """Run test suites sequentially"""
        results = []
        
        for suite_config in ORCHESTRATOR_CONFIG['test_suites']:
            result = self._run_test_suite(suite_config)
            results.append(result)
            
            # Brief pause between suites
            time.sleep(5)
        
        return results
    
    def _analyze_overall_health(self) -> Dict[str, Any]:
        """Analyze overall system health from all test results"""
        total_suites = len(self.suite_results)
        passed_suites = len([r for r in self.suite_results if r.status == 'passed'])
        failed_suites = len([r for r in self.suite_results if r.status == 'failed'])
        error_suites = len([r for r in self.suite_results if r.status in ['timeout', 'error']])
        
        # Critical vs non-critical analysis
        critical_suites = [r for r in self.suite_results if self._is_critical_suite(r.category)]
        critical_passed = len([r for r in critical_suites if r.status == 'passed'])
        
        # Calculate health score
        # Critical suites have 70% weight, non-critical 30%
        critical_weight = 0.7
        non_critical_weight = 0.3
        
        critical_score = (critical_passed / len(critical_suites)) if critical_suites else 1.0
        non_critical_suites = [r for r in self.suite_results if not self._is_critical_suite(r.category)]
        non_critical_passed = len([r for r in non_critical_suites if r.status == 'passed'])
        non_critical_score = (non_critical_passed / len(non_critical_suites)) if non_critical_suites else 1.0
        
        overall_health_score = (critical_score * critical_weight + non_critical_score * non_critical_weight) * 100
        
        # Determine overall status
        if critical_passed == len(critical_suites) and passed_suites == total_suites:
            overall_status = 'EXCELLENT'
        elif critical_passed == len(critical_suites):
            overall_status = 'GOOD'
        elif critical_passed >= len(critical_suites) * 0.75:
            overall_status = 'FAIR'
        else:
            overall_status = 'POOR'
        
        # Performance analysis
        performance_data = self._extract_performance_data()
        
        # System readiness assessment
        readiness = self._assess_system_readiness()
        
        return {
            'overall_status': overall_status,
            'health_score': overall_health_score,
            'suite_summary': {
                'total': total_suites,
                'passed': passed_suites,
                'failed': failed_suites,
                'errors': error_suites
            },
            'critical_systems': {
                'total': len(critical_suites),
                'passed': critical_passed,
                'status': 'HEALTHY' if critical_passed == len(critical_suites) else 'DEGRADED'
            },
            'performance_summary': performance_data,
            'system_readiness': readiness,
            'timestamp': datetime.now().isoformat()
        }
    
    def _is_critical_suite(self, category: str) -> bool:
        """Check if test suite category is critical"""
        critical_categories = ['integration', 'migration', 'workflows']
        return category in critical_categories
    
    def _extract_performance_data(self) -> Dict[str, Any]:
        """Extract performance data from test results"""
        performance_data = {
            'api_performance': {},
            'database_performance': {},
            'workflow_performance': {},
            'load_test_results': {}
        }
        
        for result in self.suite_results:
            if result.report_data:
                # Extract performance metrics based on suite type
                if result.category == 'performance':
                    if 'results' in result.report_data:
                        load_results = result.report_data['results']
                        performance_data['load_test_results'] = {
                            'max_throughput': max([r.get('throughput_rps', 0) for r in load_results]),
                            'min_error_rate': min([r.get('error_rate_percent', 100) for r in load_results]),
                            'max_concurrent_users': max([r.get('concurrent_users', 0) for r in load_results])
                        }
                
                elif result.category == 'integration':
                    if 'detailed_results' in result.report_data:
                        api_results = [r for r in result.report_data['detailed_results'] 
                                     if 'API Performance' in r.get('name', '')]
                        if api_results:
                            performance_data['api_performance'] = {
                                'avg_response_time': sum(r.get('duration_ms', 0) for r in api_results) / len(api_results),
                                'success_rate': len([r for r in api_results if r.get('status') == 'passed']) / len(api_results) * 100
                            }
                
                elif result.category == 'migration':
                    if 'detailed_results' in result.report_data:
                        db_results = [r for r in result.report_data['detailed_results'] 
                                    if 'Query Performance' in r.get('test_name', '')]
                        if db_results:
                            performance_data['database_performance'] = {
                                'query_performance': db_results[0].get('details', {})
                            }
                
                elif result.category == 'workflows':
                    if 'workflow_performance' in result.report_data:
                        performance_data['workflow_performance'] = result.report_data['workflow_performance']
        
        return performance_data
    
    def _assess_system_readiness(self) -> Dict[str, Any]:
        """Assess overall system readiness for production"""
        readiness_factors = {
            'migration_complete': False,
            'core_functionality': False,
            'performance_acceptable': False,
            'workflows_operational': False,
            'data_integrity': False
        }
        
        readiness_score = 0
        issues = []
        
        # Check each test suite result
        for result in self.suite_results:
            if result.status == 'passed':
                if result.category == 'migration':
                    readiness_factors['migration_complete'] = True
                    readiness_factors['data_integrity'] = True
                    readiness_score += 20
                
                elif result.category == 'integration':
                    readiness_factors['core_functionality'] = True
                    readiness_score += 25
                
                elif result.category == 'performance':
                    readiness_factors['performance_acceptable'] = True
                    readiness_score += 20
                
                elif result.category == 'workflows':
                    readiness_factors['workflows_operational'] = True
                    readiness_score += 15
                    
            else:
                # Record issues
                if result.category == 'migration':
                    issues.append("Migration validation failed - data integrity at risk")
                elif result.category == 'integration':
                    issues.append("Core integration tests failed - basic functionality compromised")
                elif result.category == 'performance':
                    issues.append("Performance tests failed - system may not handle production load")
                elif result.category == 'workflows':
                    issues.append("Workflow validation failed - agent system may not function properly")
        
        # Additional factors
        if len([r for r in self.suite_results if r.status == 'passed']) == len(self.suite_results):
            readiness_score += 20  # Bonus for all tests passing
        
        # Determine readiness level
        if readiness_score >= 90:
            readiness_level = 'PRODUCTION_READY'
        elif readiness_score >= 70:
            readiness_level = 'STAGING_READY'
        elif readiness_score >= 50:
            readiness_level = 'DEVELOPMENT_COMPLETE'
        else:
            readiness_level = 'NOT_READY'
        
        return {
            'readiness_level': readiness_level,
            'readiness_score': readiness_score,
            'readiness_factors': readiness_factors,
            'blocking_issues': issues,
            'recommendation': self._get_readiness_recommendation(readiness_level, issues)
        }
    
    def _get_readiness_recommendation(self, readiness_level: str, issues: List[str]) -> str:
        """Get recommendation based on readiness assessment"""
        if readiness_level == 'PRODUCTION_READY':
            return "✅ System is ready for production deployment"
        elif readiness_level == 'STAGING_READY':
            return "🟡 System is ready for staging environment - monitor closely"
        elif readiness_level == 'DEVELOPMENT_COMPLETE':
            return "🟠 Development phase complete - additional testing required before production"
        else:
            return "🔴 System not ready for deployment - critical issues must be resolved"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on all test results"""
        recommendations = []
        
        failed_suites = [r for r in self.suite_results if r.status in ['failed', 'timeout', 'error']]
        
        if not failed_suites:
            recommendations.append("🎉 All test suites passed! System transformation successful.")
            recommendations.append("📋 Proceed with production deployment planning")
            recommendations.append("📊 Set up continuous monitoring for production environment")
        else:
            recommendations.append("🚨 Test failures detected - immediate attention required")
            
            # Specific recommendations based on failed suites
            for suite in failed_suites:
                if suite.category == 'migration':
                    recommendations.append(f"🔄 Fix migration issues in {suite.suite_name}")
                    recommendations.append("   → Verify database schema and data integrity")
                    recommendations.append("   → Check migration rollback procedures")
                
                elif suite.category == 'integration':
                    recommendations.append(f"🔗 Resolve integration issues in {suite.suite_name}")
                    recommendations.append("   → Check service connectivity and health")
                    recommendations.append("   → Verify API endpoints and responses")
                
                elif suite.category == 'performance':
                    recommendations.append(f"⚡ Address performance issues in {suite.suite_name}")
                    recommendations.append("   → Optimize system resources and configuration")
                    recommendations.append("   → Review database query performance")
                
                elif suite.category == 'workflows':
                    recommendations.append(f"🤖 Fix workflow issues in {suite.suite_name}")
                    recommendations.append("   → Verify agent definitions and workflow configurations")
                    recommendations.append("   → Check LangGraph orchestration setup")
        
        # Performance recommendations
        performance_data = self._extract_performance_data()
        if performance_data['load_test_results']:
            load_results = performance_data['load_test_results']
            if load_results.get('min_error_rate', 0) > 5:
                recommendations.append("📈 High error rate detected - investigate system stability")
            if load_results.get('max_throughput', 0) < 10:
                recommendations.append("🚀 Low throughput detected - consider performance optimization")
        
        return recommendations
    
    def generate_unified_report(self) -> Dict[str, Any]:
        """Generate comprehensive unified report"""
        total_duration = int(time.time() - self.start_time) if self.start_time else 0
        
        # Overall health analysis
        health_analysis = self._analyze_overall_health()
        
        # Detailed results by category
        category_results = {}
        for result in self.suite_results:
            if result.category not in category_results:
                category_results[result.category] = {
                    'suites': [],
                    'total': 0,
                    'passed': 0,
                    'failed': 0
                }
            
            category_results[result.category]['suites'].append({
                'name': result.suite_name,
                'status': result.status,
                'duration_seconds': result.duration_seconds,
                'report_file': result.report_file
            })
            category_results[result.category]['total'] += 1
            if result.status == 'passed':
                category_results[result.category]['passed'] += 1
            else:
                category_results[result.category]['failed'] += 1
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Create unified report
        unified_report = {
            'execution_metadata': {
                'execution_id': self.execution_id,
                'start_time': self.start_time,
                'total_duration_seconds': total_duration,
                'timestamp': datetime.now().isoformat(),
                'orchestrator_version': '1.0.0'
            },
            'executive_summary': {
                'overall_status': health_analysis['overall_status'],
                'health_score': health_analysis['health_score'],
                'system_readiness': health_analysis['system_readiness'],
                'critical_systems_status': health_analysis['critical_systems']['status'],
                'total_test_suites': len(self.suite_results),
                'passed_suites': len([r for r in self.suite_results if r.status == 'passed']),
                'failed_suites': len([r for r in self.suite_results if r.status != 'passed'])
            },
            'detailed_analysis': {
                'health_analysis': health_analysis,
                'category_breakdown': category_results,
                'performance_summary': health_analysis['performance_summary']
            },
            'test_suite_results': [
                {
                    'suite_name': r.suite_name,
                    'category': r.category,
                    'status': r.status,
                    'duration_seconds': r.duration_seconds,
                    'exit_code': r.exit_code,
                    'report_file': r.report_file,
                    'has_detailed_report': r.report_data is not None,
                    'timestamp': r.timestamp
                }
                for r in self.suite_results
            ],
            'recommendations': recommendations,
            'next_steps': self._generate_next_steps(),
            'appendix': {
                'configuration': ORCHESTRATOR_CONFIG,
                'environment_info': self._collect_environment_info()
            }
        }
        
        return unified_report
    
    def _generate_next_steps(self) -> List[str]:
        """Generate next steps based on test results"""
        next_steps = []
        
        failed_suites = [r for r in self.suite_results if r.status != 'passed']
        
        if not failed_suites:
            next_steps.extend([
                "1. 📋 Review unified test report and performance metrics",
                "2. 🚀 Plan production deployment with monitoring setup",
                "3. 📊 Establish baseline performance metrics for monitoring",
                "4. 🔄 Set up automated regression testing for future changes",
                "5. 📚 Document deployment procedures and runbooks"
            ])
        else:
            next_steps.extend([
                "1. 🚨 Address all failed test suites immediately",
                "2. 🔍 Investigate root causes of test failures",
                "3. 🛠️ Implement fixes based on detailed test reports",
                "4. 🔄 Re-run failed test suites to validate fixes",
                "5. 📋 Update documentation based on findings"
            ])
        
        return next_steps
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect environment information"""
        try:
            return {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': os.getcwd(),
                'available_scripts': [f for f in os.listdir('.') if f.endswith('_testing.py') or f.endswith('_validation.py')],
                'timestamp': datetime.now().isoformat()
            }
        except:
            return {'error': 'Could not collect environment info'}
    
    def run_comprehensive_testing(self):
        """Execute comprehensive testing orchestration"""
        self.logger.info("🎯 Starting KnowledgeHub Comprehensive Integration Testing")
        self.logger.info(f"🆔 Execution ID: {self.execution_id}")
        
        self.start_time = time.time()
        
        # Check if test scripts exist
        missing_scripts = []
        for suite in ORCHESTRATOR_CONFIG['test_suites']:
            script_path = f"/opt/projects/knowledgehub/{suite['script']}"
            if not os.path.exists(script_path):
                missing_scripts.append(suite['script'])
        
        if missing_scripts:
            self.logger.error(f"❌ Missing test scripts: {missing_scripts}")
            sys.exit(1)
        
        # Run test suites
        if ORCHESTRATOR_CONFIG['parallel_execution']:
            self.logger.info("🔀 Running test suites in parallel...")
            self.suite_results = self._run_parallel_tests()
        else:
            self.logger.info("➡️ Running test suites sequentially...")
            self.suite_results = self._run_sequential_tests()
        
        # Generate unified report
        self.logger.info("📊 Generating unified report...")
        unified_report = self.generate_unified_report()
        
        # Save report
        report_file = f"unified_integration_report_{self.execution_id}.json"
        with open(report_file, 'w') as f:
            json.dump(unified_report, f, indent=2)
        
        self.logger.info("✅ Comprehensive testing completed!")
        return unified_report, report_file

def main():
    """Main execution function"""
    try:
        # Create orchestrator
        orchestrator = IntegrationTestOrchestrator()
        
        # Run comprehensive testing
        report, report_file = orchestrator.run_comprehensive_testing()
        
        # Print executive summary
        summary = report['executive_summary']
        print(f"\n{'='*80}")
        print(f"KNOWLEDGEHUB HYBRID RAG SYSTEM - INTEGRATION TEST REPORT")
        print(f"{'='*80}")
        print(f"🎯 Overall Status: {summary['overall_status']}")
        print(f"📊 Health Score: {summary['health_score']:.1f}%")
        print(f"🏥 System Readiness: {report['detailed_analysis']['health_analysis']['system_readiness']['readiness_level']}")
        print(f"🔧 Critical Systems: {summary['critical_systems_status']}")
        print(f"📈 Test Suites: {summary['passed_suites']}/{summary['total_test_suites']} passed")
        
        # Category breakdown
        print(f"\n📂 CATEGORY BREAKDOWN:")
        for category, data in report['detailed_analysis']['category_breakdown'].items():
            success_rate = (data['passed'] / data['total'] * 100) if data['total'] > 0 else 0
            print(f"  {category.title()}: {data['passed']}/{data['total']} ({success_rate:.1f}%)")
        
        # Failed suites
        failed_suites = [r for r in report['test_suite_results'] if r['status'] != 'passed']
        if failed_suites:
            print(f"\n❌ FAILED TEST SUITES:")
            for suite in failed_suites:
                print(f"  - {suite['suite_name']}: {suite['status']}")
        
        # Recommendations
        if report['recommendations']:
            print(f"\n💡 KEY RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'][:5], 1):
                print(f"  {i}. {rec}")
        
        # Next steps
        if report['next_steps']:
            print(f"\n📋 NEXT STEPS:")
            for step in report['next_steps'][:3]:
                print(f"  {step}")
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        print(f"{'='*80}")
        
        # Exit with appropriate code
        if summary['failed_suites'] > 0:
            print(f"\n🚨 {summary['failed_suites']} test suites failed - system requires attention")
            sys.exit(1)
        else:
            print(f"\n🎉 All test suites passed - system transformation successful!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Testing orchestration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()