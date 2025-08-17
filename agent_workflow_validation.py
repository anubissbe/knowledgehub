#!/usr/bin/env python3
"""
KnowledgeHub Agent Workflow System - Validation Testing
======================================================

Comprehensive testing for the new agent workflow system including:
- LangGraph orchestration validation
- Multi-agent coordination testing
- Workflow execution validation
- State management testing
- Error handling and recovery
- Performance benchmarking
"""

import asyncio
import requests
import json
import time
import logging
import psycopg2
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sys
import uuid

# Configuration
WORKFLOW_CONFIG = {
    'api_base_url': 'http://localhost:3000',
    'postgres_url': 'postgresql://knowledgehub:knowledgehub123@localhost:5433/knowledgehub',
    'timeout_seconds': 30,
    'max_workflow_execution_time': 120,
    'test_workflows': [
        'simple_qa',
        'multi_step_research'
    ],
    'sample_queries': [
        "What is the impact of climate change on biodiversity?",
        "Explain the principles of quantum computing",
        "How do neural networks learn from data?",
        "What are the economic implications of renewable energy adoption?",
        "Describe the latest developments in gene therapy"
    ]
}

@dataclass
class WorkflowTestResult:
    """Workflow test result"""
    test_name: str
    workflow_name: str
    execution_id: str
    status: str
    duration_ms: int
    steps_completed: int
    agents_involved: List[str]
    success: bool
    error: Optional[str] = None
    details: Dict[str, Any] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class AgentWorkflowValidator:
    """Comprehensive validator for agent workflow system"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.session = requests.Session()
        self.session.timeout = WORKFLOW_CONFIG['timeout_seconds']
        self.test_results: List[WorkflowTestResult] = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _record_result(self, result: WorkflowTestResult):
        """Record test result"""
        self.test_results.append(result)
        
        status_emoji = "✅" if result.success else "❌"
        self.logger.info(f"{status_emoji} {result.test_name} - {result.status} ({result.duration_ms}ms)")
        if result.error:
            self.logger.error(f"Error: {result.error}")
    
    def _get_database_connection(self):
        """Get database connection"""
        return psycopg2.connect(WORKFLOW_CONFIG['postgres_url'])
    
    # ===========================================
    # WORKFLOW SYSTEM VALIDATION
    # ===========================================
    
    def test_workflow_definitions_exist(self):
        """Test that workflow definitions are properly loaded"""
        try:
            start_time = time.time()
            
            # Check database for workflow definitions
            conn = self._get_database_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name, workflow_type, graph_definition, is_active, agents_required
                FROM workflow_definitions
                WHERE is_active = true
            """)
            workflows = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Check for expected workflows
            workflow_names = [w[0] for w in workflows]
            expected_workflows = set(WORKFLOW_CONFIG['test_workflows'])
            found_workflows = set(workflow_names)
            missing_workflows = expected_workflows - found_workflows
            
            success = len(missing_workflows) == 0
            
            result = WorkflowTestResult(
                test_name="Workflow Definitions Check",
                workflow_name="all",
                execution_id="n/a",
                status="passed" if success else "failed",
                duration_ms=duration_ms,
                steps_completed=len(workflows),
                agents_involved=[],
                success=success,
                error=f"Missing workflows: {missing_workflows}" if missing_workflows else None,
                details={
                    "total_workflows": len(workflows),
                    "found_workflows": workflow_names,
                    "missing_workflows": list(missing_workflows)
                }
            )
            
            self._record_result(result)
            
        except Exception as e:
            result = WorkflowTestResult(
                test_name="Workflow Definitions Check",
                workflow_name="all",
                execution_id="n/a",
                status="failed",
                duration_ms=0,
                steps_completed=0,
                agents_involved=[],
                success=False,
                error=str(e)
            )
            self._record_result(result)
    
    def test_agent_definitions_exist(self):
        """Test that agent definitions are properly loaded"""
        try:
            start_time = time.time()
            
            # Check database for agent definitions
            conn = self._get_database_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name, role, capabilities, tools_available, is_active
                FROM agent_definitions
                WHERE is_active = true
            """)
            agents = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Expected agents
            expected_agents = {'researcher', 'analyst', 'synthesizer', 'validator'}
            agent_names = set(a[0] for a in agents)
            missing_agents = expected_agents - agent_names
            
            success = len(missing_agents) == 0
            
            result = WorkflowTestResult(
                test_name="Agent Definitions Check",
                workflow_name="all",
                execution_id="n/a",
                status="passed" if success else "failed",
                duration_ms=duration_ms,
                steps_completed=len(agents),
                agents_involved=list(agent_names),
                success=success,
                error=f"Missing agents: {missing_agents}" if missing_agents else None,
                details={
                    "total_agents": len(agents),
                    "found_agents": list(agent_names),
                    "missing_agents": list(missing_agents)
                }
            )
            
            self._record_result(result)
            
        except Exception as e:
            result = WorkflowTestResult(
                test_name="Agent Definitions Check",
                workflow_name="all",
                execution_id="n/a",
                status="failed",
                duration_ms=0,
                steps_completed=0,
                agents_involved=[],
                success=False,
                error=str(e)
            )
            self._record_result(result)
    
    def test_workflow_execution(self, workflow_name: str, query: str):
        """Test individual workflow execution"""
        try:
            start_time = time.time()
            
            # Execute workflow
            url = f"{WORKFLOW_CONFIG['api_base_url']}/api/agent-workflows/execute"
            payload = {
                "workflow_name": workflow_name,
                "input_data": {
                    "query": query,
                    "context": "validation_test",
                    "max_iterations": 5
                }
            }
            
            response = self.session.post(url, json=payload)
            execution_duration = int((time.time() - start_time) * 1000)
            
            if response.status_code in [200, 201]:
                data = response.json()
                execution_id = data.get('workflow_execution_id') or data.get('id')
                
                # Monitor execution if needed
                if data.get('status') == 'pending':
                    execution_id, final_status, monitor_duration = self._monitor_workflow_execution(execution_id)
                    execution_duration += monitor_duration
                else:
                    final_status = data.get('status', 'unknown')
                
                # Get execution details from database
                execution_details = self._get_execution_details(execution_id)
                
                success = final_status in ['completed', 'success']
                
                result = WorkflowTestResult(
                    test_name=f"Workflow Execution: {workflow_name}",
                    workflow_name=workflow_name,
                    execution_id=execution_id,
                    status=final_status,
                    duration_ms=execution_duration,
                    steps_completed=execution_details.get('steps_completed', 0),
                    agents_involved=execution_details.get('agents_involved', []),
                    success=success,
                    error=execution_details.get('error_details') if not success else None,
                    details={
                        "query": query,
                        "input_data": payload['input_data'],
                        "output_data": execution_details.get('output_data'),
                        "execution_time_ms": execution_details.get('execution_time_ms'),
                        "tools_used": execution_details.get('tools_used', [])
                    }
                )
                
            else:
                result = WorkflowTestResult(
                    test_name=f"Workflow Execution: {workflow_name}",
                    workflow_name=workflow_name,
                    execution_id="failed_to_create",
                    status="failed",
                    duration_ms=execution_duration,
                    steps_completed=0,
                    agents_involved=[],
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}",
                    details={"query": query}
                )
            
            self._record_result(result)
            
        except Exception as e:
            result = WorkflowTestResult(
                test_name=f"Workflow Execution: {workflow_name}",
                workflow_name=workflow_name,
                execution_id="exception",
                status="failed",
                duration_ms=0,
                steps_completed=0,
                agents_involved=[],
                success=False,
                error=str(e),
                details={"query": query}
            )
            self._record_result(result)
    
    def _monitor_workflow_execution(self, execution_id: str) -> Tuple[str, str, int]:
        """Monitor workflow execution until completion"""
        start_time = time.time()
        max_wait_time = WORKFLOW_CONFIG['max_workflow_execution_time']
        
        while (time.time() - start_time) < max_wait_time:
            try:
                # Check execution status
                url = f"{WORKFLOW_CONFIG['api_base_url']}/api/agent-workflows/status/{execution_id}"
                response = self.session.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status', 'unknown')
                    
                    if status in ['completed', 'success', 'failed', 'error']:
                        duration_ms = int((time.time() - start_time) * 1000)
                        return execution_id, status, duration_ms
                
                # Wait before next check
                time.sleep(2)
                
            except Exception as e:
                self.logger.warning(f"Error monitoring execution {execution_id}: {e}")
                time.sleep(2)
        
        # Timeout
        duration_ms = int((time.time() - start_time) * 1000)
        return execution_id, "timeout", duration_ms
    
    def _get_execution_details(self, execution_id: str) -> Dict[str, Any]:
        """Get execution details from database"""
        try:
            conn = self._get_database_connection()
            cursor = conn.cursor()
            
            # Get workflow execution details
            cursor.execute("""
                SELECT status, output_data, error_details, execution_time_ms,
                       steps_completed, agents_involved, tools_used
                FROM workflow_executions
                WHERE id = %s
            """, (execution_id,))
            
            result = cursor.fetchone()
            
            if result:
                details = {
                    'status': result[0],
                    'output_data': result[1],
                    'error_details': result[2],
                    'execution_time_ms': result[3],
                    'steps_completed': result[4],
                    'agents_involved': result[5] if result[5] else [],
                    'tools_used': result[6] if result[6] else []
                }
            else:
                details = {}
            
            cursor.close()
            conn.close()
            
            return details
            
        except Exception as e:
            self.logger.warning(f"Error getting execution details: {e}")
            return {}
    
    def test_concurrent_workflow_execution(self):
        """Test concurrent workflow execution"""
        try:
            start_time = time.time()
            
            # Execute multiple workflows concurrently
            import concurrent.futures
            
            def execute_workflow(workflow_query_pair):
                workflow_name, query = workflow_query_pair
                try:
                    url = f"{WORKFLOW_CONFIG['api_base_url']}/api/agent-workflows/execute"
                    payload = {
                        "workflow_name": workflow_name,
                        "input_data": {
                            "query": query,
                            "context": "concurrent_test"
                        }
                    }
                    
                    response = self.session.post(url, json=payload)
                    return response.status_code, response.json() if response.status_code < 400 else response.text
                    
                except Exception as e:
                    return 0, str(e)
            
            # Prepare test cases
            test_cases = [
                ('simple_qa', WORKFLOW_CONFIG['sample_queries'][0]),
                ('simple_qa', WORKFLOW_CONFIG['sample_queries'][1]),
                ('multi_step_research', WORKFLOW_CONFIG['sample_queries'][2])
            ]
            
            # Execute concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(execute_workflow, case) for case in test_cases]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Analyze results
            successful_executions = [r for r in results if r[0] in [200, 201]]
            success = len(successful_executions) == len(test_cases)
            
            result = WorkflowTestResult(
                test_name="Concurrent Workflow Execution",
                workflow_name="multiple",
                execution_id="concurrent_test",
                status="passed" if success else "failed",
                duration_ms=duration_ms,
                steps_completed=len(successful_executions),
                agents_involved=[],
                success=success,
                error=f"Only {len(successful_executions)}/{len(test_cases)} workflows succeeded" if not success else None,
                details={
                    "total_workflows": len(test_cases),
                    "successful_workflows": len(successful_executions),
                    "results": results
                }
            )
            
            self._record_result(result)
            
        except Exception as e:
            result = WorkflowTestResult(
                test_name="Concurrent Workflow Execution",
                workflow_name="multiple",
                execution_id="exception",
                status="failed",
                duration_ms=0,
                steps_completed=0,
                agents_involved=[],
                success=False,
                error=str(e)
            )
            self._record_result(result)
    
    def test_workflow_state_persistence(self):
        """Test workflow state persistence and recovery"""
        try:
            start_time = time.time()
            
            # Create a long-running workflow
            url = f"{WORKFLOW_CONFIG['api_base_url']}/api/agent-workflows/execute"
            payload = {
                "workflow_name": "multi_step_research",
                "input_data": {
                    "query": "Complex analysis requiring multiple steps",
                    "context": "state_persistence_test"
                }
            }
            
            response = self.session.post(url, json=payload)
            
            if response.status_code in [200, 201]:
                data = response.json()
                execution_id = data.get('workflow_execution_id') or data.get('id')
                
                # Wait a bit to let the workflow start
                time.sleep(3)
                
                # Check state persistence in database
                conn = self._get_database_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT current_state, checkpoint_data, status
                    FROM workflow_executions
                    WHERE id = %s
                """, (execution_id,))
                
                state_result = cursor.fetchone()
                cursor.close()
                conn.close()
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                if state_result:
                    current_state, checkpoint_data, status = state_result
                    
                    success = current_state is not None and status in ['pending', 'running', 'completed']
                    
                    result = WorkflowTestResult(
                        test_name="Workflow State Persistence",
                        workflow_name="multi_step_research",
                        execution_id=execution_id,
                        status="passed" if success else "failed",
                        duration_ms=duration_ms,
                        steps_completed=1,
                        agents_involved=[],
                        success=success,
                        details={
                            "has_state": current_state is not None,
                            "has_checkpoint": checkpoint_data is not None,
                            "workflow_status": status
                        }
                    )
                else:
                    result = WorkflowTestResult(
                        test_name="Workflow State Persistence",
                        workflow_name="multi_step_research",
                        execution_id=execution_id,
                        status="failed",
                        duration_ms=duration_ms,
                        steps_completed=0,
                        agents_involved=[],
                        success=False,
                        error="No state found in database"
                    )
            else:
                result = WorkflowTestResult(
                    test_name="Workflow State Persistence",
                    workflow_name="multi_step_research",
                    execution_id="failed_to_create",
                    status="failed",
                    duration_ms=int((time.time() - start_time) * 1000),
                    steps_completed=0,
                    agents_involved=[],
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}"
                )
            
            self._record_result(result)
            
        except Exception as e:
            result = WorkflowTestResult(
                test_name="Workflow State Persistence",
                workflow_name="multi_step_research",
                execution_id="exception",
                status="failed",
                duration_ms=0,
                steps_completed=0,
                agents_involved=[],
                success=False,
                error=str(e)
            )
            self._record_result(result)
    
    def test_workflow_error_handling(self):
        """Test workflow error handling and recovery"""
        try:
            start_time = time.time()
            
            # Execute workflow with invalid input to trigger error handling
            url = f"{WORKFLOW_CONFIG['api_base_url']}/api/agent-workflows/execute"
            payload = {
                "workflow_name": "nonexistent_workflow",  # This should trigger an error
                "input_data": {
                    "query": "Test error handling"
                }
            }
            
            response = self.session.post(url, json=payload)
            duration_ms = int((time.time() - start_time) * 1000)
            
            # We expect this to fail gracefully with appropriate error response
            expected_error = response.status_code >= 400
            success = expected_error
            
            result = WorkflowTestResult(
                test_name="Workflow Error Handling",
                workflow_name="nonexistent_workflow",
                execution_id="error_test",
                status="passed" if success else "failed",
                duration_ms=duration_ms,
                steps_completed=0,
                agents_involved=[],
                success=success,
                error="Expected error not returned" if not expected_error else None,
                details={
                    "expected_error": True,
                    "status_code": response.status_code,
                    "response": response.text[:200] if response.text else None
                }
            )
            
            self._record_result(result)
            
        except Exception as e:
            result = WorkflowTestResult(
                test_name="Workflow Error Handling",
                workflow_name="nonexistent_workflow",
                execution_id="exception",
                status="failed",
                duration_ms=0,
                steps_completed=0,
                agents_involved=[],
                success=False,
                error=str(e)
            )
            self._record_result(result)
    
    # ===========================================
    # EXECUTION AND REPORTING
    # ===========================================
    
    def run_all_tests(self):
        """Execute all workflow validation tests"""
        self.logger.info("🤖 Starting Agent Workflow System Validation")
        
        # 1. System Setup Validation
        self.logger.info("🔍 Validating workflow definitions...")
        self.test_workflow_definitions_exist()
        self.test_agent_definitions_exist()
        
        # 2. Individual Workflow Testing
        self.logger.info("⚡ Testing individual workflow execution...")
        for workflow_name in WORKFLOW_CONFIG['test_workflows']:
            for i, query in enumerate(WORKFLOW_CONFIG['sample_queries'][:2]):  # Test first 2 queries per workflow
                self.test_workflow_execution(workflow_name, query)
                time.sleep(1)  # Brief pause between executions
        
        # 3. Concurrent Execution Testing
        self.logger.info("🔀 Testing concurrent workflow execution...")
        self.test_concurrent_workflow_execution()
        
        # 4. State Management Testing
        self.logger.info("💾 Testing state persistence...")
        self.test_workflow_state_persistence()
        
        # 5. Error Handling Testing
        self.logger.info("🚨 Testing error handling...")
        self.test_workflow_error_handling()
        
        self.logger.info("✅ Workflow validation completed!")
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.success])
        failed_tests = total_tests - passed_tests
        
        # Group by test category
        categories = {
            'setup': [r for r in self.test_results if 'Check' in r.test_name],
            'execution': [r for r in self.test_results if 'Execution' in r.test_name and 'Concurrent' not in r.test_name],
            'concurrent': [r for r in self.test_results if 'Concurrent' in r.test_name],
            'state': [r for r in self.test_results if 'State' in r.test_name],
            'error_handling': [r for r in self.test_results if 'Error' in r.test_name]
        }
        
        # Performance metrics
        execution_times = [r.duration_ms for r in self.test_results if r.success and 'Execution' in r.test_name]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # Workflow success rates
        workflow_results = {}
        for workflow_name in WORKFLOW_CONFIG['test_workflows']:
            workflow_tests = [r for r in self.test_results if r.workflow_name == workflow_name]
            if workflow_tests:
                success_count = len([r for r in workflow_tests if r.success])
                workflow_results[workflow_name] = {
                    'total_tests': len(workflow_tests),
                    'successful_tests': success_count,
                    'success_rate': (success_count / len(workflow_tests)) * 100
                }
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'avg_execution_time_ms': avg_execution_time,
                'timestamp': datetime.now().isoformat()
            },
            'workflow_performance': workflow_results,
            'category_results': {
                category: {
                    'total': len(tests),
                    'passed': len([t for t in tests if t.success]),
                    'failed': len([t for t in tests if not t.success])
                }
                for category, tests in categories.items()
            },
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'workflow_name': r.workflow_name,
                    'status': 'passed' if r.success else 'failed',
                    'duration_ms': r.duration_ms,
                    'steps_completed': r.steps_completed,
                    'agents_involved': r.agents_involved,
                    'error': r.error,
                    'details': r.details
                }
                for r in self.test_results
            ],
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_tests = [r for r in self.test_results if not r.success]
        
        if not failed_tests:
            recommendations.append("All workflow tests passed - system is ready for production")
        else:
            recommendations.append("Some workflow tests failed - review and fix issues before production deployment")
        
        # Check for specific issues
        setup_failures = [r for r in failed_tests if 'Check' in r.test_name]
        if setup_failures:
            recommendations.append("Setup issues detected - verify workflow and agent definitions")
        
        execution_failures = [r for r in failed_tests if 'Execution' in r.test_name]
        if execution_failures:
            recommendations.append("Workflow execution issues detected - review agent implementations")
        
        state_failures = [r for r in failed_tests if 'State' in r.test_name]
        if state_failures:
            recommendations.append("State persistence issues detected - verify database configuration")
        
        # Performance recommendations
        execution_times = [r.duration_ms for r in self.test_results if r.success and 'Execution' in r.test_name]
        if execution_times and max(execution_times) > 30000:  # 30 seconds
            recommendations.append("Some workflows are slow - consider performance optimization")
        
        return recommendations

def main():
    """Main execution function"""
    try:
        # Create validator
        validator = AgentWorkflowValidator()
        
        # Run all tests
        report = validator.run_all_tests()
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"agent_workflow_validation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        summary = report['summary']
        print(f"\n{'='*60}")
        print(f"AGENT WORKFLOW VALIDATION REPORT")
        print(f"{'='*60}")
        print(f"🤖 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️ Avg Execution Time: {summary['avg_execution_time_ms']:.0f}ms")
        
        # Workflow performance
        print(f"\n🔄 WORKFLOW PERFORMANCE:")
        for workflow, stats in report['workflow_performance'].items():
            print(f"  {workflow}: {stats['successful_tests']}/{stats['total_tests']} ({stats['success_rate']:.1f}%)")
        
        # Category breakdown
        print(f"\n📂 CATEGORY BREAKDOWN:")
        for category, stats in report['category_results'].items():
            success_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {category}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Show failed tests
        failed_tests = [r for r in validator.test_results if not r.success]
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"  - {test.test_name}: {test.error}")
        
        # Recommendations
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        print(f"\n📄 Report saved to: {report_file}")
        print(f"{'='*60}")
        
        # Exit with error code if tests failed
        if summary['failed_tests'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⏹️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()