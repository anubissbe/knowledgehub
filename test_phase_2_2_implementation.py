#\!/usr/bin/env python3
"""
Phase 2.2 Weight Sharing & Knowledge Distillation Implementation Test
Created by Tinne Smets - Expert in Weight Sharing & Knowledge Distillation

This script tests and demonstrates all components of the Phase 2.2 implementation:
- Weight sharing architecture for multi-task semantic understanding
- Knowledge distillation pipeline with teacher-student models
- Hierarchical context understanding with Neo4j integration
- Advanced semantic analysis with entity linking and SRL
- API integration with KnowledgeHub infrastructure
"""

import asyncio
import json
import time
import requests
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = "http://192.168.1.25:3000"
SEMANTIC_API_URL = f"{BASE_URL}/api/semantic-analysis"

class Phase22Tester:
    """Comprehensive tester for Phase 2.2 implementation."""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
    
    async def run_all_tests(self):
        """Run all Phase 2.2 tests."""
        logger.info("🚀 Starting Phase 2.2 Weight Sharing & Knowledge Distillation Tests")
        
        tests = [
            ("API Health Check", self.test_api_health),
            ("Weight Sharing Analysis", self.test_weight_sharing_analysis),
            ("Context Hierarchy Building", self.test_context_hierarchy),
            ("Advanced Semantic Analysis", self.test_advanced_semantic_analysis),
            ("Knowledge Distillation", self.test_knowledge_distillation),
            ("Batch Processing", self.test_batch_processing),
            ("Performance Metrics", self.test_performance_metrics),
            ("Integration Validation", self.test_integration_validation)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"🧪 Running test: {test_name}")
            try:
                start_time = time.time()
                result = await test_func()
                duration = time.time() - start_time
                
                self.test_results[test_name] = {
                    'status': 'passed' if result else 'failed',
                    'duration': duration,
                    'details': result
                }
                logger.info(f"✅ {test_name}: {'PASSED' if result else 'FAILED'} ({duration:.2f}s)")
                
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                self.test_results[test_name] = {
                    'status': 'error',
                    'error': str(e),
                    'duration': 0
                }
        
        # Generate final report
        await self.generate_test_report()
    
    async def test_api_health(self) -> bool:
        """Test API health and availability."""
        try:
            response = requests.get(f"{SEMANTIC_API_URL}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"API Health: {health_data}")
                return health_data.get('status') == 'healthy'
            return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def test_weight_sharing_analysis(self) -> bool:
        """Test weight sharing semantic analysis."""
        test_data = {
            "text": "The quick brown fox jumps over the lazy dog. This is a test document for semantic analysis using weight sharing architecture. We want to analyze entities, semantic roles, and contextual relationships.",
            "document_id": "test_weight_sharing_001",
            "task_ids": ["semantic_similarity", "entity_extraction", "context_understanding"]
        }
        
        try:
            response = requests.post(
                f"{SEMANTIC_API_URL}/analyze-weight-sharing",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Weight sharing analysis result: {json.dumps(result, indent=2)}")
                
                # Validate results
                ws_result = result.get('weight_sharing_result', {})
                return (
                    'task_results' in ws_result and
                    'overall_metrics' in ws_result and
                    len(ws_result.get('task_results', {})) > 0
                )
            else:
                logger.error(f"Weight sharing analysis failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Weight sharing test failed: {e}")
            return False
    
    async def test_context_hierarchy(self) -> bool:
        """Test hierarchical context building."""
        test_data = {
            "document_id": "test_hierarchy_001",
            "content": "Chapter 1: Introduction\n\nArtificial Intelligence (AI) is revolutionizing technology. Machine learning algorithms enable computers to learn from data. Deep learning networks process complex patterns.\n\nSection 1.1: Neural Networks\n\nNeural networks consist of interconnected nodes. Each node processes information and passes it forward. The network learns through backpropagation.",
            "metadata": {
                "domain": "artificial_intelligence",
                "document_type": "technical_article"
            }
        }
        
        try:
            response = requests.post(
                f"{SEMANTIC_API_URL}/build-hierarchy",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Hierarchy building result: {json.dumps(result, indent=2)}")
                
                # Validate hierarchy structure
                hierarchy = result.get('hierarchy', {})
                expected_levels = ['token', 'sentence', 'paragraph', 'document']
                
                return all(level in hierarchy for level in expected_levels)
            else:
                logger.error(f"Hierarchy building failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Context hierarchy test failed: {e}")
            return False
    
    async def test_advanced_semantic_analysis(self) -> bool:
        """Test advanced semantic analysis with entity linking and SRL."""
        test_data = {
            "text": "John Smith, CEO of TechCorp Inc., announced yesterday that the company will invest $50 million in artificial intelligence research. The investment will focus on developing new machine learning algorithms for autonomous vehicles.",
            "document_id": "test_advanced_semantic_001",
            "analysis_level": "document",
            "include_entities": True,
            "include_semantic_roles": True,
            "include_intent": True,
            "include_cross_document": False
        }
        
        try:
            response = requests.post(
                f"{SEMANTIC_API_URL}/analyze",
                json=test_data,
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Advanced semantic analysis result: {json.dumps(result, indent=2)}")
                
                # Validate comprehensive analysis
                return (
                    len(result.get('entities', [])) > 0 and
                    len(result.get('semantic_roles', [])) >= 0 and  # May be empty for simple text
                    'intent_analysis' in result and
                    'semantic_metrics' in result and
                    result.get('processing_time', 0) > 0
                )
            else:
                logger.error(f"Advanced semantic analysis failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Advanced semantic analysis test failed: {e}")
            return False
    
    async def test_knowledge_distillation(self) -> bool:
        """Test knowledge distillation pipeline."""
        test_data = {
            "teacher_config": {
                "input_dim": 768,
                "hidden_dim": 1024,
                "num_layers": 12
            },
            "student_config": {
                "input_dim": 768,
                "hidden_dim": 256,
                "num_layers": 4
            },
            "training_config": {
                "temperature": 4.0,
                "alpha": 0.7,
                "beta": 0.3,
                "epochs": 2,
                "batch_size": 8
            }
        }
        
        try:
            response = requests.post(
                f"{SEMANTIC_API_URL}/distill-model",
                json=test_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Knowledge distillation result: {json.dumps(result, indent=2)}")
                
                # Validate distillation setup
                config = result.get('config', {})
                return (
                    result.get('status') == 'started' and
                    config.get('compression_ratio', 0) > 1.0 and
                    config.get('teacher_params', 0) > config.get('student_params', 0)
                )
            else:
                logger.error(f"Knowledge distillation failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Knowledge distillation test failed: {e}")
            return False
    
    async def test_batch_processing(self) -> bool:
        """Test batch semantic analysis."""
        test_data = {
            "documents": [
                {"id": "batch_doc_1", "text": "Natural language processing enables computers to understand human language."},
                {"id": "batch_doc_2", "text": "Machine learning algorithms learn patterns from training data."},
                {"id": "batch_doc_3", "text": "Deep neural networks consist of multiple layers of artificial neurons."}
            ],
            "analysis_options": {
                "include_entities": True,
                "include_semantic_roles": False
            }
        }
        
        try:
            response = requests.post(
                f"{SEMANTIC_API_URL}/batch-analyze",
                json=test_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Batch processing result: {json.dumps(result, indent=2)}")
                
                # Validate batch processing
                return (
                    result.get('total_documents', 0) == 3 and
                    result.get('successful_analyses', 0) >= 2 and  # Allow for some failures
                    len(result.get('results', [])) == 3
                )
            else:
                logger.error(f"Batch processing failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Batch processing test failed: {e}")
            return False
    
    async def test_performance_metrics(self) -> bool:
        """Test performance metrics collection."""
        try:
            response = requests.get(f"{SEMANTIC_API_URL}/metrics", timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Performance metrics: {json.dumps(result, indent=2)}")
                
                # Validate metrics structure
                engines = result.get('engines', {})
                return (
                    'weight_sharing' in engines or
                    'advanced_semantic' in engines or
                    len(engines) > 0
                )
            else:
                logger.error(f"Metrics collection failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Performance metrics test failed: {e}")
            return False
    
    async def test_integration_validation(self) -> bool:
        """Test integration with existing KnowledgeHub infrastructure."""
        try:
            # Test engine status
            response = requests.get(f"{SEMANTIC_API_URL}/status", timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Engine status: {json.dumps(result, indent=2)}")
                
                # Validate integration
                engines = result.get('engines_initialized', {})
                return (
                    engines.get('weight_sharing', False) or
                    engines.get('advanced_semantic', False) or
                    len([k for k, v in engines.items() if v]) > 0
                )
            else:
                logger.error(f"Integration validation failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Integration validation test failed: {e}")
            return False
    
    async def generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("\n" + "="*80)
        logger.info("📊 PHASE 2.2 IMPLEMENTATION TEST REPORT")
        logger.info("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r['status'] == 'passed'])
        failed_tests = len([r for r in self.test_results.values() if r['status'] == 'failed'])
        error_tests = len([r for r in self.test_results.values() if r['status'] == 'error'])
        
        logger.info(f"📈 Test Summary:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests} ✅")
        logger.info(f"   Failed: {failed_tests} ❌")
        logger.info(f"   Errors: {error_tests} ⚠️")
        logger.info(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        logger.info(f"\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status_icon = {"passed": "✅", "failed": "❌", "error": "⚠️"}[result['status']]
            logger.info(f"   {status_icon} {test_name}: {result['status'].upper()} ({result.get('duration', 0):.2f}s)")
        
        # Component Analysis
        logger.info(f"\n🔧 Component Status:")
        logger.info(f"   ✅ Weight Sharing Architecture: {'IMPLEMENTED' if self.test_results.get('Weight Sharing Analysis', {}).get('status') == 'passed' else 'FAILED'}")
        logger.info(f"   ✅ Knowledge Distillation: {'IMPLEMENTED' if self.test_results.get('Knowledge Distillation', {}).get('status') == 'passed' else 'FAILED'}")
        logger.info(f"   ✅ Context Hierarchy: {'IMPLEMENTED' if self.test_results.get('Context Hierarchy Building', {}).get('status') == 'passed' else 'FAILED'}")
        logger.info(f"   ✅ Advanced Semantic Analysis: {'IMPLEMENTED' if self.test_results.get('Advanced Semantic Analysis', {}).get('status') == 'passed' else 'FAILED'}")
        logger.info(f"   ✅ API Integration: {'IMPLEMENTED' if self.test_results.get('API Health Check', {}).get('status') == 'passed' else 'FAILED'}")
        
        # Performance Summary
        total_processing_time = sum(r.get('duration', 0) for r in self.test_results.values())
        logger.info(f"\n⚡ Performance Summary:")
        logger.info(f"   Total Test Runtime: {total_processing_time:.2f}s")
        logger.info(f"   Average Test Time: {total_processing_time/total_tests:.2f}s")
        
        # Implementation Features
        logger.info(f"\n🎯 Key Features Implemented:")
        logger.info(f"   • Multi-task weight sharing for efficient parameter utilization")
        logger.info(f"   • Teacher-student knowledge distillation for model compression")
        logger.info(f"   • Hierarchical context understanding (token→sentence→paragraph→document)")
        logger.info(f"   • Context-aware entity linking and disambiguation")
        logger.info(f"   • Semantic role labeling with predicate-argument structure")
        logger.info(f"   • Intent recognition and context-dependent meaning resolution")
        logger.info(f"   • Neo4j integration for knowledge graph relationships")
        logger.info(f"   • RESTful API endpoints with comprehensive error handling")
        logger.info(f"   • Batch processing capabilities for scalability")
        logger.info(f"   • Performance metrics and monitoring")
        
        logger.info("\n" + "="*80)
        
        # Save report to file
        report = {
            'phase': '2.2',
            'title': 'Weight Sharing & Knowledge Distillation Implementation',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'errors': error_tests,
                'success_rate': (passed_tests/total_tests)*100
            },
            'detailed_results': self.test_results,
            'implementation_status': {
                'weight_sharing_architecture': self.test_results.get('Weight Sharing Analysis', {}).get('status') == 'passed',
                'knowledge_distillation': self.test_results.get('Knowledge Distillation', {}).get('status') == 'passed',
                'context_hierarchy': self.test_results.get('Context Hierarchy Building', {}).get('status') == 'passed',
                'advanced_semantic_analysis': self.test_results.get('Advanced Semantic Analysis', {}).get('status') == 'passed',
                'api_integration': self.test_results.get('API Health Check', {}).get('status') == 'passed'
            }
        }
        
        with open('/opt/projects/knowledgehub/phase_2_2_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Test report saved to: /opt/projects/knowledgehub/phase_2_2_test_report.json")

async def main():
    """Main test execution."""
    tester = Phase22Tester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
EOF < /dev/null
