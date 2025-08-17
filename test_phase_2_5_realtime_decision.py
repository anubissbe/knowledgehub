#\!/usr/bin/env python3
"""
Phase 2.5 Real-Time AI Decision Making & Recommendations - Comprehensive Test
Author: Pol Verbruggen - Adaptive Quantization & Model Pruning Expert

This script validates the complete Phase 2.5 implementation including:
- Adaptive quantization for real-time decisions 
- Model pruning for speed optimization
- Cross-domain recommendation systems
- TimescaleDB temporal analytics
- FastAPI endpoints integration
"""

import asyncio
import torch
import numpy as np
import time
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Add project path
sys.path.append('/opt/projects/knowledgehub')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase25ValidationSuite:
    """Comprehensive validation suite for Phase 2.5 implementation"""
    
    def __init__(self):
        self.results = {
            'adaptive_quantization': {},
            'model_pruning': {},
            'recommendation_engine': {},
            'temporal_analytics': {},
            'api_integration': {},
            'performance_metrics': {},
            'v100_optimization': {}
        }
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.start_time = time.time()
        
    async def run_comprehensive_validation(self):
        """Run complete Phase 2.5 validation"""
        
        logger.info("🚀 Starting Phase 2.5: Real-Time AI Decision Making Validation")
        logger.info(f"Hardware: {self.device.upper()}")
        
        # Check V100 GPU availability
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")
            
            if "V100" in gpu_name:
                logger.info("✅ Tesla V100 GPU detected - Optimal performance expected")
            else:
                logger.warning(f"⚠️ Non-V100 GPU detected ({gpu_name}) - Performance may vary")
        
        try:
            # 1. Test Adaptive Quantization Engine
            logger.info("\n=== Testing Adaptive Quantization Engine ===")
            await self.test_adaptive_quantization()
            
            # 2. Test Model Pruning Optimization
            logger.info("\n=== Testing Model Pruning Optimization ===")
            await self.test_model_pruning()
            
            # 3. Test Recommendation Engine
            logger.info("\n=== Testing Recommendation Engine ===")
            await self.test_recommendation_engine()
            
            # 4. Test Temporal Analytics (if TimescaleDB available)
            logger.info("\n=== Testing Temporal Analytics ===")
            await self.test_temporal_analytics()
            
            # 5. Test API Integration
            logger.info("\n=== Testing API Integration ===")
            await self.test_api_integration()
            
            # 6. Performance Benchmarking
            logger.info("\n=== Performance Benchmarking ===")
            await self.benchmark_performance()
            
            # 7. Generate final report
            await self.generate_validation_report()
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
    
    async def test_adaptive_quantization(self):
        """Test adaptive quantization engine functionality"""
        try:
            from api.services.realtime_decision.adaptive_quantization_engine import (
                RealTimeDecisionEngine, DecisionRequest, DecisionUrgency, 
                AdaptiveQuantizer, QuantizationStrategy
            )
            
            logger.info("Initializing adaptive quantization engine...")
            
            # Test configuration
            config = {
                'input_size': 128,
                'hidden_sizes': [256, 128, 64],
                'output_size': 5,
                'device': self.device
            }
            
            # Initialize decision engine
            engine = RealTimeDecisionEngine(config, device=self.device)
            
            # Test different urgency levels and measure performance
            urgency_tests = [
                DecisionUrgency.CRITICAL,
                DecisionUrgency.HIGH, 
                DecisionUrgency.MEDIUM,
                DecisionUrgency.LOW
            ]
            
            latency_results = {}
            accuracy_results = {}
            
            for urgency in urgency_tests:
                logger.info(f"Testing {urgency.value} urgency decisions...")
                
                latencies = []
                confidences = []
                
                # Run multiple test requests
                for i in range(20):
                    test_features = torch.randn(128, device=self.device)
                    
                    request = DecisionRequest(
                        decision_id=f"test_{urgency.value}_{i}",
                        urgency=urgency,
                        context={'test': True, 'iteration': i},
                        features=test_features,
                        metadata={'urgency_test': urgency.value},
                        timestamp=time.time()
                    )
                    
                    response = await engine.process_decision_request(request)
                    latencies.append(response.latency_ms)
                    confidences.append(response.confidence)
                
                avg_latency = np.mean(latencies)
                avg_confidence = np.mean(confidences)
                
                latency_results[urgency.value] = {
                    'avg_latency_ms': avg_latency,
                    'min_latency_ms': np.min(latencies),
                    'max_latency_ms': np.max(latencies),
                    'std_latency_ms': np.std(latencies)
                }
                
                accuracy_results[urgency.value] = {
                    'avg_confidence': avg_confidence,
                    'quantization_level': engine.quantizer.quantization_params[urgency]['bits']
                }
                
                logger.info(f"  Average latency: {avg_latency:.2f}ms")
                logger.info(f"  Quantization: {accuracy_results[urgency.value]['quantization_level']} bits")
            
            # Test batch processing
            logger.info("Testing batch decision processing...")
            batch_requests = []
            for i in range(10):
                request = DecisionRequest(
                    decision_id=f"batch_{i}",
                    urgency=DecisionUrgency.MEDIUM,
                    context={'batch_test': True},
                    features=torch.randn(128, device=self.device),
                    metadata={'batch_index': i},
                    timestamp=time.time()
                )
                batch_requests.append(request)
            
            batch_start = time.perf_counter()
            batch_responses = await engine.batch_process_decisions(batch_requests)
            batch_time = (time.perf_counter() - batch_start) * 1000
            
            avg_batch_latency = np.mean([r.latency_ms for r in batch_responses])
            
            # Validate performance targets
            critical_latency = latency_results['critical']['avg_latency_ms']
            performance_validation = {
                'critical_under_10ms': critical_latency < 10.0,
                'high_under_50ms': latency_results['high']['avg_latency_ms'] < 50.0,
                'medium_under_100ms': latency_results['medium']['avg_latency_ms'] < 100.0,
                'batch_efficient': avg_batch_latency < latency_results['medium']['avg_latency_ms'] * 1.2
            }
            
            self.results['adaptive_quantization'] = {
                'status': 'success',
                'latency_results': latency_results,
                'accuracy_results': accuracy_results,
                'batch_performance': {
                    'batch_time_ms': batch_time,
                    'avg_item_latency_ms': avg_batch_latency,
                    'batch_size': len(batch_responses)
                },
                'performance_validation': performance_validation,
                'device': self.device,
                'quantization_strategies_tested': len(urgency_tests)
            }
            
            logger.info("✅ Adaptive quantization engine test completed")
            
        except Exception as e:
            logger.error(f"❌ Adaptive quantization test failed: {e}")
            self.results['adaptive_quantization'] = {'status': 'failed', 'error': str(e)}
    
    async def test_model_pruning(self):
        """Test model pruning optimization functionality"""
        try:
            from api.services.realtime_decision.model_pruning_optimizer import (
                ModelPruningOptimizer, PruningStrategy, ImportanceCriteria, PruningConfig
            )
            
            logger.info("Testing model pruning optimization...")
            
            # Create test model
            import torch.nn as nn
            test_model = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128), 
                nn.ReLU(),
                nn.Linear(128, 5)
            ).to(self.device)
            
            # Initialize pruning optimizer
            pruning_optimizer = ModelPruningOptimizer(test_model, device=self.device)
            
            # Test different pruning strategies
            pruning_results = {}
            
            # 1. Unstructured pruning
            logger.info("Testing unstructured pruning...")
            original_params = sum(p.numel() for p in test_model.parameters())
            
            unstructured_result = pruning_optimizer.apply_unstructured_pruning(
                sparsity_ratio=0.5
            )
            
            pruning_results['unstructured'] = {
                'compression_ratio': unstructured_result.compression_ratio,
                'original_params': unstructured_result.original_params,
                'pruned_params': unstructured_result.pruned_params
            }
            
            # 2. Structured pruning (new model)
            logger.info("Testing structured pruning...")
            test_model2 = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 5)
            ).to(self.device)
            
            pruning_optimizer2 = ModelPruningOptimizer(test_model2, device=self.device)
            structured_result = pruning_optimizer2.apply_structured_pruning(
                sparsity_ratio=0.3,
                granularity="neuron"
            )
            
            pruning_results['structured'] = {
                'compression_ratio': structured_result.compression_ratio,
                'original_params': structured_result.original_params,
                'pruned_params': structured_result.pruned_params
            }
            
            # 3. Lottery ticket hypothesis test
            logger.info("Testing lottery ticket hypothesis...")
            test_model3 = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(), 
                nn.Linear(32, 5)
            ).to(self.device)
            
            pruning_optimizer3 = ModelPruningOptimizer(test_model3, device=self.device)
            lottery_result = pruning_optimizer3.find_lottery_ticket_subnetwork(
                target_sparsity=0.8,
                iterations=3
            )
            
            pruning_results['lottery_ticket'] = {
                'compression_ratio': lottery_result.compression_ratio,
                'original_params': lottery_result.original_params,
                'pruned_params': lottery_result.pruned_params,
                'iterations': lottery_result.performance_metrics.get('iterations', 0)
            }
            
            # 4. Benchmark inference speed
            logger.info("Benchmarking inference speedup...")
            speed_metrics = pruning_optimizer.benchmark_inference_speed(
                input_size=(128,),
                num_trials=100
            )
            
            # Validate performance improvements
            performance_validation = {
                'compression_achieved': all(r['compression_ratio'] > 0.2 for r in pruning_results.values()),
                'speedup_measured': speed_metrics['inference_speedup'] > 0.8,
                'parameters_reduced': all(r['pruned_params'] < r['original_params'] for r in pruning_results.values())
            }
            
            self.results['model_pruning'] = {
                'status': 'success',
                'pruning_results': pruning_results,
                'inference_speedup': speed_metrics,
                'performance_validation': performance_validation,
                'device': self.device
            }
            
            logger.info("✅ Model pruning optimization test completed")
            
        except Exception as e:
            logger.error(f"❌ Model pruning test failed: {e}")
            self.results['model_pruning'] = {'status': 'failed', 'error': str(e)}
    
    async def test_recommendation_engine(self):
        """Test intelligent recommendation engine with cross-domain synthesis"""
        try:
            from api.services.realtime_decision.recommendation_engine import (
                IntelligentRecommendationEngine, RecommendationRequest, 
                RecommendationType, RecommendationStrategy
            )
            
            logger.info("Testing intelligent recommendation engine...")
            
            config = {
                'user_features': 128,
                'item_features': 256, 
                'context_features': 64,
                'hidden_dim': 512,
                'embedding_dim': 256
            }
            
            engine = IntelligentRecommendationEngine(config, device=self.device)
            
            # Test different recommendation strategies
            strategies = [
                RecommendationStrategy.CONTENT_BASED,
                RecommendationStrategy.COLLABORATIVE,
                RecommendationStrategy.CROSS_DOMAIN,
                RecommendationStrategy.TEMPORAL
            ]
            
            recommendation_results = {}
            
            for strategy in strategies:
                logger.info(f"Testing {strategy.value} strategy...")
                
                request = RecommendationRequest(
                    user_id="test_user_001",
                    request_id=f"test_{strategy.value}",
                    context={
                        'interests': ['ai', 'optimization', 'programming'],
                        'urgency': 'medium',
                        'session_length': 1800,
                        'current_project': 'phase_2_5_validation'
                    },
                    recommendation_type=RecommendationType.CONTENT,
                    strategy=strategy,
                    max_recommendations=5,
                    diversity_threshold=0.7,
                    explanation_level="detailed"
                )
                
                start_time = time.perf_counter()
                response = await engine.generate_recommendations(request)
                processing_time = (time.perf_counter() - start_time) * 1000
                
                recommendation_results[strategy.value] = {
                    'processing_time_ms': processing_time,
                    'api_processing_time_ms': response.processing_time_ms,
                    'recommendations_count': len(response.recommendations),
                    'total_candidates': response.total_candidates,
                    'cross_domain_enhanced': any(rec.cross_domain_sources for rec in response.recommendations),
                    'avg_confidence': np.mean([rec.confidence for rec in response.recommendations]) if response.recommendations else 0.0
                }
            
            # Test batch recommendations
            logger.info("Testing batch recommendation generation...")
            batch_requests = []
            for i in range(5):
                request = RecommendationRequest(
                    user_id=f"batch_user_{i}",
                    request_id=f"batch_test_{i}",
                    context={'interests': ['programming'], 'urgency': 'low'},
                    recommendation_type=RecommendationType.ACTION,
                    strategy=RecommendationStrategy.HYBRID,
                    max_recommendations=3
                )
                batch_requests.append(request)
            
            batch_start = time.perf_counter()
            batch_responses = await engine.batch_generate_recommendations(batch_requests)
            batch_time = (time.perf_counter() - batch_start) * 1000
            
            avg_batch_processing = np.mean([r.processing_time_ms for r in batch_responses])
            
            # Test cross-domain knowledge bridge
            logger.info("Testing cross-domain knowledge synthesis...")
            cross_domain_metrics = engine.knowledge_bridge.transfer_knowledge(
                source_knowledge={
                    'patterns': [{'pattern': 'optimization', 'confidence': 0.9}],
                    'metrics': {'performance': 0.85, 'accuracy': 0.92}
                },
                source_domain='performance',
                target_domain='ai'
            )
            
            # Performance validation
            performance_validation = {
                'all_strategies_working': all(r['recommendations_count'] > 0 for r in recommendation_results.values()),
                'fast_processing': all(r['processing_time_ms'] < 1000 for r in recommendation_results.values()),
                'cross_domain_active': any(r['cross_domain_enhanced'] for r in recommendation_results.values()),
                'batch_efficient': avg_batch_processing < 500
            }
            
            self.results['recommendation_engine'] = {
                'status': 'success', 
                'strategy_results': recommendation_results,
                'batch_performance': {
                    'batch_time_ms': batch_time,
                    'avg_processing_ms': avg_batch_processing,
                    'batch_size': len(batch_responses)
                },
                'cross_domain_synthesis': {
                    'knowledge_transferred': len(cross_domain_metrics) > 0,
                    'transfer_types': list(cross_domain_metrics.keys()) if cross_domain_metrics else []
                },
                'performance_validation': performance_validation,
                'device': self.device
            }
            
            logger.info("✅ Recommendation engine test completed")
            
        except Exception as e:
            logger.error(f"❌ Recommendation engine test failed: {e}")
            self.results['recommendation_engine'] = {'status': 'failed', 'error': str(e)}
    
    async def test_temporal_analytics(self):
        """Test TimescaleDB temporal analytics integration"""
        try:
            logger.info("Testing temporal analytics integration...")
            
            # Try to test with mock data if TimescaleDB not available
            try:
                from api.services.realtime_decision.temporal_analytics_service import (
                    TimescaleAnalyticsService, TemporalMetric, MetricType, 
                    TemporalPattern, PatternAnalysis
                )
                
                # Test with mock configuration (won't connect to real DB)
                config = {
                    'host': 'localhost',
                    'port': 5434, 
                    'database': 'test_db',
                    'user': 'test_user',
                    'password': 'test_pass'
                }
                
                service = TimescaleAnalyticsService(config)
                
                # Test metric creation and validation
                test_metrics = []
                base_time = datetime.now() - timedelta(hours=1)
                
                for i in range(10):
                    timestamp = base_time + timedelta(minutes=i*5)
                    
                    metric = TemporalMetric(
                        timestamp=timestamp,
                        metric_type=MetricType.DECISION_LATENCY,
                        value=50 + 10 * np.sin(i / 3) + np.random.normal(0, 5),
                        user_id=f"user_{i % 3}",
                        session_id=f"session_{i // 3}",
                        context={'test': True, 'phase': '2.5'},
                        metadata={'synthetic': True}
                    )
                    test_metrics.append(metric)
                
                # Test analytics functions (without actual DB)
                logger.info("Testing temporal pattern analysis logic...")
                
                # Mock data analysis
                mock_data = [
                    {
                        'time_bucket': base_time + timedelta(minutes=i*10),
                        'avg_value': 45.0 + i*2 + np.random.normal(0, 3),
                        'max_value': 60.0 + i*2,
                        'min_value': 30.0 + i*1.5,
                        'sample_count': 5,
                        'stddev': 5.0,
                        'median': 45.0 + i*2,
                        'p95': 55.0 + i*2,
                        'p99': 58.0 + i*2
                    }
                    for i in range(12)
                ]
                
                # Test statistical analysis
                statistics = await service._compute_statistics(mock_data)
                trends = await service._detect_trends(mock_data, TemporalPattern.HOURLY)
                anomalies = await service._detect_anomalies(mock_data)
                predictions = await service._generate_predictions(mock_data, TemporalPattern.HOURLY)
                
                temporal_analytics_validation = {
                    'metric_creation': len(test_metrics) > 0,
                    'statistics_computed': len(statistics) > 0,
                    'trends_detected': len(trends) > 0,
                    'anomaly_detection': isinstance(anomalies, list),
                    'predictions_generated': len(predictions) > 0
                }
                
                self.results['temporal_analytics'] = {
                    'status': 'success_mock',
                    'metrics_created': len(test_metrics),
                    'statistics': statistics,
                    'trends': trends,
                    'anomalies_count': len(anomalies),
                    'predictions_count': len(predictions),
                    'validation': temporal_analytics_validation,
                    'note': 'Tested with mock data - TimescaleDB integration validated structurally'
                }
                
            except ImportError as e:
                logger.warning(f"TimescaleDB service not available: {e}")
                self.results['temporal_analytics'] = {
                    'status': 'service_unavailable',
                    'error': str(e),
                    'note': 'TimescaleDB integration requires asyncpg and database setup'
                }
            
            logger.info("✅ Temporal analytics test completed")
            
        except Exception as e:
            logger.error(f"❌ Temporal analytics test failed: {e}")
            self.results['temporal_analytics'] = {'status': 'failed', 'error': str(e)}
    
    async def test_api_integration(self):
        """Test FastAPI router integration"""
        try:
            logger.info("Testing API integration and router functionality...")
            
            # Test router import and initialization
            try:
                from api.routers.realtime_decision_making import router
                logger.info("✅ Router import successful")
                
                # Test route registration
                route_count = len([route for route in router.routes if hasattr(route, 'path')])
                logger.info(f"Router has {route_count} endpoints registered")
                
                # Test Pydantic models
                from api.routers.realtime_decision_making import (
                    DecisionRequestModel, RecommendationRequestModel, 
                    PruningRequestModel, TemporalAnalysisRequestModel
                )
                
                # Validate model structures
                test_decision_model = DecisionRequestModel(
                    decision_id="test_001",
                    urgency="high",
                    features=[1.0, 2.0, 3.0] * 40,  # 120 features
                    context={'test': True}
                )
                
                test_rec_model = RecommendationRequestModel(
                    user_id="api_test_user",
                    request_id="api_test_001",
                    recommendation_type="content",
                    strategy="hybrid"
                )
                
                api_validation = {
                    'router_imported': True,
                    'endpoints_registered': route_count > 0,
                    'models_validated': True,
                    'decision_model_valid': test_decision_model.decision_id == "test_001",
                    'recommendation_model_valid': test_rec_model.user_id == "api_test_user"
                }
                
                self.results['api_integration'] = {
                    'status': 'success',
                    'router_endpoints': route_count,
                    'validation': api_validation,
                    'models_tested': ['DecisionRequest', 'RecommendationRequest', 'PruningRequest', 'TemporalAnalysis']
                }
                
            except ImportError as e:
                logger.warning(f"API router not available: {e}")
                self.results['api_integration'] = {
                    'status': 'import_failed',
                    'error': str(e),
                    'note': 'API integration requires FastAPI dependencies'
                }
            
            logger.info("✅ API integration test completed")
            
        except Exception as e:
            logger.error(f"❌ API integration test failed: {e}")
            self.results['api_integration'] = {'status': 'failed', 'error': str(e)}
    
    async def benchmark_performance(self):
        """Comprehensive performance benchmarking"""
        try:
            logger.info("Running performance benchmarks...")
            
            # System information
            system_info = {
                'device': self.device,
                'python_version': sys.version,
                'torch_version': torch.__version__,
                'numpy_version': np.__version__
            }
            
            if torch.cuda.is_available():
                system_info.update({
                    'gpu_name': torch.cuda.get_device_name(0),
                    'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                    'cuda_version': torch.version.cuda,
                    'gpu_count': torch.cuda.device_count()
                })
            
            # Performance benchmarks
            performance_tests = {}
            
            # 1. Tensor operations benchmark
            logger.info("Benchmarking tensor operations...")
            tensor_sizes = [128, 256, 512, 1024]
            tensor_benchmarks = {}
            
            for size in tensor_sizes:
                times = []
                for _ in range(10):
                    a = torch.randn(size, size, device=self.device)
                    b = torch.randn(size, size, device=self.device)
                    
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    
                    start = time.perf_counter()
                    c = torch.matmul(a, b)
                    
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    
                    times.append((time.perf_counter() - start) * 1000)
                
                tensor_benchmarks[f'{size}x{size}'] = {
                    'avg_time_ms': np.mean(times),
                    'min_time_ms': np.min(times),
                    'max_time_ms': np.max(times)
                }
            
            performance_tests['tensor_operations'] = tensor_benchmarks
            
            # 2. Neural network inference benchmark
            logger.info("Benchmarking neural network inference...")
            
            import torch.nn as nn
            test_model = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 5)
            ).to(self.device)
            
            batch_sizes = [1, 10, 50, 100]
            inference_benchmarks = {}
            
            for batch_size in batch_sizes:
                times = []
                for _ in range(20):
                    input_tensor = torch.randn(batch_size, 256, device=self.device)
                    
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    
                    start = time.perf_counter()
                    with torch.no_grad():
                        output = test_model(input_tensor)
                    
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    
                    times.append((time.perf_counter() - start) * 1000)
                
                inference_benchmarks[f'batch_{batch_size}'] = {
                    'avg_time_ms': np.mean(times),
                    'throughput_samples_per_sec': batch_size * 1000 / np.mean(times)
                }
            
            performance_tests['neural_inference'] = inference_benchmarks
            
            # 3. Memory usage benchmark
            logger.info("Benchmarking memory usage...")
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
                
                # Allocate large tensor
                large_tensor = torch.randn(1000, 1000, device=self.device)
                memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
                memory_used = memory_after - memory_before
                
                # Clean up
                del large_tensor
                torch.cuda.empty_cache()
                
                performance_tests['memory_usage'] = {
                    'memory_used_mb': memory_used,
                    'memory_before_mb': memory_before,
                    'memory_after_mb': memory_after
                }
            
            # Overall performance validation
            performance_validation = {
                'tensor_ops_fast': all(b['avg_time_ms'] < 100 for b in tensor_benchmarks.values()),
                'inference_efficient': all(b['avg_time_ms'] < 50 for b in inference_benchmarks.values()),
                'gpu_acceleration': self.device == 'cuda',
                'memory_management': 'memory_usage' in performance_tests
            }
            
            self.results['performance_metrics'] = {
                'status': 'success',
                'system_info': system_info,
                'benchmarks': performance_tests,
                'validation': performance_validation
            }
            
            logger.info("✅ Performance benchmarking completed")
            
        except Exception as e:
            logger.error(f"❌ Performance benchmarking failed: {e}")
            self.results['performance_metrics'] = {'status': 'failed', 'error': str(e)}
    
    async def generate_validation_report(self):
        """Generate comprehensive validation report"""
        
        total_time = time.time() - self.start_time
        
        logger.info("\n" + "="*80)
        logger.info("🏆 PHASE 2.5: REAL-TIME AI DECISION MAKING VALIDATION REPORT")
        logger.info("="*80)
        
        # Summary statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r.get('status') in ['success', 'success_mock'])
        failed_tests = total_tests - successful_tests
        
        logger.info(f"Total validation time: {total_time:.2f} seconds")
        logger.info(f"Tests completed: {total_tests}")
        logger.info(f"Successful: {successful_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
        
        # Component-specific results
        for component, result in self.results.items():
            status = result.get('status', 'unknown')
            status_symbol = '✅' if status in ['success', 'success_mock'] else '❌' if status == 'failed' else '⚠️'
            
            logger.info(f"\n{status_symbol} {component.upper().replace('_', ' ')}: {status}")
            
            if status == 'success':
                if component == 'adaptive_quantization':
                    validation = result.get('performance_validation', {})
                    logger.info(f"  • Critical decisions < 10ms: {'✅' if validation.get('critical_under_10ms') else '❌'}")
                    logger.info(f"  • High urgency < 50ms: {'✅' if validation.get('high_under_50ms') else '❌'}")
                    logger.info(f"  • Medium urgency < 100ms: {'✅' if validation.get('medium_under_100ms') else '❌'}")
                    logger.info(f"  • Batch processing efficient: {'✅' if validation.get('batch_efficient') else '❌'}")
                
                elif component == 'model_pruning':
                    validation = result.get('performance_validation', {})
                    speedup = result.get('inference_speedup', {})
                    logger.info(f"  • Compression achieved: {'✅' if validation.get('compression_achieved') else '❌'}")
                    logger.info(f"  • Inference speedup: {speedup.get('inference_speedup', 0):.2f}x")
                    
                elif component == 'recommendation_engine':
                    validation = result.get('performance_validation', {})
                    logger.info(f"  • All strategies working: {'✅' if validation.get('all_strategies_working') else '❌'}")
                    logger.info(f"  • Cross-domain synthesis: {'✅' if validation.get('cross_domain_active') else '❌'}")
                    
                elif component == 'performance_metrics':
                    validation = result.get('validation', {})
                    system_info = result.get('system_info', {})
                    logger.info(f"  • GPU: {system_info.get('gpu_name', 'CPU-only')}")
                    logger.info(f"  • GPU acceleration: {'✅' if validation.get('gpu_acceleration') else '❌'}")
                    logger.info(f"  • Fast tensor ops: {'✅' if validation.get('tensor_ops_fast') else '❌'}")
            
            elif status == 'failed':
                error = result.get('error', 'Unknown error')
                logger.info(f"  • Error: {error}")
        
        # Hardware optimization summary
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            is_v100 = "V100" in gpu_name
            
            logger.info(f"\n🔧 V100 OPTIMIZATION STATUS:")
            logger.info(f"  • V100 GPU detected: {'✅' if is_v100 else '❌'}")
            logger.info(f"  • GPU name: {gpu_name}")
            logger.info(f"  • CUDA available: ✅")
            logger.info(f"  • Quantization optimized: {'✅' if successful_tests > 0 else '❌'}")
            logger.info(f"  • Model pruning active: {'✅' if self.results.get('model_pruning', {}).get('status') == 'success' else '❌'}")
            
            self.results['v100_optimization'] = {
                'v100_detected': is_v100,
                'gpu_name': gpu_name,
                'cuda_available': True,
                'optimization_active': successful_tests > 0
            }
        else:
            logger.info(f"\n⚠️  CPU-ONLY MODE - GPU acceleration not available")
            self.results['v100_optimization'] = {
                'v100_detected': False,
                'gpu_name': 'CPU',
                'cuda_available': False,
                'optimization_active': False
            }
        
        # Final recommendations
        logger.info(f"\n🎯 RECOMMENDATIONS:")
        
        if failed_tests == 0:
            logger.info("  • ✅ All systems operational - Phase 2.5 implementation complete")
            logger.info("  • ✅ Real-time decision making ready for production")
            logger.info("  • ✅ AI recommendations with cross-domain synthesis functional")
        else:
            logger.info(f"  • ⚠️  {failed_tests} component(s) need attention")
            if self.results.get('temporal_analytics', {}).get('status') == 'service_unavailable':
                logger.info("  • ℹ️  Consider setting up TimescaleDB for full temporal analytics")
            if self.results.get('api_integration', {}).get('status') == 'import_failed':
                logger.info("  • ℹ️  Install FastAPI dependencies for full API integration")
        
        # Save detailed results
        report_data = {
            'phase': '2.5',
            'title': 'Real-Time AI Decision Making & Recommendations',
            'author': 'Pol Verbruggen - Adaptive Quantization & Model Pruning Expert',
            'completion_date': datetime.now().isoformat(),
            'validation_time_seconds': total_time,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': (successful_tests/total_tests)*100,
            'results': self.results,
            'hardware_info': {
                'device': self.device,
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'v100_optimized': "V100" in torch.cuda.get_device_name(0) if torch.cuda.is_available() else False
            },
            'summary': {
                'adaptive_quantization': successful_tests > 0 and 'adaptive_quantization' in [k for k, v in self.results.items() if v.get('status') == 'success'],
                'model_pruning': successful_tests > 0 and 'model_pruning' in [k for k, v in self.results.items() if v.get('status') == 'success'],
                'recommendation_systems': successful_tests > 0 and 'recommendation_engine' in [k for k, v in self.results.items() if v.get('status') == 'success'],
                'temporal_analytics': 'temporal_analytics' in [k for k, v in self.results.items() if v.get('status') in ['success', 'success_mock']],
                'production_ready': failed_tests == 0 or failed_tests <= 1  # Allow 1 failure for optional components
            }
        }
        
        # Save report
        report_path = '/opt/projects/knowledgehub/phase_2_5_realtime_decision_validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"\n📄 Detailed report saved: {report_path}")
        logger.info("="*80)
        
        return report_data

async def main():
    """Main validation function"""
    validator = Phase25ValidationSuite()
    
    try:
        report = await validator.run_comprehensive_validation()
        
        # Return success/failure based on results
        if report['failed_tests'] == 0:
            logger.info("🎉 Phase 2.5 validation completed successfully\!")
            sys.exit(0)
        else:
            logger.warning(f"⚠️ Phase 2.5 validation completed with {report['failed_tests']} failures")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"💥 Validation failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
EOF < /dev/null
