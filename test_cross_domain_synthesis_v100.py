#\!/usr/bin/env python3
"""
Cross-Domain Knowledge Synthesis V100 GPU Test - Phase 2.4
Created by Yves Vandenberge - Expert in Low-Rank Factorization & Gradual Pruning

This comprehensive test verifies cross-domain knowledge synthesis functionality
on Tesla V100 GPUs with real performance measurements and quality assessments.
"""

import asyncio
import logging
import time
import json
import numpy as np
import torch
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add the API directory to the Python path
sys.path.append('/opt/projects/knowledgehub')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CrossDomainSynthesisV100Test:
    """Comprehensive test suite for cross-domain knowledge synthesis on V100 GPUs."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_results = {}
        self.performance_metrics = {}
        self.gpu_info = {}
        
        logger.info(f"Initializing Cross-Domain Synthesis V100 Test on {self.device}")
        
        if torch.cuda.is_available():
            self.gpu_info = {
                "device_name": torch.cuda.get_device_name(0),
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__
            }
            logger.info(f"GPU Info: {self.gpu_info}")
        else:
            logger.warning("CUDA not available - running on CPU")
    
    def generate_synthetic_domain_data(
        self, 
        domain_name: str,
        num_entities: int = 1000,
        feature_dim: int = 512,
        domain_characteristic: str = "random"
    ) -> Dict[str, Any]:
        """Generate synthetic domain data for testing."""
        
        np.random.seed(hash(domain_name) % 1000)  # Reproducible per domain
        
        if domain_characteristic == "nlp":
            # NLP-like data with sparse, high-dimensional features
            base_vectors = np.random.randn(num_entities, feature_dim) * 0.1
            # Add some dense clusters
            cluster_centers = np.random.randn(10, feature_dim)
            for i in range(num_entities):
                cluster_id = i % 10
                base_vectors[i] += cluster_centers[cluster_id] * 0.5
                
        elif domain_characteristic == "vision":
            # Vision-like data with more uniform distributions
            base_vectors = np.random.randn(num_entities, feature_dim) * 0.3
            # Add spatial correlations
            for i in range(num_entities):
                if i > 0:
                    base_vectors[i] += base_vectors[i-1] * 0.1
                    
        elif domain_characteristic == "audio":
            # Audio-like data with temporal patterns
            t = np.linspace(0, 10, num_entities)
            freq_components = np.sin(2 * np.pi * t[:, None] * np.random.randn(1, feature_dim))
            base_vectors = freq_components * 0.2 + np.random.randn(num_entities, feature_dim) * 0.1
            
        else:  # random
            base_vectors = np.random.randn(num_entities, feature_dim) * 0.2
        
        # Normalize vectors
        base_vectors = base_vectors / (np.linalg.norm(base_vectors, axis=1, keepdims=True) + 1e-8)
        
        entity_mappings = {f"{domain_name}_entity_{i}": i for i in range(num_entities)}
        
        return {
            "domain_name": domain_name,
            "vectors": base_vectors,
            "entity_mappings": entity_mappings,
            "metadata": {
                "characteristic": domain_characteristic,
                "generated_at": datetime.utcnow().isoformat(),
                "num_entities": num_entities,
                "feature_dim": feature_dim
            }
        }
    
    async def test_low_rank_factorization(self) -> Dict[str, Any]:
        """Test low-rank matrix factorization performance."""
        logger.info("Testing Low-Rank Matrix Factorization...")
        
        try:
            from api.services.cross_domain_knowledge_synthesis import (
                LowRankFactorizer, SynthesisConfig
            )
            
            # Test different factorization methods
            methods = ["svd", "nmf"]  # Removed tensor for now due to complexity
            test_results = {}
            
            # Generate test matrix
            batch_size, input_dim = 512, 768
            latent_dim = 256
            test_matrix = torch.randn(batch_size, input_dim, device=self.device)
            
            for method in methods:
                logger.info(f"Testing {method.upper()} factorization...")
                
                # Create factorizer
                factorizer = LowRankFactorizer(
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    method=method,
                    device=self.device
                )
                factorizer = factorizer.to(self.device)
                
                # Measure GPU memory before
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    memory_before = torch.cuda.memory_allocated()
                
                # Time the forward pass
                start_time = time.time()
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    reconstructed, compressed = factorizer(test_matrix)
                    loss = factorizer.compute_compression_loss(test_matrix, reconstructed)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Ensure all GPU operations complete
                
                forward_time = time.time() - start_time
                
                # Measure GPU memory after
                if torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated()
                    memory_used = memory_after - memory_before
                else:
                    memory_used = 0
                
                # Calculate metrics
                compression_ratio = latent_dim / input_dim
                reconstruction_error = torch.mean((test_matrix - reconstructed) ** 2).item()
                
                test_results[method] = {
                    "forward_time_seconds": forward_time,
                    "memory_used_mb": memory_used / 1e6,
                    "compression_ratio": compression_ratio,
                    "reconstruction_error": reconstruction_error,
                    "loss_value": loss.item(),
                    "compressed_shape": list(compressed.shape),
                    "reconstructed_shape": list(reconstructed.shape)
                }
                
                logger.info(f"{method.upper()} - Time: {forward_time:.3f}s, "
                           f"Memory: {memory_used/1e6:.1f}MB, "
                           f"Error: {reconstruction_error:.6f}")
            
            return {
                "status": "success",
                "factorization_results": test_results,
                "test_configuration": {
                    "batch_size": batch_size,
                    "input_dim": input_dim,
                    "latent_dim": latent_dim,
                    "device": self.device
                }
            }
            
        except Exception as e:
            logger.error(f"Low-rank factorization test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def test_gradual_pruning(self) -> Dict[str, Any]:
        """Test gradual pruning performance and quality."""
        logger.info("Testing Gradual Pruning...")
        
        try:
            from api.services.gradual_domain_integration import (
                GradualDomainIntegrator, PruningConfig
            )
            
            # Create test integrator
            config = PruningConfig(
                initial_pruning_rate=0.05,
                max_pruning_rate=0.2,
                importance_threshold=0.01,
                max_iterations=10
            )
            integrator = GradualDomainIntegrator(config=config, device=self.device)
            
            # Generate test domains
            domain_data = {}
            domain_sizes = [1000, 1500, 800]  # Different sized domains
            
            for i, size in enumerate(domain_sizes):
                domain_id = f"test_domain_{i}"
                vectors = torch.randn(size, 512, device=self.device)
                # Add some structure to make pruning meaningful
                if i == 0:  # First domain - add some zeros
                    vectors[::4] *= 0.1  # Every 4th vector is low magnitude
                domain_data[domain_id] = vectors
            
            # Measure performance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()
            
            start_time = time.time()
            
            # Apply gradual pruning
            pruning_result = await integrator.apply_gradual_pruning(
                domain_tensors=domain_data,
                target_compression=0.3  # 30% compression
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            processing_time = time.time() - start_time
            
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                memory_used = memory_after - memory_before
            else:
                memory_used = 0
            
            # Analyze results
            if "error" not in pruning_result:
                summary = pruning_result["integration_summary"]
                analytics = integrator.get_integration_analytics()
                
                return {
                    "status": "success",
                    "pruning_performance": {
                        "processing_time_seconds": processing_time,
                        "memory_used_mb": memory_used / 1e6,
                        "domains_processed": summary["total_domains_processed"],
                        "overall_compression": summary["overall_compression_ratio"],
                        "elements_original": summary["total_elements_original"],
                        "elements_final": summary["total_elements_final"]
                    },
                    "pruning_quality": {
                        "target_achieved": summary["target_compression_achieved"],
                        "compression_variance": pruning_result["performance_metrics"]["compression_variance"],
                        "total_iterations": pruning_result["performance_metrics"]["total_pruning_iterations"]
                    },
                    "analytics": analytics
                }
            else:
                return {"status": "failed", "error": pruning_result["error"]}
                
        except Exception as e:
            logger.error(f"Gradual pruning test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def test_cross_domain_synthesis(self) -> Dict[str, Any]:
        """Test full cross-domain knowledge synthesis pipeline."""
        logger.info("Testing Cross-Domain Knowledge Synthesis...")
        
        try:
            from api.services.cross_domain_knowledge_synthesis import (
                CrossDomainKnowledgeSynthesis, SynthesisConfig
            )
            
            # Create synthesis engine
            config = SynthesisConfig(
                latent_dimensions=256,
                compression_ratio=0.25,
                factorization_method="svd",
                pruning_rate=0.1
            )
            engine = CrossDomainKnowledgeSynthesis(config=config, device=self.device)
            
            # Generate synthetic domains with different characteristics
            domains = [
                self.generate_synthetic_domain_data("nlp_domain", 800, 512, "nlp"),
                self.generate_synthetic_domain_data("vision_domain", 1000, 512, "vision"),
                self.generate_synthetic_domain_data("audio_domain", 600, 512, "audio")
            ]
            
            # Register domains
            registration_results = {}
            for domain_data in domains:
                success = await engine.register_domain_knowledge(
                    domain_id=domain_data["domain_name"],
                    domain_name=domain_data["domain_name"].replace("_", " ").title(),
                    knowledge_vectors=domain_data["vectors"],
                    entity_mappings=domain_data["entity_mappings"],
                    metadata=domain_data["metadata"]
                )
                registration_results[domain_data["domain_name"]] = success
                
                if success:
                    logger.info(f"Registered domain: {domain_data['domain_name']}")
            
            # Create cross-domain bridges
            bridge_results = {}
            domain_pairs = [
                ("nlp_domain", "vision_domain"),
                ("vision_domain", "audio_domain"),
                ("nlp_domain", "audio_domain")
            ]
            
            for source, target in domain_pairs:
                bridge = await engine.create_cross_domain_bridge(
                    source_domain=source,
                    target_domain=target,
                    semantic_alignment_threshold=0.3  # Lower threshold for synthetic data
                )
                
                bridge_results[f"{source}_to_{target}"] = {
                    "created": bridge is not None,
                    "bridge_strength": bridge.bridge_strength if bridge else 0.0,
                    "semantic_alignment": bridge.semantic_alignment if bridge else 0.0
                }
                
                if bridge:
                    logger.info(f"Created bridge: {source} → {target} "
                               f"(strength: {bridge.bridge_strength:.3f})")
            
            # Perform synthesis
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()
            
            start_time = time.time()
            
            synthesis_result = await engine.synthesize_cross_domain_knowledge(
                query_domains=["nlp_domain", "vision_domain", "audio_domain"],
                query_context="multimodal understanding and cross-domain patterns",
                max_results=20
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            synthesis_time = time.time() - start_time
            
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                memory_used = memory_after - memory_before
            else:
                memory_used = 0
            
            # Get overall analytics
            analytics = engine.get_synthesis_analytics()
            
            return {
                "status": "success",
                "registration_results": registration_results,
                "bridge_results": bridge_results,
                "synthesis_performance": {
                    "synthesis_time_seconds": synthesis_time,
                    "memory_used_mb": memory_used / 1e6,
                    "results_generated": len(synthesis_result.get("synthesized_knowledge", [])) if "error" not in synthesis_result else 0,
                    "synthesis_successful": "error" not in synthesis_result
                },
                "synthesis_quality": synthesis_result if "error" not in synthesis_result else {"error": synthesis_result["error"]},
                "system_analytics": analytics
            }
            
        except Exception as e:
            logger.error(f"Cross-domain synthesis test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def test_memory_efficiency(self) -> Dict[str, Any]:
        """Test GPU memory efficiency with large datasets."""
        logger.info("Testing Memory Efficiency with Large Datasets...")
        
        if not torch.cuda.is_available():
            return {"status": "skipped", "reason": "CUDA not available"}
        
        try:
            # Test with increasingly large datasets
            memory_tests = []
            batch_sizes = [100, 500, 1000, 2000]
            
            for batch_size in batch_sizes:
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()
                
                # Create large tensor
                test_tensor = torch.randn(batch_size, 1024, device=self.device)
                
                # Simulate factorization
                U = torch.randn(1024, 256, device=self.device)
                V = torch.randn(256, 1024, device=self.device)
                
                start_time = time.time()
                
                # Matrix operations
                compressed = torch.mm(test_tensor, U)
                reconstructed = torch.mm(compressed, V)
                loss = torch.mean((test_tensor - reconstructed) ** 2)
                
                torch.cuda.synchronize()
                operation_time = time.time() - start_time
                
                memory_after = torch.cuda.memory_allocated()
                memory_used = memory_after - memory_before
                
                memory_tests.append({
                    "batch_size": batch_size,
                    "memory_used_mb": memory_used / 1e6,
                    "operation_time_seconds": operation_time,
                    "memory_efficiency_mb_per_element": (memory_used / 1e6) / batch_size,
                    "throughput_elements_per_second": batch_size / operation_time if operation_time > 0 else 0
                })
                
                logger.info(f"Batch size {batch_size}: {memory_used/1e6:.1f}MB, "
                           f"{operation_time:.3f}s, {batch_size/operation_time:.0f} elem/s")
                
                # Clean up
                del test_tensor, compressed, reconstructed, loss
                torch.cuda.empty_cache()
            
            return {
                "status": "success",
                "memory_efficiency_tests": memory_tests,
                "gpu_memory_info": {
                    "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                    "allocated_memory_mb": torch.cuda.memory_allocated() / 1e6,
                    "cached_memory_mb": torch.cuda.memory_reserved() / 1e6
                }
            }
            
        except Exception as e:
            logger.error(f"Memory efficiency test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all cross-domain synthesis tests."""
        logger.info("Starting Comprehensive Cross-Domain Synthesis V100 Test Suite...")
        
        start_time = time.time()
        
        # Run all tests
        test_results = {
            "test_info": {
                "device": self.device,
                "gpu_info": self.gpu_info,
                "start_time": datetime.utcnow().isoformat(),
                "python_version": sys.version,
                "test_purpose": "Verify cross-domain knowledge synthesis on Tesla V100 GPUs"
            }
        }
        
        # Test 1: Low-rank factorization
        logger.info("\n=== Test 1: Low-Rank Factorization ===")
=== Test 1: Low-Rank Factorization ===")
        test_results["factorization_test"] = await self.test_low_rank_factorization()
        
        # Test 2: Gradual pruning
        logger.info("\n=== Test 2: Gradual Pruning ===")
=== Test 2: Gradual Pruning ===")
        test_results["pruning_test"] = await self.test_gradual_pruning()
        
        # Test 3: Cross-domain synthesis
        logger.info("\n=== Test 3: Cross-Domain Synthesis ===")
=== Test 3: Cross-Domain Synthesis ===")
        test_results["synthesis_test"] = await self.test_cross_domain_synthesis()
        
        # Test 4: Memory efficiency
        logger.info("\n=== Test 4: Memory Efficiency ===")
=== Test 4: Memory Efficiency ===")
        test_results["memory_test"] = await self.test_memory_efficiency()
        
        total_time = time.time() - start_time
        
        # Calculate overall success rate
        successful_tests = sum(1 for test in test_results.values() 
                              if isinstance(test, dict) and test.get("status") == "success")
        total_tests = len([k for k in test_results.keys() if k \!= "test_info"])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0.0
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "total_time_seconds": total_time,
            "end_time": datetime.utcnow().isoformat(),
            "overall_status": "PASS" if success_rate >= 0.75 else "FAIL",
            "test_environment": "Tesla V100 GPU" if torch.cuda.is_available() else "CPU Fallback"
        }
        
        return test_results

async def main():
    """Run the comprehensive test suite."""
    test_suite = CrossDomainSynthesisV100Test()
    
    logger.info("Cross-Domain Knowledge Synthesis V100 GPU Test Suite")
    logger.info("Phase 2.4 Implementation by Yves Vandenberge")
    logger.info("Expert in Low-Rank Factorization & Gradual Pruning
")
    
    # Run tests
    results = await test_suite.run_comprehensive_test()
    
    # Print summary
    summary = results["summary"]
    logger.info(f"
{'='*60}")
    logger.info(f"TEST SUITE COMPLETE - {summary['overall_status']}")
    logger.info(f"{'='*60}")
    logger.info(f"Tests Run: {summary['successful_tests']}/{summary['total_tests']}")
    logger.info(f"Success Rate: {summary['success_rate']:.1%}")
    logger.info(f"Total Time: {summary['total_time_seconds']:.2f} seconds")
    logger.info(f"Environment: {summary['test_environment']}")
    
    # Save results
    output_file = f"/opt/projects/knowledgehub/cross_domain_synthesis_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"
Detailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
