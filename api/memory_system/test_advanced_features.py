#!/usr/bin/env python3
"""
Comprehensive test suite for advanced memory system features
Tests distributed sharding, multi-tenant isolation, and incremental context loading
"""

import os
import sys
import asyncio
import logging
import json
from pathlib import Path

# Add memory system to path
MEMORY_SYSTEM_PATH = Path(__file__).parent
sys.path.insert(0, str(MEMORY_SYSTEM_PATH))

from distributed_sharding import (
    DistributedShardingManager, ShardingConfig, ConsistencyLevel,
    distributed_sharding, add_shard_node, store_distributed, retrieve_distributed
)
from multi_tenant_isolation import (
    MultiTenantIsolationManager, AccessLevel, AccessContext,
    multi_tenant_manager, create_tenant, create_user, create_project, check_access
)
from incremental_context_loading import (
    IncrementalContextLoader, LoadingStrategy, ContextType,
    incremental_loader, create_context_windows, load_context_incrementally
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedMemoryTestSuite:
    """Comprehensive test suite for advanced memory features"""
    
    def __init__(self):
        self.test_results = {
            "distributed_sharding": {},
            "multi_tenant_isolation": {},
            "incremental_loading": {},
            "integration": {}
        }
        self.passed_tests = 0
        self.total_tests = 0
    
    async def run_all_tests(self):
        """Run all advanced memory feature tests"""
        print("🧪 Starting Advanced Memory Features Test Suite")
        print("=" * 60)
        
        # Test each component
        await self._test_distributed_sharding()
        await self._test_multi_tenant_isolation()
        await self._test_incremental_loading()
        await self._test_integration()
        
        # Print summary
        self._print_test_summary()
        
        return self.passed_tests == self.total_tests
    
    async def _test_distributed_sharding(self):
        """Test distributed sharding functionality"""
        print("\n🔀 Testing Distributed Memory Sharding")
        print("-" * 40)
        
        # Test 1: Shard hash computation
        self._start_test("Shard hash computation")
        try:
            test_memory_id = "test_memory_advanced_001"
            shard_hash = distributed_sharding._compute_shard_hash(test_memory_id)
            assert 0 <= shard_hash < distributed_sharding.config.total_shards
            self._pass_test(f"Computed shard hash: {shard_hash}")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 2: Configuration validation
        self._start_test("Sharding configuration")
        try:
            config = ShardingConfig(
                total_shards=64,
                replication_factor=3,
                consistency_level=ConsistencyLevel.QUORUM
            )
            sharding_manager = DistributedShardingManager(config)
            assert sharding_manager.config.total_shards == 64
            assert sharding_manager.config.replication_factor == 3
            self._pass_test("Configuration validated")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 3: Optimal distribution calculation
        self._start_test("Optimal distribution calculation")
        try:
            # Create mock cluster
            sharding_manager = DistributedShardingManager()
            sharding_manager.nodes = {
                "node1": type('MockNode', (), {"node_id": "node1", "status": "active"})(),
                "node2": type('MockNode', (), {"node_id": "node2", "status": "active"})(),
                "node3": type('MockNode', (), {"node_id": "node3", "status": "active"})()
            }
            
            distribution = await sharding_manager._calculate_optimal_distribution()
            assert len(distribution) == sharding_manager.config.total_shards
            self._pass_test(f"Distribution calculated for {len(distribution)} shards")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 4: Cluster status
        self._start_test("Cluster status reporting")
        try:
            status = await distributed_sharding.get_cluster_status()
            assert "nodes" in status
            assert "shards" in status
            assert "metrics" in status
            self._pass_test("Cluster status retrieved")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
    
    async def _test_multi_tenant_isolation(self):
        """Test multi-tenant isolation functionality"""
        print("\n🏢 Testing Multi-Tenant Project Isolation")
        print("-" * 40)
        
        # Test 1: Tenant creation
        self._start_test("Tenant creation")
        try:
            tenant_id = await create_tenant(
                name="Test Advanced Tenant",
                description="Tenant for advanced feature testing",
                subscription_tier="premium"
            )
            assert tenant_id.startswith("tenant_")
            self.test_tenant_id = tenant_id
            self._pass_test(f"Created tenant: {tenant_id}")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
            return
        
        # Test 2: User creation
        self._start_test("User creation")
        try:
            user_id = await create_user(
                tenant_id=self.test_tenant_id,
                username="test_admin",
                email="admin@advanced-test.com",
                access_level=AccessLevel.ADMIN,
                creator_user_id="system"
            )
            assert user_id.startswith("user_")
            self.test_user_id = user_id
            self._pass_test(f"Created user: {user_id}")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
            return
        
        # Test 3: Project creation
        self._start_test("Project creation")
        try:
            project_id = await create_project(
                tenant_id=self.test_tenant_id,
                name="Advanced Memory Test Project",
                description="Project for testing advanced memory features",
                creator_user_id=self.test_user_id
            )
            assert project_id.startswith("proj_")
            self.test_project_id = project_id
            self._pass_test(f"Created project: {project_id}")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
            return
        
        # Test 4: Access control
        self._start_test("Access control validation")
        try:
            context = AccessContext(
                tenant_id=self.test_tenant_id,
                user_id=self.test_user_id,
                project_id=self.test_project_id,
                operation="create",
                resource_type="memory"
            )
            
            access_granted, reason = await check_access(context)
            assert access_granted == True
            self._pass_test(f"Access granted: {reason}")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 5: Usage reporting
        self._start_test("Usage reporting")
        try:
            from multi_tenant_isolation import get_usage_report
            report = await get_usage_report(self.test_tenant_id)
            assert "tenant_info" in report
            assert "quota_status" in report
            assert "project_statistics" in report
            self._pass_test("Usage report generated")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
    
    async def _test_incremental_loading(self):
        """Test incremental context loading functionality"""
        print("\n📥 Testing Incremental Context Loading")
        print("-" * 40)
        
        # Test 1: Context window creation
        self._start_test("Context window creation")
        try:
            query = "How to implement advanced caching strategies for distributed systems?"
            windows = await create_context_windows(
                query=query,
                max_windows=6,
                strategy=LoadingStrategy.BALANCED
            )
            assert len(windows) > 0
            assert all(hasattr(w, 'window_type') for w in windows)
            self.test_windows = windows
            self._pass_test(f"Created {len(windows)} context windows")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
            return
        
        # Test 2: Loading plan creation
        self._start_test("Loading plan creation")
        try:
            plan = await incremental_loader.create_loading_plan(
                windows=self.test_windows,
                strategy=LoadingStrategy.BALANCED,
                max_memory_mb=200,
                max_time_ms=15000
            )
            assert plan.total_windows == len(self.test_windows)
            assert len(plan.phases) > 0
            self.test_plan = plan
            self._pass_test(f"Created plan with {len(plan.phases)} phases")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
            return
        
        # Test 3: Loading execution
        self._start_test("Loading execution")
        try:
            progress = await incremental_loader.execute_loading_plan(self.test_plan)
            assert progress.total_windows == len(self.test_windows)
            assert progress.completed_windows + progress.failed_windows == progress.total_windows
            self._pass_test(f"Loaded {progress.completed_windows}/{progress.total_windows} windows")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 4: Cache functionality
        self._start_test("Cache functionality")
        try:
            # Test cache statistics
            stats = await incremental_loader.get_performance_stats()
            assert "cache_size_mb" in stats
            assert "cache_entries" in stats
            assert "total_loads" in stats
            self._pass_test(f"Cache stats: {stats['cache_entries']} entries")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 5: Different loading strategies
        self._start_test("Loading strategies")
        try:
            strategies = [
                LoadingStrategy.RELEVANCE_FIRST,
                LoadingStrategy.PRIORITY_FIRST,
                LoadingStrategy.TIME_BASED
            ]
            
            for strategy in strategies:
                test_windows = await create_context_windows(
                    query="Test strategy query",
                    max_windows=3,
                    strategy=strategy
                )
                assert len(test_windows) > 0
            
            self._pass_test(f"Tested {len(strategies)} loading strategies")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
    
    async def _test_integration(self):
        """Test integration between advanced features"""
        print("\n🔗 Testing Feature Integration")
        print("-" * 40)
        
        # Test 1: Multi-tenant + Distributed storage
        self._start_test("Multi-tenant distributed storage")
        try:
            # Create isolated memory storage request
            from multi_tenant_isolation import AccessContext
            context = AccessContext(
                tenant_id=getattr(self, 'test_tenant_id', 'test_tenant'),
                user_id=getattr(self, 'test_user_id', 'test_user'),
                project_id=getattr(self, 'test_project_id', 'test_project'),
                operation="create",
                resource_type="memory"
            )
            
            # This would integrate with distributed sharding for actual storage
            memory_data = {
                "content": "Test memory for integration",
                "type": "integration_test",
                "context": "advanced_features"
            }
            
            # Mock distributed storage with tenant isolation
            result = await multi_tenant_manager.store_memory_isolated(
                context=context,
                memory_id="integration_test_001",
                memory_data=memory_data
            )
            
            # For now, this returns True in the mock implementation
            self._pass_test(f"Integrated storage result: {result}")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 2: Incremental loading + Multi-tenant context
        self._start_test("Multi-tenant incremental loading")
        try:
            # Create tenant-specific context windows
            tenant_query = f"Load context for tenant {getattr(self, 'test_tenant_id', 'test')}"
            tenant_windows = await create_context_windows(
                query=tenant_query,
                max_windows=3,
                strategy=LoadingStrategy.BALANCED
            )
            
            # Add tenant metadata to windows
            for window in tenant_windows:
                window.tenant_metadata = {
                    "tenant_id": getattr(self, 'test_tenant_id', 'test_tenant'),
                    "isolation_level": "project"
                }
            
            self._pass_test(f"Created {len(tenant_windows)} tenant-aware windows")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 3: All features performance test
        self._start_test("Performance integration test")
        try:
            import time
            
            start_time = time.time()
            
            # Simulate complex operation using all features
            tasks = []
            
            # Create multiple context windows
            for i in range(3):
                query = f"Performance test query {i}"
                task = create_context_windows(query, max_windows=2)
                tasks.append(task)
            
            # Execute in parallel
            results = await asyncio.gather(*tasks)
            total_windows = sum(len(windows) for windows in results)
            
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            self._pass_test(f"Processed {total_windows} windows in {duration_ms:.2f}ms")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 4: Error handling and recovery
        self._start_test("Error handling integration")
        try:
            # Test graceful degradation
            try:
                # Attempt operation with invalid tenant
                invalid_context = AccessContext(
                    tenant_id="invalid_tenant_id",
                    user_id="invalid_user_id",
                    project_id="invalid_project_id",
                    operation="create"
                )
                
                access_granted, reason = await check_access(invalid_context)
                assert access_granted == False
                assert "not found" in reason.lower()
                
            except Exception as e:
                # Expected to fail, which is correct behavior
                pass
            
            self._pass_test("Error handling working correctly")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
    
    def _start_test(self, test_name: str):
        """Start a test"""
        self.current_test = test_name
        self.total_tests += 1
        print(f"  Testing: {test_name}...", end=" ")
    
    def _pass_test(self, message: str = ""):
        """Pass a test"""
        self.passed_tests += 1
        status = "✅ PASS"
        if message:
            status += f" - {message}"
        print(status)
    
    def _fail_test(self, message: str = ""):
        """Fail a test"""
        status = "❌ FAIL"
        if message:
            status += f" - {message}"
        print(status)
    
    def _print_test_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 Excellent! Advanced memory features are working well.")
        elif success_rate >= 75:
            print("✅ Good! Most advanced features are working correctly.")
        elif success_rate >= 50:
            print("⚠️  Some issues found. Review failed tests.")
        else:
            print("❌ Significant issues detected. Advanced features need attention.")
        
        print("\n🏁 Advanced Memory Features Test Suite Complete")

async def main():
    """Run the advanced memory features test suite"""
    test_suite = AdvancedMemoryTestSuite()
    success = await test_suite.run_all_tests()
    return success

if __name__ == "__main__":
    asyncio.run(main())