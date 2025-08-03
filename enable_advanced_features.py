#!/usr/bin/env python3
"""
Enable Advanced Memory Features in KnowledgeHub
This script initializes and enables distributed sharding, multi-tenant isolation,
and incremental context loading.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "api"))

from api.config import settings
from api.memory_system.distributed_sharding import DistributedShardingManager
from api.memory_system.multi_tenant_isolation import MultiTenantManager
from api.memory_system.incremental_context_loading import IncrementalContextLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def enable_distributed_sharding():
    """Enable and configure distributed sharding"""
    if not settings.ENABLE_DISTRIBUTED_SHARDING:
        logger.info("Distributed sharding is disabled in configuration")
        return
    
    logger.info("Enabling distributed sharding...")
    
    try:
        # Initialize sharding manager
        sharding_manager = DistributedShardingManager(
            total_shards=settings.SHARDING_VIRTUAL_SHARDS,
            replication_factor=settings.SHARDING_REPLICATION_FACTOR,
            consistency_level=settings.SHARDING_CONSISTENCY_LEVEL
        )
        
        # Add shard nodes from configuration
        for node in settings.SHARDING_NODES:
            host, port = node.split(':')
            node_id = await sharding_manager.add_node(host, int(port))
            logger.info(f"Added shard node: {node_id} at {host}:{port}")
        
        # Start health monitoring
        await sharding_manager.start_monitoring()
        logger.info("✅ Distributed sharding enabled successfully")
        
        return sharding_manager
        
    except Exception as e:
        logger.error(f"Failed to enable distributed sharding: {e}")
        raise


async def enable_multi_tenant_isolation():
    """Enable multi-tenant isolation"""
    if not settings.ENABLE_MULTI_TENANT_ISOLATION:
        logger.info("Multi-tenant isolation is disabled in configuration")
        return
    
    logger.info("Enabling multi-tenant isolation...")
    
    try:
        # Initialize tenant manager
        tenant_manager = MultiTenantManager(
            isolation_level=settings.TENANT_ISOLATION_LEVEL
        )
        
        # Create default tenant
        default_tenant = await tenant_manager.create_tenant(
            tenant_id=settings.DEFAULT_TENANT_ID,
            tenant_name="Default Tenant",
            quota_config={
                "max_memories": settings.TENANT_QUOTA_MEMORIES,
                "max_storage_mb": settings.TENANT_QUOTA_STORAGE_MB,
                "max_requests_per_minute": 1000
            }
        )
        
        logger.info(f"Created default tenant: {default_tenant.tenant_id}")
        logger.info("✅ Multi-tenant isolation enabled successfully")
        
        return tenant_manager
        
    except Exception as e:
        logger.error(f"Failed to enable multi-tenant isolation: {e}")
        raise


async def enable_incremental_context_loading():
    """Enable incremental context loading"""
    if not settings.ENABLE_INCREMENTAL_CONTEXT_LOADING:
        logger.info("Incremental context loading is disabled in configuration")
        return
    
    logger.info("Enabling incremental context loading...")
    
    try:
        # Initialize context loader
        context_loader = IncrementalContextLoader(
            window_size=settings.CONTEXT_WINDOW_SIZE,
            chunk_size=settings.CONTEXT_CHUNK_SIZE,
            cache_ttl=settings.CONTEXT_CACHE_TTL,
            enable_compression=settings.CONTEXT_COMPRESSION_ENABLED
        )
        
        # Warm up cache with common patterns
        await context_loader.warm_cache()
        
        logger.info("✅ Incremental context loading enabled successfully")
        logger.info(f"  - Window size: {settings.CONTEXT_WINDOW_SIZE:,} tokens")
        logger.info(f"  - Chunk size: {settings.CONTEXT_CHUNK_SIZE:,} tokens")
        logger.info(f"  - Cache TTL: {settings.CONTEXT_CACHE_TTL} seconds")
        logger.info(f"  - Compression: {'Enabled' if settings.CONTEXT_COMPRESSION_ENABLED else 'Disabled'}")
        
        return context_loader
        
    except Exception as e:
        logger.error(f"Failed to enable incremental context loading: {e}")
        raise


async def verify_advanced_features():
    """Verify that all advanced features are working"""
    logger.info("\n📊 Verifying advanced features...")
    
    # Test distributed sharding
    if settings.ENABLE_DISTRIBUTED_SHARDING:
        try:
            from api.memory_system.distributed_sharding import test_sharding
            result = await test_sharding()
            logger.info(f"  ✅ Distributed sharding: {result}")
        except Exception as e:
            logger.error(f"  ❌ Distributed sharding test failed: {e}")
    
    # Test multi-tenant isolation
    if settings.ENABLE_MULTI_TENANT_ISOLATION:
        try:
            from api.memory_system.multi_tenant_isolation import test_isolation
            result = await test_isolation()
            logger.info(f"  ✅ Multi-tenant isolation: {result}")
        except Exception as e:
            logger.error(f"  ❌ Multi-tenant isolation test failed: {e}")
    
    # Test incremental context loading
    if settings.ENABLE_INCREMENTAL_CONTEXT_LOADING:
        try:
            from api.memory_system.incremental_context_loading import test_loading
            result = await test_loading()
            logger.info(f"  ✅ Incremental context loading: {result}")
        except Exception as e:
            logger.error(f"  ❌ Incremental context loading test failed: {e}")


async def main():
    """Main function to enable all advanced features"""
    logger.info("🚀 KnowledgeHub Advanced Features Enablement")
    logger.info("=" * 50)
    
    # Enable features
    sharding_manager = await enable_distributed_sharding()
    tenant_manager = await enable_multi_tenant_isolation()
    context_loader = await enable_incremental_context_loading()
    
    # Verify everything is working
    await verify_advanced_features()
    
    logger.info("\n✨ All advanced features have been enabled!")
    logger.info("\nConfiguration Summary:")
    logger.info(f"  - Distributed Sharding: {'Enabled' if settings.ENABLE_DISTRIBUTED_SHARDING else 'Disabled'}")
    logger.info(f"  - Multi-Tenant Isolation: {'Enabled' if settings.ENABLE_MULTI_TENANT_ISOLATION else 'Disabled'}")
    logger.info(f"  - Incremental Context Loading: {'Enabled' if settings.ENABLE_INCREMENTAL_CONTEXT_LOADING else 'Disabled'}")
    
    # Save state for API to use
    state = {
        "sharding_enabled": settings.ENABLE_DISTRIBUTED_SHARDING,
        "multi_tenant_enabled": settings.ENABLE_MULTI_TENANT_ISOLATION,
        "incremental_loading_enabled": settings.ENABLE_INCREMENTAL_CONTEXT_LOADING,
        "timestamp": str(asyncio.get_event_loop().time())
    }
    
    state_file = Path(__file__).parent / ".advanced_features_state.json"
    import json
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    logger.info(f"\nState saved to: {state_file}")
    logger.info("\n🎉 Advanced features are now active!")


if __name__ == "__main__":
    asyncio.run(main())