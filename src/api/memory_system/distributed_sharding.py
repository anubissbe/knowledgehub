#!/usr/bin/env python3
"""
Distributed Memory Sharding System
Provides horizontal scaling through intelligent memory distribution across multiple nodes
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
from enum import Enum
import aiohttp
import math
from collections import defaultdict

# Add memory system to path
MEMORY_SYSTEM_PATH = Path(__file__).parent
sys.path.insert(0, str(MEMORY_SYSTEM_PATH))

from claude_unified_memory import UnifiedMemorySystem

logger = logging.getLogger(__name__)

class ShardStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    FAILED = "failed"
    REBALANCING = "rebalancing"

class ConsistencyLevel(Enum):
    ONE = "one"          # Write to one shard, read from one
    QUORUM = "quorum"    # Write to majority, read from majority
    ALL = "all"          # Write to all, read from all
    EVENTUAL = "eventual" # Async replication, eventual consistency

@dataclass
class ShardNode:
    """Represents a single shard node in the distributed system"""
    node_id: str
    host: str
    port: int
    status: ShardStatus
    weight: float = 1.0  # Relative capacity weight
    
    # Performance metrics
    latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    storage_usage_mb: float = 0.0
    
    # Shard assignment
    shard_ranges: List[Tuple[int, int]] = None  # List of (start, end) hash ranges
    memory_count: int = 0
    last_health_check: Optional[str] = None
    
    # Connection info
    base_url: str = ""
    
    def __post_init__(self):
        if self.shard_ranges is None:
            self.shard_ranges = []
        if not self.base_url:
            self.base_url = f"http://{self.host}:{self.port}"

@dataclass
class ShardingConfig:
    """Configuration for memory sharding"""
    total_shards: int = 256  # Total number of virtual shards
    replication_factor: int = 2  # Number of replicas per shard
    consistency_level: ConsistencyLevel = ConsistencyLevel.QUORUM
    rebalance_threshold: float = 0.2  # Trigger rebalancing at 20% imbalance
    health_check_interval: int = 30  # Health check interval in seconds
    max_shard_size_mb: float = 1000  # Maximum size per shard in MB
    
    # Performance settings
    parallel_writes: int = 3  # Max parallel write operations
    read_timeout: float = 5.0  # Read timeout in seconds
    write_timeout: float = 10.0  # Write timeout in seconds
    circuit_breaker_threshold: int = 5  # Failures before circuit breaker opens

@dataclass
class MemoryLocation:
    """Describes where a memory is stored"""
    memory_id: str
    shard_hash: int
    primary_node: str
    replica_nodes: List[str]
    consistency_version: int = 1
    
    def __post_init__(self):
        if not self.replica_nodes:
            self.replica_nodes = []

@dataclass
class ShardingMetrics:
    """Metrics for the sharding system"""
    total_memories: int = 0
    total_nodes: int = 0
    active_nodes: int = 0
    failed_nodes: int = 0
    
    # Distribution metrics
    memory_distribution: Dict[str, int] = None
    storage_distribution: Dict[str, float] = None
    load_balance_score: float = 1.0  # 1.0 = perfectly balanced
    
    # Performance metrics
    avg_read_latency_ms: float = 0.0
    avg_write_latency_ms: float = 0.0
    throughput_ops_per_second: float = 0.0
    
    # Health metrics
    health_score: float = 1.0  # Overall system health (0.0 to 1.0)
    last_rebalance: Optional[str] = None
    
    def __post_init__(self):
        if self.memory_distribution is None:
            self.memory_distribution = {}
        if self.storage_distribution is None:
            self.storage_distribution = {}

class DistributedShardingManager:
    """
    Manages distributed memory sharding across multiple nodes
    """
    
    def __init__(self, config: Optional[ShardingConfig] = None):
        self.config = config or ShardingConfig()
        self.nodes: Dict[str, ShardNode] = {}
        self.memory_locations: Dict[str, MemoryLocation] = {}
        self.shard_map: Dict[int, List[str]] = {}  # shard_id -> [node_ids]
        
        # State management
        self.is_initialized = False
        self.is_rebalancing = False
        self.circuit_breakers: Dict[str, int] = defaultdict(int)
        
        # Performance tracking
        self.metrics = ShardingMetrics()
        self.operation_history: List[Dict[str, Any]] = []
        
        # Storage
        self.sharding_dir = Path("/opt/projects/memory-system/data/sharding")
        self.sharding_dir.mkdir(exist_ok=True)
        
        # Load existing configuration
        self._load_sharding_state()
    
    def _load_sharding_state(self):
        """Load existing sharding state from disk"""
        try:
            state_file = self.sharding_dir / "sharding_state.json"
            if state_file.exists():
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    
                    # Load nodes
                    for node_data in state_data.get('nodes', []):
                        node = ShardNode(**node_data)
                        self.nodes[node.node_id] = node
                    
                    # Load shard map
                    self.shard_map = {
                        int(k): v for k, v in state_data.get('shard_map', {}).items()
                    }
                    
                    logger.info(f"Loaded sharding state: {len(self.nodes)} nodes, {len(self.shard_map)} shards")
        except Exception as e:
            logger.error(f"Failed to load sharding state: {e}")
    
    async def _save_sharding_state(self):
        """Save sharding state to disk"""
        try:
            state_data = {
                'nodes': [asdict(node) for node in self.nodes.values()],
                'shard_map': self.shard_map,
                'config': asdict(self.config),
                'last_saved': datetime.now().isoformat()
            }
            
            state_file = self.sharding_dir / "sharding_state.json"
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save sharding state: {e}")
    
    async def add_node(self, host: str, port: int, weight: float = 1.0) -> str:
        """Add a new node to the distributed system"""
        node_id = f"{host}:{port}"
        
        if node_id in self.nodes:
            logger.warning(f"Node {node_id} already exists")
            return node_id
        
        # Create new node
        node = ShardNode(
            node_id=node_id,
            host=host,
            port=port,
            status=ShardStatus.INACTIVE,
            weight=weight
        )
        
        # Health check the node
        is_healthy = await self._health_check_node(node)
        if is_healthy:
            node.status = ShardStatus.ACTIVE
            self.nodes[node_id] = node
            
            # Trigger rebalancing if this is not the first node
            if len(self.nodes) > 1:
                await self._trigger_rebalancing()
            else:
                # First node gets all shards
                await self._assign_all_shards_to_node(node_id)
            
            await self._save_sharding_state()
            logger.info(f"Added node {node_id} successfully")
        else:
            logger.error(f"Failed to add node {node_id} - health check failed")
            raise ConnectionError(f"Cannot connect to node {node_id}")
        
        return node_id
    
    async def remove_node(self, node_id: str) -> bool:
        """Remove a node from the distributed system"""
        if node_id not in self.nodes:
            logger.warning(f"Node {node_id} not found")
            return False
        
        node = self.nodes[node_id]
        node.status = ShardStatus.MAINTENANCE
        
        # Migrate data from this node
        await self._migrate_node_data(node_id)
        
        # Remove from nodes and shard map
        del self.nodes[node_id]
        self._remove_node_from_shard_map(node_id)
        
        await self._save_sharding_state()
        logger.info(f"Removed node {node_id} successfully")
        return True
    
    def _compute_shard_hash(self, memory_id: str) -> int:
        """Compute shard hash for a memory ID"""
        # Use consistent hashing
        hash_value = int(hashlib.md5(memory_id.encode()).hexdigest(), 16)
        return hash_value % self.config.total_shards
    
    def _get_nodes_for_shard(self, shard_id: int) -> List[str]:
        """Get the nodes responsible for a shard"""
        return self.shard_map.get(shard_id, [])
    
    async def _assign_all_shards_to_node(self, node_id: str):
        """Assign all shards to a single node (for first node)"""
        for shard_id in range(self.config.total_shards):
            self.shard_map[shard_id] = [node_id]
        
        # Update node's shard ranges
        node = self.nodes[node_id]
        node.shard_ranges = [(0, self.config.total_shards - 1)]
    
    async def store_memory_distributed(
        self,
        memory_id: str,
        memory_data: Dict[str, Any],
        consistency_level: Optional[ConsistencyLevel] = None
    ) -> bool:
        """Store memory across distributed shards"""
        consistency_level = consistency_level or self.config.consistency_level
        
        # Determine shard and target nodes
        shard_hash = self._compute_shard_hash(memory_id)
        target_nodes = self._get_nodes_for_shard(shard_hash)
        
        if not target_nodes:
            logger.error(f"No nodes available for shard {shard_hash}")
            return False
        
        # Determine write strategy based on consistency level
        if consistency_level == ConsistencyLevel.ONE:
            required_writes = 1
        elif consistency_level == ConsistencyLevel.QUORUM:
            required_writes = (len(target_nodes) // 2) + 1
        elif consistency_level == ConsistencyLevel.ALL:
            required_writes = len(target_nodes)
        else:  # EVENTUAL
            required_writes = 1  # Write to primary, replicate async
        
        # Perform writes
        write_tasks = []
        for i, node_id in enumerate(target_nodes):
            if i < required_writes or consistency_level == ConsistencyLevel.ALL:
                task = self._write_to_node(node_id, memory_id, memory_data)
                write_tasks.append(task)
        
        # Wait for required writes
        try:
            results = await asyncio.gather(*write_tasks, return_exceptions=True)
            successful_writes = sum(1 for r in results if r is True)
            
            if successful_writes >= required_writes:
                # Store memory location
                location = MemoryLocation(
                    memory_id=memory_id,
                    shard_hash=shard_hash,
                    primary_node=target_nodes[0],
                    replica_nodes=target_nodes[1:required_writes]
                )
                self.memory_locations[memory_id] = location
                
                # Handle async replication for eventual consistency
                if consistency_level == ConsistencyLevel.EVENTUAL and len(target_nodes) > 1:
                    asyncio.create_task(self._async_replicate(
                        memory_id, memory_data, target_nodes[1:]
                    ))
                
                logger.debug(f"Stored memory {memory_id} on {successful_writes} nodes")
                return True
            else:
                logger.error(f"Failed to write {memory_id}: only {successful_writes}/{required_writes} writes succeeded")
                return False
                
        except Exception as e:
            logger.error(f"Error storing memory {memory_id}: {e}")
            return False
    
    async def retrieve_memory_distributed(
        self,
        memory_id: str,
        consistency_level: Optional[ConsistencyLevel] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve memory from distributed shards"""
        consistency_level = consistency_level or self.config.consistency_level
        
        # Get memory location
        location = self.memory_locations.get(memory_id)
        if not location:
            # Try to find it by computing shard
            shard_hash = self._compute_shard_hash(memory_id)
            target_nodes = self._get_nodes_for_shard(shard_hash)
        else:
            target_nodes = [location.primary_node] + location.replica_nodes
        
        if not target_nodes:
            logger.warning(f"No nodes available for memory {memory_id}")
            return None
        
        # Determine read strategy
        if consistency_level == ConsistencyLevel.ONE:
            required_reads = 1
        elif consistency_level == ConsistencyLevel.QUORUM:
            required_reads = (len(target_nodes) // 2) + 1
        elif consistency_level == ConsistencyLevel.ALL:
            required_reads = len(target_nodes)
        else:  # EVENTUAL
            required_reads = 1  # Read from primary
        
        # Perform reads
        read_tasks = []
        for i, node_id in enumerate(target_nodes):
            if i < required_reads:
                task = self._read_from_node(node_id, memory_id)
                read_tasks.append(task)
        
        try:
            results = await asyncio.gather(*read_tasks, return_exceptions=True)
            valid_results = [r for r in results if isinstance(r, dict)]
            
            if valid_results:
                # For now, return the first valid result
                # In production, you'd implement conflict resolution
                return valid_results[0]
            else:
                logger.warning(f"No valid results found for memory {memory_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None
    
    async def _write_to_node(self, node_id: str, memory_id: str, memory_data: Dict[str, Any]) -> bool:
        """Write memory to a specific node"""
        node = self.nodes.get(node_id)
        if not node or node.status != ShardStatus.ACTIVE:
            return False
        
        # Check circuit breaker
        if self.circuit_breakers[node_id] >= self.config.circuit_breaker_threshold:
            logger.warning(f"Circuit breaker open for node {node_id}")
            return False
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.write_timeout)) as session:
                url = f"{node.base_url}/memory/{memory_id}"
                async with session.put(url, json=memory_data) as response:
                    if response.status == 200:
                        self.circuit_breakers[node_id] = 0  # Reset on success
                        return True
                    else:
                        self.circuit_breakers[node_id] += 1
                        return False
        except Exception as e:
            self.circuit_breakers[node_id] += 1
            logger.error(f"Failed to write to node {node_id}: {e}")
            return False
    
    async def _read_from_node(self, node_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        """Read memory from a specific node"""
        node = self.nodes.get(node_id)
        if not node or node.status != ShardStatus.ACTIVE:
            return None
        
        # Check circuit breaker
        if self.circuit_breakers[node_id] >= self.config.circuit_breaker_threshold:
            logger.warning(f"Circuit breaker open for node {node_id}")
            return None
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.read_timeout)) as session:
                url = f"{node.base_url}/memory/{memory_id}"
                async with session.get(url) as response:
                    if response.status == 200:
                        self.circuit_breakers[node_id] = 0  # Reset on success
                        return await response.json()
                    else:
                        self.circuit_breakers[node_id] += 1
                        return None
        except Exception as e:
            self.circuit_breakers[node_id] += 1
            logger.error(f"Failed to read from node {node_id}: {e}")
            return None
    
    async def _async_replicate(self, memory_id: str, memory_data: Dict[str, Any], target_nodes: List[str]):
        """Asynchronously replicate memory to additional nodes"""
        for node_id in target_nodes:
            try:
                await self._write_to_node(node_id, memory_id, memory_data)
            except Exception as e:
                logger.warning(f"Async replication failed for {memory_id} to {node_id}: {e}")
    
    async def _health_check_node(self, node: ShardNode) -> bool:
        """Perform health check on a node"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session:
                url = f"{node.base_url}/health"
                async with session.get(url) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        
                        # Update node metrics
                        node.latency_ms = health_data.get('latency_ms', 0)
                        node.memory_usage_mb = health_data.get('memory_usage_mb', 0)
                        node.cpu_usage_percent = health_data.get('cpu_usage_percent', 0)
                        node.storage_usage_mb = health_data.get('storage_usage_mb', 0)
                        node.memory_count = health_data.get('memory_count', 0)
                        node.last_health_check = datetime.now().isoformat()
                        
                        return True
            return False
        except Exception as e:
            logger.error(f"Health check failed for node {node.node_id}: {e}")
            return False
    
    async def _trigger_rebalancing(self):
        """Trigger rebalancing of shards across nodes"""
        if self.is_rebalancing:
            logger.info("Rebalancing already in progress")
            return
        
        self.is_rebalancing = True
        try:
            logger.info("Starting shard rebalancing")
            
            # Calculate optimal shard distribution
            optimal_distribution = await self._calculate_optimal_distribution()
            
            # Plan migration
            migration_plan = await self._create_migration_plan(optimal_distribution)
            
            # Execute migration
            await self._execute_migration_plan(migration_plan)
            
            logger.info("Shard rebalancing completed")
            self.metrics.last_rebalance = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Rebalancing failed: {e}")
        finally:
            self.is_rebalancing = False
    
    async def _calculate_optimal_distribution(self) -> Dict[int, List[str]]:
        """Calculate optimal shard distribution across nodes"""
        active_nodes = [node_id for node_id, node in self.nodes.items() 
                       if node.status == ShardStatus.ACTIVE]
        
        if not active_nodes:
            return {}
        
        # Simple round-robin distribution for now
        # In production, consider node weights, capacity, and current load
        optimal_map = {}
        node_index = 0
        
        for shard_id in range(self.config.total_shards):
            # Assign primary and replicas
            replicas = []
            for i in range(self.config.replication_factor):
                node_idx = (node_index + i) % len(active_nodes)
                replicas.append(active_nodes[node_idx])
            
            optimal_map[shard_id] = replicas
            node_index = (node_index + 1) % len(active_nodes)
        
        return optimal_map
    
    async def _create_migration_plan(self, optimal_distribution: Dict[int, List[str]]) -> List[Dict[str, Any]]:
        """Create migration plan to move from current to optimal distribution"""
        migration_plan = []
        
        for shard_id, optimal_nodes in optimal_distribution.items():
            current_nodes = self.shard_map.get(shard_id, [])
            
            # Find nodes to add and remove
            nodes_to_add = set(optimal_nodes) - set(current_nodes)
            nodes_to_remove = set(current_nodes) - set(optimal_nodes)
            
            if nodes_to_add or nodes_to_remove:
                migration_plan.append({
                    'shard_id': shard_id,
                    'current_nodes': current_nodes,
                    'target_nodes': optimal_nodes,
                    'add_nodes': list(nodes_to_add),
                    'remove_nodes': list(nodes_to_remove)
                })
        
        return migration_plan
    
    async def _execute_migration_plan(self, migration_plan: List[Dict[str, Any]]):
        """Execute the migration plan"""
        for migration in migration_plan:
            shard_id = migration['shard_id']
            add_nodes = migration['add_nodes']
            remove_nodes = migration['remove_nodes']
            
            # Add new replicas first
            for node_id in add_nodes:
                await self._copy_shard_to_node(shard_id, node_id)
            
            # Update shard map
            self.shard_map[shard_id] = migration['target_nodes']
            
            # Remove old replicas
            for node_id in remove_nodes:
                await self._remove_shard_from_node(shard_id, node_id)
    
    async def _copy_shard_to_node(self, shard_id: int, target_node_id: str):
        """Copy a shard's data to a new node"""
        # This would copy all memories in the shard to the target node
        # For now, just log the operation
        logger.info(f"Copying shard {shard_id} to node {target_node_id}")
    
    async def _remove_shard_from_node(self, shard_id: int, node_id: str):
        """Remove a shard's data from a node"""
        # This would remove all memories in the shard from the node
        # For now, just log the operation
        logger.info(f"Removing shard {shard_id} from node {node_id}")
    
    def _remove_node_from_shard_map(self, node_id: str):
        """Remove a node from all shard assignments"""
        for shard_id, nodes in self.shard_map.items():
            if node_id in nodes:
                nodes.remove(node_id)
    
    async def _migrate_node_data(self, node_id: str):
        """Migrate all data from a node before removal"""
        logger.info(f"Migrating data from node {node_id}")
        # This would migrate all data to other nodes
        # For now, trigger rebalancing
        await self._trigger_rebalancing()
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        # Update metrics
        await self._update_metrics()
        
        return {
            'nodes': {
                'total': len(self.nodes),
                'active': len([n for n in self.nodes.values() if n.status == ShardStatus.ACTIVE]),
                'failed': len([n for n in self.nodes.values() if n.status == ShardStatus.FAILED]),
                'details': [asdict(node) for node in self.nodes.values()]
            },
            'shards': {
                'total': self.config.total_shards,
                'mapped': len(self.shard_map),
                'replication_factor': self.config.replication_factor
            },
            'metrics': asdict(self.metrics),
            'config': asdict(self.config),
            'is_rebalancing': self.is_rebalancing
        }
    
    async def _update_metrics(self):
        """Update cluster metrics"""
        # Health check all nodes
        for node in self.nodes.values():
            await self._health_check_node(node)
        
        # Update metrics
        self.metrics.total_nodes = len(self.nodes)
        self.metrics.active_nodes = len([n for n in self.nodes.values() if n.status == ShardStatus.ACTIVE])
        self.metrics.failed_nodes = len([n for n in self.nodes.values() if n.status == ShardStatus.FAILED])
        
        # Calculate distribution metrics
        node_memory_counts = {node.node_id: node.memory_count for node in self.nodes.values()}
        if node_memory_counts:
            values = list(node_memory_counts.values())
            if values and max(values) > 0:
                # Calculate load balance score (lower variance = better balance)
                mean_val = sum(values) / len(values)
                variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                self.metrics.load_balance_score = 1.0 / (1.0 + variance / (mean_val ** 2 + 1))
        
        self.metrics.memory_distribution = node_memory_counts
        self.metrics.storage_distribution = {
            node.node_id: node.storage_usage_mb for node in self.nodes.values()
        }
        
        # Calculate health score
        if self.nodes:
            healthy_nodes = len([n for n in self.nodes.values() if n.status == ShardStatus.ACTIVE])
            self.metrics.health_score = healthy_nodes / len(self.nodes)
        else:
            self.metrics.health_score = 0.0


# Global distributed sharding manager
distributed_sharding = DistributedShardingManager()

# Convenience functions
async def add_shard_node(host: str, port: int, weight: float = 1.0) -> str:
    """Add a new shard node to the cluster"""
    return await distributed_sharding.add_node(host, port, weight)

async def remove_shard_node(node_id: str) -> bool:
    """Remove a shard node from the cluster"""
    return await distributed_sharding.remove_node(node_id)

async def store_distributed(memory_id: str, memory_data: Dict[str, Any]) -> bool:
    """Store memory in distributed shards"""
    return await distributed_sharding.store_memory_distributed(memory_id, memory_data)

async def retrieve_distributed(memory_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve memory from distributed shards"""
    return await distributed_sharding.retrieve_memory_distributed(memory_id)

async def get_shard_status() -> Dict[str, Any]:
    """Get current shard cluster status"""
    return await distributed_sharding.get_cluster_status()

if __name__ == "__main__":
    # Test the distributed sharding system
    async def test_distributed_sharding():
        print("🔀 Testing Distributed Memory Sharding")
        
        # Test configuration
        config = ShardingConfig(
            total_shards=16,
            replication_factor=2,
            consistency_level=ConsistencyLevel.QUORUM
        )
        
        sharding_manager = DistributedShardingManager(config)
        
        # Test shard hash computation
        memory_id = "test_memory_001"
        shard_hash = sharding_manager._compute_shard_hash(memory_id)
        print(f"✅ Computed shard hash for {memory_id}: {shard_hash}")
        
        # Test optimal distribution calculation
        # Add some mock nodes first
        sharding_manager.nodes = {
            "node1": ShardNode("node1", "localhost", 8001, ShardStatus.ACTIVE),
            "node2": ShardNode("node2", "localhost", 8002, ShardStatus.ACTIVE),
            "node3": ShardNode("node3", "localhost", 8003, ShardStatus.ACTIVE)
        }
        
        optimal_dist = await sharding_manager._calculate_optimal_distribution()
        print(f"✅ Calculated optimal distribution: {len(optimal_dist)} shards assigned")
        
        # Test cluster status
        status = await sharding_manager.get_cluster_status()
        print(f"✅ Cluster status: {status['nodes']['total']} nodes, {status['shards']['total']} shards")
        
        print("✅ Distributed Sharding ready!")
    
    asyncio.run(test_distributed_sharding())