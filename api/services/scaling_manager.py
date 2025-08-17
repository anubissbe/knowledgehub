"""
Horizontal Scaling Manager for KnowledgeHub Enterprise
Implements distributed architecture, load balancing, and auto-scaling
"""

import asyncio
import hashlib
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from datetime import datetime, timedelta
import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import AsyncSessionLocal, async_engine

logger = logging.getLogger(__name__)

class ServiceType(str, Enum):
    """Service type enumeration"""
    API = "api"
    WORKER = "worker"
    AI_SERVICE = "ai_service"
    SEARCH = "search"
    CACHE = "cache"
    DATABASE = "database"

class NodeStatus(str, Enum):
    """Node status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"

@dataclass
class ServiceNode:
    """Represents a service node in the cluster"""
    id: str
    service_type: ServiceType
    host: str
    port: int
    status: NodeStatus = NodeStatus.HEALTHY
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_count: int = 0
    response_time_avg: float = 0.0
    error_rate: float = 0.0
    
    # GPU-specific metrics for AI nodes
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    vram_free: int = 0  # MB
    
    @property
    def health_score(self) -> float:
        """Calculate health score (0.0 to 1.0)"""
        if self.status == NodeStatus.OFFLINE:
            return 0.0
        if self.status == NodeStatus.UNHEALTHY:
            return 0.1
        if self.status == NodeStatus.DEGRADED:
            return 0.5
            
        # Calculate based on metrics
        cpu_score = 1.0 - min(self.cpu_usage / 100.0, 1.0)
        memory_score = 1.0 - min(self.memory_usage / 100.0, 1.0)
        error_score = 1.0 - min(self.error_rate, 1.0)
        
        return (cpu_score + memory_score + error_score) / 3.0
    
    @property
    def endpoint_url(self) -> str:
        """Get endpoint URL for this node"""
        return f"http://{self.host}:{self.port}"

@dataclass
class LoadBalancerPool:
    """Load balancer pool for a service type"""
    service_type: ServiceType
    nodes: List[ServiceNode] = field(default_factory=list)
    algorithm: str = "weighted_round_robin"  # round_robin, least_connections, weighted_round_robin
    _current_index: int = 0
    
    def add_node(self, node: ServiceNode):
        """Add node to pool"""
        if node not in self.nodes:
            self.nodes.append(node)
    
    def remove_node(self, node_id: str):
        """Remove node from pool"""
        self.nodes = [n for n in self.nodes if n.id != node_id]
    
    def get_healthy_nodes(self) -> List[ServiceNode]:
        """Get list of healthy nodes"""
        return [n for n in self.nodes if n.status == NodeStatus.HEALTHY and n.health_score > 0.5]
    
    def select_node(self) -> Optional[ServiceNode]:
        """Select best node using configured algorithm"""
        healthy_nodes = self.get_healthy_nodes()
        
        if not healthy_nodes:
            return None
        
        if self.algorithm == "round_robin":
            node = healthy_nodes[self._current_index % len(healthy_nodes)]
            self._current_index += 1
            return node
        
        elif self.algorithm == "least_connections":
            # Select node with lowest request count
            return min(healthy_nodes, key=lambda n: n.request_count)
        
        elif self.algorithm == "weighted_round_robin":
            # Weight by health score and inverse response time
            weights = []
            for node in healthy_nodes:
                weight = node.health_score * (1.0 / max(node.response_time_avg, 0.001))
                weights.append(weight)
            
            if weights:
                total_weight = sum(weights)
                import random
                r = random.uniform(0, total_weight)
                
                cumulative = 0
                for i, weight in enumerate(weights):
                    cumulative += weight
                    if r <= cumulative:
                        return healthy_nodes[i]
        
        return healthy_nodes[0] if healthy_nodes else None

class ScalingManager:
    """Manages horizontal scaling and load balancing"""
    
    def __init__(self, redis_url: str = "redis://localhost:6381"):
        self.redis_url = redis_url
        self._redis = None
        self.service_pools: Dict[ServiceType, LoadBalancerPool] = {}
        self.scaling_policies: Dict[ServiceType, Dict[str, Any]] = {}
        
        # Initialize service pools
        for service_type in ServiceType:
            self.service_pools[service_type] = LoadBalancerPool(service_type)
        
        # Default scaling policies
        self._init_scaling_policies()
    
    def _init_scaling_policies(self):
        """Initialize default auto-scaling policies"""
        self.scaling_policies = {
            ServiceType.API: {
                "min_nodes": 2,
                "max_nodes": 10,
                "target_cpu": 70.0,
                "target_response_time": 200.0,  # ms
                "scale_up_threshold": 80.0,
                "scale_down_threshold": 30.0,
                "cooldown_period": 300  # seconds
            },
            ServiceType.AI_SERVICE: {
                "min_nodes": 1,
                "max_nodes": 4,  # Limited by GPU availability
                "target_gpu": 80.0,
                "target_response_time": 5000.0,  # ms (AI operations are slower)
                "scale_up_threshold": 90.0,
                "scale_down_threshold": 40.0,
                "cooldown_period": 600,
                "gpu_required": True
            },
            ServiceType.WORKER: {
                "min_nodes": 2,
                "max_nodes": 20,
                "target_cpu": 60.0,
                "scale_up_threshold": 75.0,
                "scale_down_threshold": 25.0,
                "cooldown_period": 180
            }
        }
    
    async def get_redis(self) -> redis.Redis:
        """Get Redis connection"""
        if self._redis is None:
            self._redis = await redis.from_url(self.redis_url, decode_responses=True)
        return self._redis
    
    async def register_node(self, node: ServiceNode) -> bool:
        """Register a service node"""
        try:
            # Add to service pool
            pool = self.service_pools[node.service_type]
            pool.add_node(node)
            
            # Store in Redis for cluster coordination
            redis_client = await self.get_redis()
            node_data = {
                "id": node.id,
                "service_type": node.service_type.value,
                "host": node.host,
                "port": node.port,
                "status": node.status.value,
                "last_heartbeat": node.last_heartbeat.isoformat(),
                "metadata": node.metadata,
                "endpoint_url": node.endpoint_url
            }
            
            await redis_client.hset(
                f"cluster:nodes:{node.service_type.value}",
                node.id,
                json.dumps(node_data)
            )
            
            # Set TTL for automatic cleanup
            await redis_client.expire(f"cluster:nodes:{node.service_type.value}", 3600)
            
            logger.info(f"Registered node {node.id} ({node.service_type.value}) at {node.endpoint_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register node {node.id}: {e}")
            return False
    
    async def update_node_metrics(self, node_id: str, metrics: Dict[str, Any]) -> bool:
        """Update node performance metrics"""
        try:
            # Find node in pools
            node = None
            for pool in self.service_pools.values():
                for n in pool.nodes:
                    if n.id == node_id:
                        node = n
                        break
                if node:
                    break
            
            if not node:
                logger.warning(f"Node {node_id} not found for metrics update")
                return False
            
            # Update metrics
            node.cpu_usage = metrics.get("cpu_usage", node.cpu_usage)
            node.memory_usage = metrics.get("memory_usage", node.memory_usage)
            node.request_count = metrics.get("request_count", node.request_count)
            node.response_time_avg = metrics.get("response_time_avg", node.response_time_avg)
            node.error_rate = metrics.get("error_rate", node.error_rate)
            node.gpu_usage = metrics.get("gpu_usage", node.gpu_usage)
            node.gpu_memory_usage = metrics.get("gpu_memory_usage", node.gpu_memory_usage)
            node.vram_free = metrics.get("vram_free", node.vram_free)
            node.last_heartbeat = datetime.utcnow()
            
            # Update status based on metrics
            if node.cpu_usage > 95 or node.memory_usage > 95 or node.error_rate > 0.1:
                node.status = NodeStatus.DEGRADED
            elif node.health_score > 0.8:
                node.status = NodeStatus.HEALTHY
            
            # Store metrics in Redis for monitoring
            redis_client = await self.get_redis()
            metrics_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "node_id": node_id,
                **metrics
            }
            
            await redis_client.lpush(f"metrics:node:{node_id}", json.dumps(metrics_data))
            await redis_client.ltrim(f"metrics:node:{node_id}", 0, 99)  # Keep last 100 metrics
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update metrics for node {node_id}: {e}")
            return False
    
    async def get_service_endpoint(self, service_type: ServiceType) -> Optional[str]:
        """Get endpoint for a service using load balancing"""
        pool = self.service_pools[service_type]
        node = pool.select_node()
        
        if node:
            # Increment request count for load balancing
            node.request_count += 1
            return node.endpoint_url
        
        logger.warning(f"No healthy nodes available for service {service_type.value}")
        return None
    
    async def check_scaling_needs(self) -> List[Dict[str, Any]]:
        """Check if any services need scaling up or down"""
        scaling_actions = []
        
        for service_type, policy in self.scaling_policies.items():
            pool = self.service_pools[service_type]
            healthy_nodes = pool.get_healthy_nodes()
            
            if not healthy_nodes:
                continue
            
            # Calculate average metrics
            avg_cpu = sum(n.cpu_usage for n in healthy_nodes) / len(healthy_nodes)
            avg_response_time = sum(n.response_time_avg for n in healthy_nodes) / len(healthy_nodes)
            avg_gpu = sum(n.gpu_usage for n in healthy_nodes) / len(healthy_nodes) if any(n.gpu_usage > 0 for n in healthy_nodes) else 0
            
            current_nodes = len(healthy_nodes)
            min_nodes = policy["min_nodes"]
            max_nodes = policy["max_nodes"]
            
            # Check if scaling is needed
            should_scale_up = False
            should_scale_down = False
            
            # Scale up conditions
            if current_nodes < max_nodes:
                if avg_cpu > policy["scale_up_threshold"]:
                    should_scale_up = True
                elif avg_response_time > policy.get("target_response_time", 1000):
                    should_scale_up = True
                elif service_type == ServiceType.AI_SERVICE and avg_gpu > policy["scale_up_threshold"]:
                    should_scale_up = True
            
            # Scale down conditions
            if current_nodes > min_nodes:
                if avg_cpu < policy["scale_down_threshold"] and avg_response_time < policy.get("target_response_time", 1000) * 0.5:
                    should_scale_down = True
            
            if should_scale_up:
                scaling_actions.append({
                    "action": "scale_up",
                    "service_type": service_type.value,
                    "current_nodes": current_nodes,
                    "target_nodes": min(current_nodes + 1, max_nodes),
                    "reason": f"CPU: {avg_cpu:.1f}%, Response time: {avg_response_time:.1f}ms"
                })
            
            elif should_scale_down:
                scaling_actions.append({
                    "action": "scale_down",
                    "service_type": service_type.value,
                    "current_nodes": current_nodes,
                    "target_nodes": max(current_nodes - 1, min_nodes),
                    "reason": f"CPU: {avg_cpu:.1f}%, Response time: {avg_response_time:.1f}ms"
                })
        
        return scaling_actions
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status"""
        cluster_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "services": {},
            "total_nodes": 0,
            "healthy_nodes": 0,
            "gpu_nodes": 0,
            "total_vram_free": 0
        }
        
        for service_type, pool in self.service_pools.items():
            healthy_nodes = pool.get_healthy_nodes()
            all_nodes = pool.nodes
            
            gpu_nodes = [n for n in healthy_nodes if n.gpu_usage > 0 or n.vram_free > 0]
            
            cluster_status["services"][service_type.value] = {
                "total_nodes": len(all_nodes),
                "healthy_nodes": len(healthy_nodes),
                "gpu_nodes": len(gpu_nodes),
                "avg_cpu": sum(n.cpu_usage for n in healthy_nodes) / len(healthy_nodes) if healthy_nodes else 0,
                "avg_memory": sum(n.memory_usage for n in healthy_nodes) / len(healthy_nodes) if healthy_nodes else 0,
                "avg_response_time": sum(n.response_time_avg for n in healthy_nodes) / len(healthy_nodes) if healthy_nodes else 0,
                "total_requests": sum(n.request_count for n in healthy_nodes),
                "avg_error_rate": sum(n.error_rate for n in healthy_nodes) / len(healthy_nodes) if healthy_nodes else 0,
                "total_vram_free": sum(n.vram_free for n in gpu_nodes)
            }
            
            cluster_status["total_nodes"] += len(all_nodes)
            cluster_status["healthy_nodes"] += len(healthy_nodes)
            cluster_status["gpu_nodes"] += len(gpu_nodes)
            cluster_status["total_vram_free"] += sum(n.vram_free for n in gpu_nodes)
        
        return cluster_status
    
    async def cleanup_stale_nodes(self, max_age_minutes: int = 10):
        """Remove nodes that haven't sent heartbeat in specified time"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        
        for service_type, pool in self.service_pools.items():
            stale_nodes = [
                n for n in pool.nodes 
                if n.last_heartbeat < cutoff_time
            ]
            
            for node in stale_nodes:
                logger.warning(f"Removing stale node {node.id} (last heartbeat: {node.last_heartbeat})")
                pool.remove_node(node.id)
                
                # Remove from Redis
                try:
                    redis_client = await self.get_redis()
                    await redis_client.hdel(f"cluster:nodes:{service_type.value}", node.id)
                except Exception as e:
                    logger.error(f"Failed to remove stale node from Redis: {e}")

# GPU-specific scaling functions
async def get_optimal_gpu_allocation(
    required_vram: int,
    scaling_manager: ScalingManager
) -> Optional[ServiceNode]:
    """Find optimal GPU node for AI workload"""
    
    ai_pool = scaling_manager.service_pools[ServiceType.AI_SERVICE]
    gpu_nodes = [n for n in ai_pool.get_healthy_nodes() if n.vram_free >= required_vram]
    
    if not gpu_nodes:
        return None
    
    # Select node with most free VRAM and lowest GPU utilization
    optimal_node = min(
        gpu_nodes,
        key=lambda n: (n.gpu_usage, -n.vram_free)
    )
    
    return optimal_node

async def distribute_inference_load(
    models: List[str],
    scaling_manager: ScalingManager
) -> Dict[str, str]:
    """Distribute AI model loading across available GPU nodes"""
    
    ai_pool = scaling_manager.service_pools[ServiceType.AI_SERVICE]
    gpu_nodes = ai_pool.get_healthy_nodes()
    
    if not gpu_nodes:
        return {}
    
    # Simple round-robin distribution
    model_assignments = {}
    for i, model in enumerate(models):
        node = gpu_nodes[i % len(gpu_nodes)]
        model_assignments[model] = node.endpoint_url
    
    return model_assignments

# Global scaling manager instance
scaling_manager = ScalingManager()
