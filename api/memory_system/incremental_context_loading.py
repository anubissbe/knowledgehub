#!/usr/bin/env python3
"""
Incremental Context Loading System
Provides intelligent, incremental loading of memory contexts based on relevance and resource constraints
"""

import os
import sys
import json
import asyncio
import logging
import time
import heapq
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
from enum import Enum
import math
from collections import defaultdict, deque
import hashlib

# Add memory system to path
MEMORY_SYSTEM_PATH = Path(__file__).parent
sys.path.insert(0, str(MEMORY_SYSTEM_PATH))

from claude_unified_memory import UnifiedMemorySystem
from ..path_config import CONTEXT_CACHE_BASE, INCREMENTAL_LOADING_BASE

logger = logging.getLogger(__name__)

class LoadingStrategy(Enum):
    RELEVANCE_FIRST = "relevance_first"
    PRIORITY_FIRST = "priority_first"
    BALANCED = "balanced"
    TIME_BASED = "time_based"
    ADAPTIVE = "adaptive"

class ContextType(Enum):
    CONVERSATION = "conversation"
    PROJECT = "project"
    TECHNICAL = "technical"
    DECISION = "decision"
    PATTERN = "pattern"
    REFERENCE = "reference"
    INSIGHT = "insight"

class LoadingStatus(Enum):
    PENDING = "pending"
    LOADING = "loading"
    LOADED = "loaded"
    CACHED = "cached"
    FAILED = "failed"
    EXPIRED = "expired"

@dataclass
class ContextWindow:
    """Represents a context window for incremental loading"""
    window_id: str
    window_type: ContextType
    priority: int  # 1-10 scale
    relevance_score: float  # 0.0-1.0
    estimated_size_mb: float
    estimated_load_time_ms: float
    
    # Content identifiers
    memory_ids: List[str]
    tag_filters: List[str]
    time_range: Optional[Tuple[str, str]] = None
    
    # Loading metadata
    status: LoadingStatus = LoadingStatus.PENDING
    load_start_time: Optional[str] = None
    load_end_time: Optional[str] = None
    actual_size_mb: Optional[float] = None
    cache_key: str = ""
    
    # Dependencies
    depends_on: List[str] = None  # Other window IDs this depends on
    enables: List[str] = None     # Windows this enables
    
    def __post_init__(self):
        if not self.cache_key:
            self.cache_key = self._generate_cache_key()
        if self.depends_on is None:
            self.depends_on = []
        if self.enables is None:
            self.enables = []
    
    def _generate_cache_key(self) -> str:
        """Generate cache key for this context window"""
        key_components = [
            self.window_type.value,
            str(sorted(self.memory_ids)),
            str(sorted(self.tag_filters)),
            str(self.time_range)
        ]
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()[:12]

@dataclass
class LoadingPlan:
    """Plan for incremental context loading"""
    plan_id: str
    strategy: LoadingStrategy
    total_windows: int
    estimated_total_size_mb: float
    estimated_total_time_ms: float
    
    # Loading phases
    phases: List[List[str]]  # Lists of window IDs per phase
    phase_priorities: List[int]
    
    # Resource constraints
    max_memory_mb: float
    max_loading_time_ms: float
    max_concurrent_loads: int
    
    # Optimization settings
    enable_caching: bool = True
    enable_compression: bool = False
    enable_parallel_loading: bool = True
    
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

@dataclass
class LoadingProgress:
    """Progress tracking for incremental loading"""
    progress_id: str
    plan_id: str
    
    # Overall progress
    total_windows: int
    completed_windows: int
    failed_windows: int
    
    # Resource usage
    loaded_size_mb: float
    cached_size_mb: float
    peak_memory_mb: float
    
    # Timing
    start_time: str
    last_update: str
    estimated_completion: Optional[str] = None
    
    # Current state
    current_phase: int = 0
    loading_windows: List[str] = None
    
    # Performance metrics
    average_load_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    throughput_mb_per_sec: float = 0.0
    
    def __post_init__(self):
        if self.loading_windows is None:
            self.loading_windows = []

@dataclass
class ContextCache:
    """Cache entry for loaded context"""
    cache_id: str
    window_id: str
    content: Dict[str, Any]
    size_mb: float
    
    # Cache metadata
    created_at: str
    last_accessed: str
    access_count: int = 0
    hit_count: int = 0
    
    # Expiration
    ttl_seconds: int = 3600  # 1 hour default
    expires_at: str = ""
    
    # Optimization
    compressed: bool = False
    compression_ratio: float = 1.0
    
    def __post_init__(self):
        if not self.expires_at:
            expiry_time = datetime.now() + timedelta(seconds=self.ttl_seconds)
            self.expires_at = expiry_time.isoformat()

class IncrementalContextLoader:
    """
    Intelligent incremental context loading system
    """
    
    def __init__(self, memory_system: Optional[UnifiedMemorySystem] = None):
        self.memory_system = memory_system or UnifiedMemorySystem()
        
        # Storage directories
        self.loading_dir = INCREMENTAL_LOADING_BASE
        self.cache_dir = CONTEXT_CACHE_BASE
        self.plans_dir = self.loading_dir / "plans"
        self.progress_dir = self.loading_dir / "progress"
        
        for directory in [self.loading_dir, self.cache_dir, self.plans_dir, self.progress_dir]:
            directory.mkdir(exist_ok=True)
        
        # Loading state
        self.context_windows: Dict[str, ContextWindow] = {}
        self.loading_plans: Dict[str, LoadingPlan] = {}
        self.active_progress: Dict[str, LoadingProgress] = {}
        self.context_cache: Dict[str, ContextCache] = {}
        
        # Configuration
        self.config = {
            "default_max_memory_mb": 500,
            "default_max_load_time_ms": 30000,  # 30 seconds
            "default_max_concurrent": 3,
            "cache_ttl_seconds": 3600,
            "cache_max_size_mb": 1000,
            "adaptive_threshold": 0.8,
            "relevance_decay_factor": 0.9
        }
        
        # Performance tracking
        self.performance_stats = {
            "total_loads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_load_time": 0.0,
            "peak_memory_usage": 0.0
        }
        
        # Load existing data
        self._load_existing_cache()
        
    def _load_existing_cache(self):
        """Load existing cache entries"""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    cache_entry = ContextCache(**cache_data)
                    
                    # Check if cache entry is still valid
                    if datetime.fromisoformat(cache_entry.expires_at) > datetime.now():
                        self.context_cache[cache_entry.cache_id] = cache_entry
                    else:
                        # Remove expired cache file
                        cache_file.unlink()
            
            logger.info(f"Loaded {len(self.context_cache)} cache entries")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    
    async def create_context_windows(
        self,
        query_context: str,
        max_windows: int = 10,
        strategy: LoadingStrategy = LoadingStrategy.BALANCED
    ) -> List[ContextWindow]:
        """Create context windows based on query context"""
        
        # Analyze query context to determine relevant memory types
        context_analysis = await self._analyze_query_context(query_context)
        
        windows = []
        
        # Create conversation context window
        if context_analysis.get("needs_conversation_context", True):
            conv_window = await self._create_conversation_window(query_context)
            if conv_window:
                windows.append(conv_window)
        
        # Create project context window
        if context_analysis.get("needs_project_context", True):
            proj_window = await self._create_project_window(query_context)
            if proj_window:
                windows.append(proj_window)
        
        # Create technical context windows
        technical_needs = context_analysis.get("technical_domains", [])
        for domain in technical_needs[:3]:  # Limit technical windows
            tech_window = await self._create_technical_window(query_context, domain)
            if tech_window:
                windows.append(tech_window)
        
        # Create decision context window
        if context_analysis.get("needs_decision_context", False):
            decision_window = await self._create_decision_window(query_context)
            if decision_window:
                windows.append(decision_window)
        
        # Create pattern/insight windows
        if context_analysis.get("needs_patterns", False):
            pattern_window = await self._create_pattern_window(query_context)
            if pattern_window:
                windows.append(pattern_window)
        
        # Rank windows by relevance and priority
        ranked_windows = self._rank_windows(windows, strategy)
        
        # Store windows
        for window in ranked_windows[:max_windows]:
            self.context_windows[window.window_id] = window
        
        return ranked_windows[:max_windows]
    
    async def _analyze_query_context(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine context needs"""
        query_lower = query.lower()
        
        analysis = {
            "needs_conversation_context": True,  # Default to always include
            "needs_project_context": True,
            "needs_decision_context": any(keyword in query_lower for keyword in [
                "decide", "choice", "option", "alternative", "should", "recommend"
            ]),
            "needs_patterns": any(keyword in query_lower for keyword in [
                "pattern", "similar", "like", "approach", "strategy", "best practice"
            ]),
            "technical_domains": []
        }
        
        # Identify technical domains
        domain_keywords = {
            "database": ["database", "sql", "nosql", "postgres", "mongo", "redis"],
            "web_development": ["frontend", "backend", "api", "web", "http", "rest"],
            "devops": ["deploy", "ci/cd", "docker", "kubernetes", "infrastructure"],
            "ai_ml": ["ai", "ml", "machine learning", "model", "training"],
            "security": ["security", "auth", "encryption", "vulnerability"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                analysis["technical_domains"].append(domain)
        
        return analysis
    
    async def _create_conversation_window(self, query_context: str) -> Optional[ContextWindow]:
        """Create conversation context window"""
        # Get recent conversation memories
        recent_memories = await self._search_memories_by_type("conversation", limit=50)
        
        if not recent_memories:
            return None
        
        window = ContextWindow(
            window_id=f"conv_{uuid.uuid4().hex[:8]}",
            window_type=ContextType.CONVERSATION,
            priority=8,  # High priority for conversation context
            relevance_score=0.9,
            estimated_size_mb=2.0,
            estimated_load_time_ms=500,
            memory_ids=[m["memory_id"] for m in recent_memories],
            tag_filters=["conversation", "context"]
        )
        
        return window
    
    async def _create_project_window(self, query_context: str) -> Optional[ContextWindow]:
        """Create project context window"""
        # Get project-related memories
        project_memories = await self._search_memories_by_tags(["project", "code", "implementation"], limit=30)
        
        if not project_memories:
            return None
        
        window = ContextWindow(
            window_id=f"proj_{uuid.uuid4().hex[:8]}",
            window_type=ContextType.PROJECT,
            priority=7,
            relevance_score=0.8,
            estimated_size_mb=3.0,
            estimated_load_time_ms=800,
            memory_ids=[m["memory_id"] for m in project_memories],
            tag_filters=["project", "code", "implementation"]
        )
        
        return window
    
    async def _create_technical_window(self, query_context: str, domain: str) -> Optional[ContextWindow]:
        """Create technical domain context window"""
        # Get technical memories for specific domain
        technical_memories = await self._search_memories_by_tags([domain, "technical"], limit=20)
        
        if not technical_memories:
            return None
        
        # Calculate relevance based on query similarity
        relevance = await self._calculate_domain_relevance(query_context, domain)
        
        window = ContextWindow(
            window_id=f"tech_{domain}_{uuid.uuid4().hex[:8]}",
            window_type=ContextType.TECHNICAL,
            priority=6,
            relevance_score=relevance,
            estimated_size_mb=1.5,
            estimated_load_time_ms=400,
            memory_ids=[m["memory_id"] for m in technical_memories],
            tag_filters=[domain, "technical"]
        )
        
        return window
    
    async def _create_decision_window(self, query_context: str) -> Optional[ContextWindow]:
        """Create decision context window"""
        decision_memories = await self._search_memories_by_tags(["decision", "choice"], limit=15)
        
        if not decision_memories:
            return None
        
        window = ContextWindow(
            window_id=f"decision_{uuid.uuid4().hex[:8]}",
            window_type=ContextType.DECISION,
            priority=5,
            relevance_score=0.7,
            estimated_size_mb=1.0,
            estimated_load_time_ms=300,
            memory_ids=[m["memory_id"] for m in decision_memories],
            tag_filters=["decision", "choice", "analysis"]
        )
        
        return window
    
    async def _create_pattern_window(self, query_context: str) -> Optional[ContextWindow]:
        """Create pattern/insight context window"""
        pattern_memories = await self._search_memories_by_tags(["pattern", "insight", "best_practice"], limit=10)
        
        if not pattern_memories:
            return None
        
        window = ContextWindow(
            window_id=f"pattern_{uuid.uuid4().hex[:8]}",
            window_type=ContextType.PATTERN,
            priority=4,
            relevance_score=0.6,
            estimated_size_mb=0.8,
            estimated_load_time_ms=200,
            memory_ids=[m["memory_id"] for m in pattern_memories],
            tag_filters=["pattern", "insight", "best_practice"]
        )
        
        return window
    
    async def _search_memories_by_type(self, memory_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search memories by type"""
        # Mock implementation - would integrate with actual memory system
        return [
            {"memory_id": f"mem_{i}", "type": memory_type, "content": f"Mock {memory_type} memory {i}"}
            for i in range(min(limit, 10))
        ]
    
    async def _search_memories_by_tags(self, tags: List[str], limit: int = 50) -> List[Dict[str, Any]]:
        """Search memories by tags"""
        # Mock implementation - would integrate with actual memory system
        return [
            {"memory_id": f"mem_tag_{i}", "tags": tags, "content": f"Mock memory with tags {tags}"}
            for i in range(min(limit, 8))
        ]
    
    async def _calculate_domain_relevance(self, query: str, domain: str) -> float:
        """Calculate relevance of domain to query"""
        # Simple keyword-based relevance calculation
        domain_keywords = {
            "database": ["data", "store", "query", "table", "sql"],
            "web_development": ["web", "api", "frontend", "backend", "http"],
            "devops": ["deploy", "build", "ci", "cd", "infrastructure"],
            "ai_ml": ["ai", "model", "train", "predict", "learn"],
            "security": ["secure", "auth", "encrypt", "protect", "safe"]
        }
        
        query_lower = query.lower()
        keywords = domain_keywords.get(domain, [])
        
        matches = sum(1 for keyword in keywords if keyword in query_lower)
        relevance = min(1.0, matches / len(keywords)) if keywords else 0.5
        
        return relevance
    
    def _rank_windows(self, windows: List[ContextWindow], strategy: LoadingStrategy) -> List[ContextWindow]:
        """Rank context windows by strategy"""
        
        if strategy == LoadingStrategy.RELEVANCE_FIRST:
            return sorted(windows, key=lambda w: w.relevance_score, reverse=True)
        
        elif strategy == LoadingStrategy.PRIORITY_FIRST:
            return sorted(windows, key=lambda w: w.priority, reverse=True)
        
        elif strategy == LoadingStrategy.BALANCED:
            # Balanced scoring: priority * 0.6 + relevance * 0.4
            for window in windows:
                window.balanced_score = (window.priority / 10) * 0.6 + window.relevance_score * 0.4
            return sorted(windows, key=lambda w: w.balanced_score, reverse=True)
        
        elif strategy == LoadingStrategy.TIME_BASED:
            # Favor faster loading windows
            return sorted(windows, key=lambda w: w.estimated_load_time_ms)
        
        else:  # ADAPTIVE
            # Adaptive strategy based on current context
            return self._adaptive_ranking(windows)
    
    def _adaptive_ranking(self, windows: List[ContextWindow]) -> List[ContextWindow]:
        """Adaptive ranking based on current system state"""
        # Consider cache hits, system load, and past performance
        for window in windows:
            score = 0.0
            
            # Base score from priority and relevance
            score += (window.priority / 10) * 0.4
            score += window.relevance_score * 0.4
            
            # Cache hit bonus
            if window.cache_key in self.context_cache:
                score += 0.15
            
            # Size penalty for large windows
            if window.estimated_size_mb > 5.0:
                score -= 0.1
            
            # Speed bonus for fast loading
            if window.estimated_load_time_ms < 1000:
                score += 0.05
            
            window.adaptive_score = score
        
        return sorted(windows, key=lambda w: w.adaptive_score, reverse=True)
    
    async def create_loading_plan(
        self,
        windows: List[ContextWindow],
        strategy: LoadingStrategy = LoadingStrategy.BALANCED,
        max_memory_mb: float = None,
        max_time_ms: float = None,
        max_concurrent: int = None
    ) -> LoadingPlan:
        """Create incremental loading plan"""
        
        # Use default values if not specified
        max_memory_mb = max_memory_mb or self.config["default_max_memory_mb"]
        max_time_ms = max_time_ms or self.config["default_max_load_time_ms"]
        max_concurrent = max_concurrent or self.config["default_max_concurrent"]
        
        # Calculate phases based on dependencies and resource constraints
        phases = await self._calculate_loading_phases(windows, max_memory_mb, max_concurrent)
        
        # Calculate totals
        total_size = sum(w.estimated_size_mb for w in windows)
        total_time = sum(max(w.estimated_load_time_ms for w in phase) for phase in phases) if phases else 0
        
        plan = LoadingPlan(
            plan_id=str(uuid.uuid4()),
            strategy=strategy,
            total_windows=len(windows),
            estimated_total_size_mb=total_size,
            estimated_total_time_ms=total_time,
            phases=[[w.window_id for w in phase] for phase in phases],
            phase_priorities=[max(w.priority for w in phase) for phase in phases],
            max_memory_mb=max_memory_mb,
            max_loading_time_ms=max_time_ms,
            max_concurrent_loads=max_concurrent
        )
        
        # Store plan
        self.loading_plans[plan.plan_id] = plan
        await self._save_loading_plan(plan)
        
        return plan
    
    async def _calculate_loading_phases(
        self,
        windows: List[ContextWindow],
        max_memory_mb: float,
        max_concurrent: int
    ) -> List[List[ContextWindow]]:
        """Calculate loading phases based on constraints"""
        phases = []
        remaining_windows = windows.copy()
        
        while remaining_windows:
            current_phase = []
            current_memory = 0.0
            
            # Add windows to current phase until constraints are hit
            for window in remaining_windows[:]:
                if (len(current_phase) < max_concurrent and 
                    current_memory + window.estimated_size_mb <= max_memory_mb):
                    
                    # Check dependencies
                    if self._dependencies_satisfied(window, phases):
                        current_phase.append(window)
                        current_memory += window.estimated_size_mb
                        remaining_windows.remove(window)
            
            if current_phase:
                phases.append(current_phase)
            else:
                # Force add at least one window to avoid infinite loop
                if remaining_windows:
                    phases.append([remaining_windows.pop(0)])
        
        return phases
    
    def _dependencies_satisfied(self, window: ContextWindow, completed_phases: List[List[ContextWindow]]) -> bool:
        """Check if window dependencies are satisfied"""
        if not window.depends_on:
            return True
        
        completed_window_ids = set()
        for phase in completed_phases:
            completed_window_ids.update(w.window_id for w in phase)
        
        return all(dep_id in completed_window_ids for dep_id in window.depends_on)
    
    async def execute_loading_plan(self, plan: LoadingPlan) -> LoadingProgress:
        """Execute incremental loading plan"""
        
        progress = LoadingProgress(
            progress_id=str(uuid.uuid4()),
            plan_id=plan.plan_id,
            total_windows=plan.total_windows,
            completed_windows=0,
            failed_windows=0,
            loaded_size_mb=0.0,
            cached_size_mb=0.0,
            peak_memory_mb=0.0,
            start_time=datetime.now().isoformat(),
            last_update=datetime.now().isoformat()
        )
        
        self.active_progress[progress.progress_id] = progress
        
        try:
            # Execute phases sequentially
            for phase_idx, window_ids in enumerate(plan.phases):
                progress.current_phase = phase_idx
                await self._execute_phase(window_ids, progress)
                
                # Update progress
                progress.last_update = datetime.now().isoformat()
                await self._save_progress(progress)
                
                # Check if we should stop early
                if await self._should_stop_loading(progress, plan):
                    break
            
            # Calculate final metrics
            await self._finalize_loading_progress(progress)
            
        except Exception as e:
            logger.error(f"Error executing loading plan: {e}")
            progress.last_update = datetime.now().isoformat()
        
        return progress
    
    async def _execute_phase(self, window_ids: List[str], progress: LoadingProgress):
        """Execute a single loading phase"""
        # Load windows in parallel
        loading_tasks = []
        
        for window_id in window_ids:
            window = self.context_windows.get(window_id)
            if window:
                task = self._load_context_window(window, progress)
                loading_tasks.append(task)
        
        # Wait for all windows in phase to complete
        if loading_tasks:
            await asyncio.gather(*loading_tasks, return_exceptions=True)
    
    async def _load_context_window(self, window: ContextWindow, progress: LoadingProgress):
        """Load a single context window"""
        start_time = time.time()
        window.status = LoadingStatus.LOADING
        window.load_start_time = datetime.now().isoformat()
        
        try:
            # Check cache first
            if window.cache_key in self.context_cache:
                cache_entry = self.context_cache[window.cache_key]
                
                # Validate cache
                if datetime.fromisoformat(cache_entry.expires_at) > datetime.now():
                    # Cache hit
                    window.status = LoadingStatus.CACHED
                    cache_entry.hit_count += 1
                    cache_entry.last_accessed = datetime.now().isoformat()
                    
                    progress.cached_size_mb += cache_entry.size_mb
                    self.performance_stats["cache_hits"] += 1
                    
                    logger.debug(f"Cache hit for window {window.window_id}")
                    return cache_entry.content
                else:
                    # Cache expired
                    del self.context_cache[window.cache_key]
            
            # Load from memory system
            content = await self._load_window_content(window)
            
            # Calculate actual size
            content_size = len(json.dumps(content).encode()) / (1024 * 1024)  # MB
            window.actual_size_mb = content_size
            
            # Cache the content
            if self.config.get("enable_caching", True):
                await self._cache_window_content(window, content, content_size)
            
            # Update progress
            progress.loaded_size_mb += content_size
            progress.completed_windows += 1
            progress.peak_memory_mb = max(progress.peak_memory_mb, 
                                        progress.loaded_size_mb + progress.cached_size_mb)
            
            window.status = LoadingStatus.LOADED
            window.load_end_time = datetime.now().isoformat()
            
            # Update performance stats
            load_time = (time.time() - start_time) * 1000  # ms
            self.performance_stats["total_loads"] += 1
            self.performance_stats["cache_misses"] += 1
            
            logger.debug(f"Loaded window {window.window_id} in {load_time:.2f}ms")
            
            return content
            
        except Exception as e:
            window.status = LoadingStatus.FAILED
            progress.failed_windows += 1
            logger.error(f"Failed to load window {window.window_id}: {e}")
            return {}
    
    async def _load_window_content(self, window: ContextWindow) -> Dict[str, Any]:
        """Load content for a context window"""
        content = {
            "window_id": window.window_id,
            "window_type": window.window_type.value,
            "memories": []
        }
        
        # Load memories based on window configuration
        if window.memory_ids:
            # Load specific memories
            for memory_id in window.memory_ids:
                memory = await self._load_memory_by_id(memory_id)
                if memory:
                    content["memories"].append(memory)
        
        if window.tag_filters:
            # Load memories by tags
            tag_memories = await self._load_memories_by_tags(window.tag_filters)
            content["memories"].extend(tag_memories)
        
        # Apply time range filter if specified
        if window.time_range:
            content["memories"] = self._filter_memories_by_time(content["memories"], window.time_range)
        
        return content
    
    async def _load_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Load memory by ID"""
        # Mock implementation - would integrate with actual memory system
        return {
            "memory_id": memory_id,
            "content": f"Mock memory content for {memory_id}",
            "timestamp": datetime.now().isoformat(),
            "tags": ["mock"]
        }
    
    async def _load_memories_by_tags(self, tags: List[str]) -> List[Dict[str, Any]]:
        """Load memories by tags"""
        # Mock implementation - would integrate with actual memory system
        return [
            {
                "memory_id": f"tag_mem_{i}",
                "content": f"Mock memory with tags {tags}",
                "timestamp": datetime.now().isoformat(),
                "tags": tags
            }
            for i in range(5)  # Mock 5 memories per tag search
        ]
    
    def _filter_memories_by_time(self, memories: List[Dict[str, Any]], time_range: Tuple[str, str]) -> List[Dict[str, Any]]:
        """Filter memories by time range"""
        start_time, end_time = time_range
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
        
        filtered = []
        for memory in memories:
            if "timestamp" in memory:
                memory_time = datetime.fromisoformat(memory["timestamp"])
                if start_dt <= memory_time <= end_dt:
                    filtered.append(memory)
        
        return filtered
    
    async def _cache_window_content(self, window: ContextWindow, content: Dict[str, Any], size_mb: float):
        """Cache window content"""
        cache_entry = ContextCache(
            cache_id=window.cache_key,
            window_id=window.window_id,
            content=content,
            size_mb=size_mb,
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            ttl_seconds=self.config["cache_ttl_seconds"]
        )
        
        # Check cache size limits
        current_cache_size = sum(entry.size_mb for entry in self.context_cache.values())
        max_cache_size = self.config["cache_max_size_mb"]
        
        if current_cache_size + size_mb > max_cache_size:
            await self._evict_cache_entries(size_mb)
        
        # Store in cache
        self.context_cache[cache_entry.cache_id] = cache_entry
        await self._save_cache_entry(cache_entry)
    
    async def _evict_cache_entries(self, needed_space_mb: float):
        """Evict cache entries to make space"""
        # LRU eviction strategy
        entries_by_access = sorted(
            self.context_cache.values(),
            key=lambda e: e.last_accessed
        )
        
        freed_space = 0.0
        for entry in entries_by_access:
            if freed_space >= needed_space_mb:
                break
            
            # Remove entry
            freed_space += entry.size_mb
            del self.context_cache[entry.cache_id]
            
            # Remove cache file
            cache_file = self.cache_dir / f"{entry.cache_id}.json"
            if cache_file.exists():
                cache_file.unlink()
    
    async def _should_stop_loading(self, progress: LoadingProgress, plan: LoadingPlan) -> bool:
        """Check if loading should stop early"""
        # Check resource limits
        if progress.peak_memory_mb > plan.max_memory_mb * 1.2:  # 20% tolerance
            logger.warning("Memory limit exceeded, stopping loading")
            return True
        
        # Check time limits
        elapsed_time = time.time() - datetime.fromisoformat(progress.start_time).timestamp()
        if elapsed_time * 1000 > plan.max_loading_time_ms:
            logger.warning("Time limit exceeded, stopping loading")
            return True
        
        # Check failure rate
        total_processed = progress.completed_windows + progress.failed_windows
        if total_processed > 0 and progress.failed_windows / total_processed > 0.5:
            logger.warning("High failure rate, stopping loading")
            return True
        
        return False
    
    async def _finalize_loading_progress(self, progress: LoadingProgress):
        """Finalize loading progress with final metrics"""
        end_time = datetime.now()
        start_time = datetime.fromisoformat(progress.start_time)
        total_time_seconds = (end_time - start_time).total_seconds()
        
        # Calculate final metrics
        if progress.completed_windows > 0:
            progress.average_load_time_ms = (total_time_seconds * 1000) / progress.completed_windows
        
        total_cache_operations = self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]
        if total_cache_operations > 0:
            progress.cache_hit_rate = self.performance_stats["cache_hits"] / total_cache_operations
        
        if total_time_seconds > 0:
            progress.throughput_mb_per_sec = progress.loaded_size_mb / total_time_seconds
        
        progress.estimated_completion = end_time.isoformat()
    
    async def _save_loading_plan(self, plan: LoadingPlan):
        """Save loading plan to disk"""
        plan_file = self.plans_dir / f"{plan.plan_id}.json"
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(plan), f, indent=2, default=str)
    
    async def _save_progress(self, progress: LoadingProgress):
        """Save loading progress to disk"""
        progress_file = self.progress_dir / f"{progress.progress_id}.json"
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(progress), f, indent=2, default=str)
    
    async def _save_cache_entry(self, cache_entry: ContextCache):
        """Save cache entry to disk"""
        cache_file = self.cache_dir / f"{cache_entry.cache_id}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(cache_entry), f, indent=2, default=str)
    
    async def get_loading_status(self, progress_id: str) -> Optional[LoadingProgress]:
        """Get loading status"""
        return self.active_progress.get(progress_id)
    
    async def get_cached_content(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached content"""
        cache_entry = self.context_cache.get(cache_key)
        if cache_entry and datetime.fromisoformat(cache_entry.expires_at) > datetime.now():
            cache_entry.access_count += 1
            cache_entry.last_accessed = datetime.now().isoformat()
            return cache_entry.content
        return None
    
    async def clear_cache(self, older_than_hours: int = 24):
        """Clear old cache entries"""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        to_remove = []
        for cache_id, entry in self.context_cache.items():
            if datetime.fromisoformat(entry.created_at) < cutoff_time:
                to_remove.append(cache_id)
        
        for cache_id in to_remove:
            del self.context_cache[cache_id]
            cache_file = self.cache_dir / f"{cache_id}.json"
            if cache_file.exists():
                cache_file.unlink()
        
        logger.info(f"Cleared {len(to_remove)} cache entries")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        
        # Add cache statistics
        stats["cache_size_mb"] = sum(entry.size_mb for entry in self.context_cache.values())
        stats["cache_entries"] = len(self.context_cache)
        
        # Add active loading statistics
        stats["active_loadings"] = len(self.active_progress)
        stats["total_windows"] = len(self.context_windows)
        
        return stats


# Global incremental context loader
incremental_loader = IncrementalContextLoader()

# Convenience functions
async def create_context_windows(query: str, max_windows: int = 10, 
                                strategy: LoadingStrategy = LoadingStrategy.BALANCED) -> List[ContextWindow]:
    """Create context windows for incremental loading"""
    return await incremental_loader.create_context_windows(query, max_windows, strategy)

async def load_context_incrementally(windows: List[ContextWindow], 
                                   strategy: LoadingStrategy = LoadingStrategy.BALANCED) -> LoadingProgress:
    """Load context incrementally"""
    plan = await incremental_loader.create_loading_plan(windows, strategy)
    return await incremental_loader.execute_loading_plan(plan)

async def get_loading_progress(progress_id: str) -> Optional[LoadingProgress]:
    """Get loading progress"""
    return await incremental_loader.get_loading_status(progress_id)

async def clear_context_cache(older_than_hours: int = 24):
    """Clear context cache"""
    await incremental_loader.clear_cache(older_than_hours)

async def get_loader_stats() -> Dict[str, Any]:
    """Get loader performance statistics"""
    return await incremental_loader.get_performance_stats()

if __name__ == "__main__":
    # Test the incremental context loading system
    async def test_incremental_loading():
        print("📥 Testing Incremental Context Loading")
        
        # Test context window creation
        query = "How should I implement caching with Redis for our microservice API?"
        windows = await create_context_windows(query, max_windows=5, strategy=LoadingStrategy.BALANCED)
        print(f"✅ Created {len(windows)} context windows")
        
        for window in windows:
            print(f"   Window: {window.window_type.value} (Priority: {window.priority}, Relevance: {window.relevance_score:.2f})")
        
        # Test loading plan creation
        plan = await incremental_loader.create_loading_plan(
            windows, 
            strategy=LoadingStrategy.BALANCED,
            max_memory_mb=100,
            max_time_ms=10000
        )
        print(f"✅ Created loading plan with {len(plan.phases)} phases")
        print(f"   Estimated size: {plan.estimated_total_size_mb:.2f} MB")
        print(f"   Estimated time: {plan.estimated_total_time_ms:.0f} ms")
        
        # Test loading execution
        progress = await incremental_loader.execute_loading_plan(plan)
        print(f"✅ Loading completed:")
        print(f"   Completed: {progress.completed_windows}/{progress.total_windows} windows")
        print(f"   Failed: {progress.failed_windows} windows")
        print(f"   Loaded: {progress.loaded_size_mb:.2f} MB")
        print(f"   Cached: {progress.cached_size_mb:.2f} MB")
        print(f"   Cache hit rate: {progress.cache_hit_rate:.2%}")
        
        # Test performance stats
        stats = await get_loader_stats()
        print(f"✅ Performance stats:")
        print(f"   Total loads: {stats['total_loads']}")
        print(f"   Cache hits: {stats['cache_hits']}")
        print(f"   Cache size: {stats['cache_size_mb']:.2f} MB")
        
        print("✅ Incremental Context Loading ready!")
    
    asyncio.run(test_incremental_loading())