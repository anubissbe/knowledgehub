"""
Pattern Recognition Workers
Background workers for continuous pattern analysis and learning
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..dependencies import get_db
from ..models.memory import MemoryItem
from ..models.document import Document, DocumentChunk
from .pattern_recognition_engine import get_pattern_engine, CodePattern, PatternCategory
from .realtime_learning_pipeline import get_learning_pipeline, StreamEvent, EventType
from .cache import get_cache_service

logger = logging.getLogger(__name__)


class PatternWorkerManager:
    """Manages pattern recognition background workers"""
    
    def __init__(self):
        self.workers = []
        self.running = False
        self.pattern_engine = None
        self.learning_pipeline = None
        self.cache = None
        
    async def initialize(self):
        """Initialize the worker manager"""
        try:
            self.pattern_engine = await get_pattern_engine()
            self.learning_pipeline = await get_learning_pipeline()
            self.cache = await get_cache_service()
            logger.info("Pattern worker manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pattern workers: {e}")
            raise
    
    async def start_workers(self):
        """Start all pattern recognition workers"""
        if self.running:
            logger.warning("Pattern workers already running")
            return
            
        self.running = True
        
        # Start different types of workers
        workers = [
            self._code_pattern_worker(),
            self._memory_pattern_worker(),
            self._documentation_pattern_worker(),
            self._error_pattern_worker(),
            self._performance_pattern_worker()
        ]
        
        for worker in workers:
            task = asyncio.create_task(worker)
            self.workers.append(task)
        
        logger.info(f"Started {len(self.workers)} pattern recognition workers")
    
    async def stop_workers(self):
        """Stop all pattern recognition workers"""
        self.running = False
        
        # Cancel all worker tasks
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to complete
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("All pattern workers stopped")
    
    async def _code_pattern_worker(self):
        """Worker for analyzing code patterns in memories and documents"""
        logger.info("Code pattern worker started")
        
        while self.running:
            try:
                db = next(get_db())
                
                # Get recent code memories
                since = datetime.utcnow() - timedelta(hours=1)
                # Use PostgreSQL array operators
                from sqlalchemy import text
                code_memories = db.query(MemoryItem).filter(
                    MemoryItem.created_at >= since,
                    text("tags @> ARRAY['code']")
                ).limit(50).all()
                
                for memory in code_memories:
                    if not self.running:
                        break
                        
                    # Analyze code patterns
                    patterns = await self.pattern_engine.analyze_code(
                        memory.content,
                        language="python"  # Could be detected from content
                    )
                    
                    # Store patterns
                    for pattern in patterns:
                        await self._store_pattern(pattern, memory.id)
                    
                    # Publish pattern detection event
                    if patterns:
                        await self.learning_pipeline.publish_event(StreamEvent(
                            event_type=EventType.PATTERN_DETECTED,
                            data={
                                'source': 'code_worker',
                                'memory_id': str(memory.id),
                                'patterns_found': len(patterns)
                            }
                        ))
                
                db.close()
                
                # Sleep before next iteration
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in code pattern worker: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
        
        logger.info("Code pattern worker stopped")
    
    async def _memory_pattern_worker(self):
        """Worker for analyzing patterns in user memories"""
        logger.info("Memory pattern worker started")
        
        while self.running:
            try:
                db = next(get_db())
                
                # Analyze memory creation patterns
                # Since MemoryItem doesn't have user_id, analyze global patterns
                memory_count = db.query(func.count(MemoryItem.id)).scalar() or 0
                
                # Get recent memories with tags
                recent_memories = db.query(MemoryItem).filter(
                    MemoryItem.tags != None
                ).order_by(MemoryItem.created_at.desc()).limit(100).all()
                
                # Analyze tag patterns
                tag_counts = {}
                for memory in recent_memories:
                    if memory.tags:
                        for tag in memory.tags:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
                # Store global memory patterns
                pattern_data = {
                    'total_memories': memory_count,
                    'analyzed_memories': len(recent_memories),
                    'top_tags': sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20],
                    'pattern_type': 'global_memory_trends'
                }
                
                await self.cache.set(
                    "memory_patterns:global",
                    pattern_data,
                    expiry=86400  # 24 hours
                )
                
                db.close()
                
                await asyncio.sleep(600)  # 10 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory pattern worker: {e}")
                await asyncio.sleep(60)
        
        logger.info("Memory pattern worker stopped")
    
    async def _documentation_pattern_worker(self):
        """Worker for analyzing documentation patterns"""
        logger.info("Documentation pattern worker started")
        
        while self.running:
            try:
                db = next(get_db())
                
                # Get documentation chunks by analyzing content patterns
                # Since Document doesn't have source_type, check by content patterns
                from ..models.source import KnowledgeSource
                
                doc_chunks = db.query(DocumentChunk).join(Document).join(KnowledgeSource).filter(
                    KnowledgeSource.type.in_(['documentation', 'api', 'wiki'])
                ).limit(100).all()
                
                patterns_found = 0
                
                for chunk in doc_chunks:
                    if not self.running:
                        break
                    
                    # Simple pattern detection for documentation
                    content = chunk.content.lower()
                    
                    doc_patterns = []
                    
                    # Detect common documentation patterns
                    if 'installation' in content or 'install' in content:
                        doc_patterns.append('installation_guide')
                    if 'api' in content and ('endpoint' in content or 'request' in content):
                        doc_patterns.append('api_documentation')
                    if 'example' in content or 'usage' in content:
                        doc_patterns.append('usage_example')
                    if 'configuration' in content or 'config' in content:
                        doc_patterns.append('configuration_guide')
                    
                    # Store patterns
                    if doc_patterns:
                        await self.cache.set(
                            f"doc_patterns:{chunk.id}",
                            doc_patterns,
                            expire=86400 * 7  # 7 days
                        )
                        patterns_found += len(doc_patterns)
                
                db.close()
                
                if patterns_found > 0:
                    logger.info(f"Found {patterns_found} documentation patterns")
                
                await asyncio.sleep(1800)  # 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in documentation pattern worker: {e}")
                await asyncio.sleep(60)
        
        logger.info("Documentation pattern worker stopped")
    
    async def _error_pattern_worker(self):
        """Worker for analyzing error patterns"""
        logger.info("Error pattern worker started")
        
        while self.running:
            try:
                # Get recent errors from cache
                error_keys = []  # Would need pattern-based key retrieval
                
                # Analyze error patterns
                error_patterns = {}
                
                # This is simplified - would integrate with mistake tracking
                
                await asyncio.sleep(900)  # 15 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in error pattern worker: {e}")
                await asyncio.sleep(60)
        
        logger.info("Error pattern worker stopped")
    
    async def _performance_pattern_worker(self):
        """Worker for analyzing performance patterns"""
        logger.info("Performance pattern worker started")
        
        while self.running:
            try:
                # Get performance metrics from cache
                if self.cache:
                    metrics = await self.cache.get('performance:latest')
                    
                    if metrics:
                        # Analyze performance patterns
                        if metrics.get('avg_response_time', 0) > 500:
                            # Slow response pattern detected
                            await self.learning_pipeline.publish_event(StreamEvent(
                                event_type=EventType.PATTERN_DETECTED,
                                data={
                                    'pattern_type': 'slow_response',
                                    'avg_time': metrics['avg_response_time'],
                                    'threshold': 500
                                }
                            ))
                
                await asyncio.sleep(600)  # 10 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance pattern worker: {e}")
                await asyncio.sleep(60)
        
        logger.info("Performance pattern worker stopped")
    
    async def _store_pattern(self, pattern: CodePattern, source_id: str):
        """Store detected pattern in cache"""
        try:
            if self.cache:
                key = f"pattern:{pattern.category.value}:{pattern.pattern_type}:{source_id}"
                await self.cache.set(key, {
                    'name': pattern.name,
                    'description': pattern.description,
                    'confidence': pattern.confidence,
                    'metadata': pattern.metadata,
                    'detected_at': datetime.utcnow().isoformat()
                }, expire=86400 * 30)  # Keep for 30 days
        except Exception as e:
            logger.error(f"Error storing pattern: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of pattern workers"""
        return {
            'running': self.running,
            'worker_count': len(self.workers),
            'workers': [
                {
                    'name': worker.get_name() if hasattr(worker, 'get_name') else f"Worker-{i}",
                    'done': worker.done(),
                    'cancelled': worker.cancelled()
                }
                for i, worker in enumerate(self.workers)
            ]
        }


# Singleton instance
_worker_manager: Optional[PatternWorkerManager] = None


async def get_pattern_workers() -> PatternWorkerManager:
    """Get or create the pattern worker manager"""
    global _worker_manager
    
    if _worker_manager is None:
        _worker_manager = PatternWorkerManager()
        await _worker_manager.initialize()
    
    return _worker_manager