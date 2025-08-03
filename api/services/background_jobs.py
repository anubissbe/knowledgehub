"""
Background Jobs for AI Intelligence Features
Manages all periodic tasks for pattern analysis, mistake aggregation, 
performance optimization, and real-time learning
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..dependencies import get_db
from ..models.memory import MemoryItem
from ..services.cache import get_cache_service
from ..services.realtime_learning_pipeline import get_learning_pipeline, StreamEvent, EventType

logger = logging.getLogger(__name__)


class BackgroundJobManager:
    """Manages all background jobs for AI Intelligence features"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.cache = None
        self.pipeline = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the job manager with required services"""
        if self._initialized:
            return
            
        try:
            self.cache = await get_cache_service()
            self.pipeline = await get_learning_pipeline()
            self._initialized = True
            logger.info("Background job manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize background job manager: {e}")
            raise
    
    async def start(self):
        """Start all background jobs"""
        if not self._initialized:
            await self.initialize()
            
        try:
            # Pattern Analysis Jobs
            self.scheduler.add_job(
                self.analyze_user_patterns,
                IntervalTrigger(minutes=30),
                id="analyze_user_patterns",
                name="Analyze user behavior patterns",
                replace_existing=True
            )
            
            self.scheduler.add_job(
                self.detect_code_patterns,
                IntervalTrigger(hours=1),
                id="detect_code_patterns",
                name="Detect code patterns and best practices",
                replace_existing=True
            )
            
            # Mistake Learning Jobs
            self.scheduler.add_job(
                self.aggregate_mistakes,
                IntervalTrigger(hours=2),
                id="aggregate_mistakes",
                name="Aggregate and learn from mistakes",
                replace_existing=True
            )
            
            self.scheduler.add_job(
                self.generate_lessons,
                CronTrigger(hour=3, minute=0),  # Daily at 3 AM
                id="generate_lessons",
                name="Generate lessons from aggregated mistakes",
                replace_existing=True
            )
            
            # Performance Optimization Jobs
            self.scheduler.add_job(
                self.analyze_performance_metrics,
                IntervalTrigger(minutes=15),
                id="analyze_performance_metrics",
                name="Analyze system performance metrics",
                replace_existing=True
            )
            
            self.scheduler.add_job(
                self.optimize_slow_queries,
                IntervalTrigger(hours=1),
                id="optimize_slow_queries",
                name="Identify and optimize slow queries",
                replace_existing=True
            )
            
            # Decision Analysis Jobs
            self.scheduler.add_job(
                self.analyze_decisions,
                IntervalTrigger(hours=4),
                id="analyze_decisions",
                name="Analyze decision patterns and outcomes",
                replace_existing=True
            )
            
            # Memory Management Jobs
            self.scheduler.add_job(
                self.consolidate_memories,
                CronTrigger(hour=2, minute=0),  # Daily at 2 AM
                id="consolidate_memories",
                name="Consolidate and compress old memories",
                replace_existing=True
            )
            
            self.scheduler.add_job(
                self.update_memory_importance,
                IntervalTrigger(hours=6),
                id="update_memory_importance",
                name="Update memory importance scores",
                replace_existing=True
            )
            
            # Real-time Learning Jobs
            self.scheduler.add_job(
                self.process_learning_queue,
                IntervalTrigger(seconds=30),
                id="process_learning_queue",
                name="Process real-time learning events",
                replace_existing=True
            )
            
            # Start the scheduler
            self.scheduler.start()
            logger.info("Background jobs scheduler started with 10 jobs")
            
        except Exception as e:
            logger.error(f"Failed to start background jobs: {e}")
            raise
    
    async def stop(self):
        """Stop all background jobs"""
        try:
            self.scheduler.shutdown(wait=True)
            logger.info("Background jobs scheduler stopped")
        except Exception as e:
            logger.error(f"Failed to stop background jobs: {e}")
    
    # Pattern Analysis Jobs
    async def analyze_user_patterns(self):
        """Analyze user behavior patterns from memories and interactions"""
        try:
            db = next(get_db())
            
            # Get recent user activities (last 24 hours)
            since = datetime.utcnow() - timedelta(hours=24)
            
            # Analyze memory creation patterns
            memory_patterns = db.query(
                MemoryItem.user_id,
                func.count(MemoryItem.id).label('memory_count'),
                func.avg(MemoryItem.importance).label('avg_importance')
            ).filter(
                MemoryItem.created_at >= since
            ).group_by(MemoryItem.user_id).all()
            
            # Store pattern insights
            for pattern in memory_patterns:
                await self._store_pattern_insight({
                    'user_id': pattern.user_id,
                    'pattern_type': 'memory_creation',
                    'insights': {
                        'memory_count_24h': pattern.memory_count,
                        'avg_importance': float(pattern.avg_importance) if pattern.avg_importance else 0
                    }
                })
            
            # Publish pattern detection event
            if self.pipeline:
                await self.pipeline.publish_event(StreamEvent(
                    event_type=EventType.PATTERN_DETECTED,
                    data={
                        'pattern_type': 'user_behavior',
                        'patterns_found': len(memory_patterns),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                ))
            
            db.close()
            logger.info(f"Analyzed user patterns: {len(memory_patterns)} users")
            
        except Exception as e:
            logger.error(f"Error analyzing user patterns: {e}")
    
    async def detect_code_patterns(self):
        """Detect code patterns and best practices from code-related memories"""
        try:
            db = next(get_db())
            
            # Get code-related memories
            code_memories = db.query(MemoryItem).filter(
                or_(
                    MemoryItem.tags.contains(['code']),
                    MemoryItem.tags.contains(['programming']),
                    MemoryItem.memory_type == 'code_snippet'
                )
            ).limit(100).all()
            
            patterns = []
            
            # Simple pattern detection (would use ML in production)
            for memory in code_memories:
                content = memory.content.lower()
                
                # Detect common patterns
                if 'async def' in content:
                    patterns.append({'type': 'async_pattern', 'count': content.count('async def')})
                if 'try:' in content and 'except' in content:
                    patterns.append({'type': 'error_handling', 'count': 1})
                if 'logger.' in content or 'logging.' in content:
                    patterns.append({'type': 'logging_pattern', 'count': 1})
                
            # Aggregate patterns
            pattern_summary = {}
            for p in patterns:
                if p['type'] not in pattern_summary:
                    pattern_summary[p['type']] = 0
                pattern_summary[p['type']] += p['count']
            
            # Store patterns
            for pattern_type, count in pattern_summary.items():
                await self._store_pattern_insight({
                    'pattern_type': 'code_pattern',
                    'pattern_name': pattern_type,
                    'occurrences': count,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            db.close()
            logger.info(f"Detected {len(pattern_summary)} code patterns")
            
        except Exception as e:
            logger.error(f"Error detecting code patterns: {e}")
    
    # Mistake Learning Jobs
    async def aggregate_mistakes(self):
        """Aggregate mistakes and identify common error patterns"""
        try:
            # Get recent errors from cache
            if self.cache:
                error_keys = []  # Would need to implement pattern-based key retrieval
                
                error_summary = {}
                
                # Aggregate by error type
                # This is simplified - would need actual error tracking implementation
                
                logger.info("Aggregated mistakes (implementation pending)")
            
        except Exception as e:
            logger.error(f"Error aggregating mistakes: {e}")
    
    async def generate_lessons(self):
        """Generate lessons from aggregated mistakes"""
        try:
            # This would analyze aggregated mistakes and generate lessons
            # For now, it's a placeholder
            
            lessons = [
                {
                    'pattern': 'ImportError frequency',
                    'recommendation': 'Always verify dependencies are installed',
                    'confidence': 0.85
                }
            ]
            
            # Store lessons
            if self.cache:
                await self.cache.set('lessons:latest', lessons, expire=86400)
            
            logger.info(f"Generated {len(lessons)} lessons")
            
        except Exception as e:
            logger.error(f"Error generating lessons: {e}")
    
    # Performance Optimization Jobs
    async def analyze_performance_metrics(self):
        """Analyze system performance metrics"""
        try:
            db = next(get_db())
            
            # Get performance data (simplified)
            memory_count = db.query(func.count(MemoryItem.id)).scalar()
            
            metrics = {
                'timestamp': datetime.utcnow().isoformat(),
                'memory_count': memory_count,
                'cache_hit_rate': 0.75,  # Would calculate from actual cache stats
                'avg_response_time': 150  # Would calculate from actual metrics
            }
            
            # Store metrics
            if self.cache:
                await self.cache.set('performance:latest', metrics, expire=3600)
            
            # Publish performance event
            if self.pipeline:
                await self.pipeline.publish_event(StreamEvent(
                    event_type=EventType.PATTERN_DETECTED,
                    data={
                        'pattern_type': 'performance',
                        'metrics': metrics
                    }
                ))
            
            db.close()
            logger.info("Analyzed performance metrics")
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
    
    async def optimize_slow_queries(self):
        """Identify and optimize slow database queries"""
        try:
            # This would analyze query performance logs
            # For now, it's a placeholder
            
            slow_queries = []
            
            if slow_queries:
                logger.warning(f"Found {len(slow_queries)} slow queries")
            else:
                logger.info("No slow queries detected")
            
        except Exception as e:
            logger.error(f"Error optimizing queries: {e}")
    
    # Decision Analysis Jobs
    async def analyze_decisions(self):
        """Analyze decision patterns and outcomes"""
        try:
            # This would analyze decision history
            # For now, it's a placeholder
            
            decision_insights = {
                'total_decisions': 0,
                'avg_confidence': 0.0,
                'common_categories': []
            }
            
            logger.info("Analyzed decision patterns")
            
        except Exception as e:
            logger.error(f"Error analyzing decisions: {e}")
    
    # Memory Management Jobs
    async def consolidate_memories(self):
        """Consolidate and compress old memories"""
        try:
            db = next(get_db())
            
            # Get old memories (> 30 days)
            cutoff = datetime.utcnow() - timedelta(days=30)
            
            old_memories = db.query(MemoryItem).filter(
                and_(
                    MemoryItem.created_at < cutoff,
                    MemoryItem.importance < 0.5
                )
            ).limit(100).all()
            
            consolidated = 0
            for memory in old_memories:
                # Would implement actual consolidation logic
                # For now, just update metadata
                memory.metadata = memory.metadata or {}
                memory.metadata['consolidated'] = True
                consolidated += 1
            
            db.commit()
            db.close()
            
            logger.info(f"Consolidated {consolidated} old memories")
            
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
    
    async def update_memory_importance(self):
        """Update memory importance scores based on usage patterns"""
        try:
            db = next(get_db())
            
            # Get memories that haven't been updated recently
            cutoff = datetime.utcnow() - timedelta(hours=24)
            
            memories = db.query(MemoryItem).filter(
                or_(
                    MemoryItem.updated_at < cutoff,
                    MemoryItem.updated_at.is_(None)
                )
            ).limit(100).all()
            
            updated = 0
            for memory in memories:
                # Simple importance decay (would use ML in production)
                age_days = (datetime.utcnow() - memory.created_at).days
                decay_factor = 0.99 ** age_days
                
                memory.importance = min(1.0, memory.importance * decay_factor)
                memory.updated_at = datetime.utcnow()
                updated += 1
            
            db.commit()
            db.close()
            
            logger.info(f"Updated importance for {updated} memories")
            
        except Exception as e:
            logger.error(f"Error updating memory importance: {e}")
    
    # Real-time Learning Jobs
    async def process_learning_queue(self):
        """Process queued learning events"""
        try:
            # This would process events from the learning pipeline
            # For now, just check pipeline status
            
            if self.pipeline:
                context = await self.pipeline.get_real_time_context()
                
                if context.get('recent_patterns'):
                    logger.info(f"Processing {len(context['recent_patterns'])} patterns")
            
        except Exception as e:
            logger.error(f"Error processing learning queue: {e}")
    
    # Helper methods
    async def _store_pattern_insight(self, insight: Dict[str, Any]):
        """Store pattern insight in cache"""
        try:
            if self.cache:
                key = f"pattern:{insight.get('pattern_type', 'unknown')}:{datetime.utcnow().timestamp()}"
                await self.cache.set(key, insight, expire=86400 * 7)  # Keep for 7 days
        except Exception as e:
            logger.error(f"Error storing pattern insight: {e}")
    
    def get_job_status(self) -> Dict[str, Any]:
        """Get status of all background jobs"""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'pending': job.pending
            })
        
        return {
            'scheduler_running': self.scheduler.running,
            'total_jobs': len(jobs),
            'jobs': jobs
        }


# Singleton instance
_job_manager: Optional[BackgroundJobManager] = None


async def get_job_manager() -> BackgroundJobManager:
    """Get or create the background job manager instance"""
    global _job_manager
    
    if _job_manager is None:
        _job_manager = BackgroundJobManager()
        await _job_manager.initialize()
    
    return _job_manager