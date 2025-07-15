"""Background session cleanup service"""

import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional
from sqlalchemy.orm import Session

from ..models import MemorySession
from ..core.session_manager import SessionManager
from ...models import get_db
from ...services.cache import redis_client

logger = logging.getLogger(__name__)


class SessionCleanupService:
    """Service for cleaning up stale and inactive sessions"""
    
    def __init__(self):
        self.cleanup_interval_hours = 6  # Run cleanup every 6 hours
        self.stale_session_hours = 24   # Sessions inactive for 24+ hours are stale
        self.old_session_days = 30      # Sessions older than 30 days get archived
        self._running = False
        self._task = None
    
    async def start(self):
        """Start the background cleanup service"""
        if self._running:
            logger.warning("Session cleanup service already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._cleanup_loop())
        logger.info("Session cleanup service started")
    
    async def stop(self):
        """Stop the background cleanup service"""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Session cleanup service stopped")
    
    async def _cleanup_loop(self):
        """Main cleanup loop"""
        while self._running:
            try:
                await self.run_cleanup()
                
                # Wait for next cleanup cycle
                await asyncio.sleep(self.cleanup_interval_hours * 3600)
                
            except asyncio.CancelledError:
                logger.info("Session cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(300)  # 5 minutes
    
    async def run_cleanup(self):
        """Run a cleanup cycle"""
        logger.info("Starting session cleanup cycle")
        
        db = next(get_db())
        try:
            session_manager = SessionManager(db)
            
            # 1. Clean up stale sessions
            stale_count = await self._cleanup_stale_sessions(session_manager)
            
            # 2. Archive old sessions
            archived_count = await self._archive_old_sessions(session_manager)
            
            # 3. Clean up orphaned cache entries
            cache_cleaned = await self._cleanup_cache()
            
            # 4. Update session statistics
            await self._update_cleanup_stats(stale_count, archived_count, cache_cleaned)
            
            logger.info(
                f"Cleanup cycle completed: {stale_count} stale sessions closed, "
                f"{archived_count} old sessions archived, "
                f"{cache_cleaned} cache entries cleaned"
            )
            
        except Exception as e:
            logger.error(f"Cleanup cycle failed: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    async def _cleanup_stale_sessions(self, session_manager: SessionManager) -> int:
        """Clean up sessions that have been inactive for too long"""
        try:
            stale_count = await session_manager.cleanup_stale_sessions(
                hours=self.stale_session_hours
            )
            logger.info(f"Cleaned up {stale_count} stale sessions")
            return stale_count
        except Exception as e:
            logger.error(f"Stale session cleanup failed: {e}")
            return 0
    
    async def _archive_old_sessions(self, session_manager: SessionManager) -> int:
        """Archive very old sessions to reduce active data size"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.old_session_days)
        
        try:
            old_sessions = session_manager.db.query(MemorySession).filter(
                MemorySession.created_at < cutoff,
                MemorySession.ended_at.isnot(None)  # Only archive ended sessions
            ).all()
            
            archived_count = 0
            
            for session in old_sessions:
                # Add archive tag
                if 'archived' not in session.tags:
                    session.add_tag('archived')
                    session.add_metadata('archived_at', datetime.now(timezone.utc).isoformat())
                    archived_count += 1
            
            if archived_count > 0:
                session_manager.db.commit()
                logger.info(f"Archived {archived_count} old sessions")
            
            return archived_count
            
        except Exception as e:
            logger.error(f"Old session archiving failed: {e}")
            session_manager.db.rollback()
            return 0
    
    async def _cleanup_cache(self) -> int:
        """Clean up orphaned or expired cache entries"""
        try:
            # Get all session cache keys
            pattern = "session:*"
            keys = await redis_client.keys(pattern)
            
            if not keys:
                return 0
            
            cleaned_count = 0
            db = next(get_db())
            
            try:
                for key in keys:
                    # Extract session ID from key
                    session_id = key.replace("session:", "")
                    
                    # Check if session exists in database
                    session_exists = db.query(MemorySession).filter_by(id=session_id).first()
                    
                    if not session_exists:
                        # Session doesn't exist in DB, remove from cache
                        await redis_client.delete(key)
                        cleaned_count += 1
                        logger.debug(f"Removed orphaned cache entry: {key}")
                
                logger.info(f"Cleaned {cleaned_count} orphaned cache entries")
                return cleaned_count
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return 0
    
    async def _update_cleanup_stats(self, stale_count: int, archived_count: int, cache_cleaned: int):
        """Update cleanup statistics in Redis"""
        try:
            stats = {
                'last_cleanup': datetime.now(timezone.utc).isoformat(),
                'stale_sessions_cleaned': stale_count,
                'sessions_archived': archived_count,
                'cache_entries_cleaned': cache_cleaned,
                'total_cleanups': await self._increment_cleanup_counter()
            }
            
            await redis_client.set('session_cleanup_stats', stats, 86400 * 7)  # Keep for 7 days
            logger.debug(f"Updated cleanup stats: {stats}")
            
        except Exception as e:
            logger.error(f"Failed to update cleanup stats: {e}")
    
    async def _increment_cleanup_counter(self) -> int:
        """Increment and return the total cleanup counter"""
        try:
            counter_key = 'session_cleanup_counter'
            current = await redis_client.get(counter_key) or 0
            new_count = int(current) + 1
            await redis_client.set(counter_key, new_count, 86400 * 30)  # Keep for 30 days
            return new_count
        except Exception:
            return 1
    
    async def get_cleanup_stats(self) -> dict:
        """Get current cleanup statistics"""
        try:
            stats = await redis_client.get('session_cleanup_stats')
            if stats:
                return stats
            else:
                return {
                    'last_cleanup': None,
                    'stale_sessions_cleaned': 0,
                    'sessions_archived': 0,
                    'cache_entries_cleaned': 0,
                    'total_cleanups': 0
                }
        except Exception as e:
            logger.error(f"Failed to get cleanup stats: {e}")
            return {}
    
    async def manual_cleanup(self) -> dict:
        """Run a manual cleanup cycle and return results"""
        logger.info("Manual session cleanup requested")
        
        start_time = datetime.now(timezone.utc)
        
        try:
            await self.run_cleanup()
            stats = await self.get_cleanup_stats()
            
            return {
                'success': True,
                'started_at': start_time.isoformat(),
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Manual cleanup failed: {e}")
            return {
                'success': False,
                'started_at': start_time.isoformat(),
                'error': str(e)
            }


# Global instance
session_cleanup_service = SessionCleanupService()