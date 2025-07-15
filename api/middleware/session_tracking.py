"""Session tracking middleware for automatic Claude-Code session management"""

import logging
import time
import uuid
from typing import Optional, Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class SessionTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to automatically track Claude-Code sessions and inject context.
    
    This middleware:
    1. Detects Claude-Code requests via headers or patterns
    2. Manages session lifecycle (start/continue/end)
    3. Injects relevant context into requests
    4. Tracks conversation flow for memory system
    """
    
    def __init__(self, app, session_manager_factory=None):
        super().__init__(app)
        self.session_manager_factory = session_manager_factory
        self.excluded_paths = {
            '/health', '/api/docs', '/api/redoc', '/api/openapi.json',
            '/metrics', '/favicon.ico', '/static'
        }
        
    async def dispatch(self, request: Request, call_next):
        """Process request with session tracking"""
        start_time = time.time()
        logger.debug(f"Session tracking middleware processing: {request.method} {request.url.path}")
        
        # Skip session tracking for excluded paths
        if self._should_skip_tracking(request):
            return await call_next(request)
        
        # Extract session information
        session_info = await self._extract_session_info(request)
        logger.debug(f"Session info extracted: {session_info}")
        
        # Get session manager for this request
        session_manager = None
        if self.session_manager_factory:
            try:
                session_manager = self.session_manager_factory()
            except Exception as e:
                logger.warning(f"Failed to create session manager: {e}")
        
        # Initialize or continue session
        session = await self._manage_session(session_info, request, session_manager)
        
        # Inject context into request
        if session:
            await self._inject_context(request, session)
        
        # Process the request
        try:
            response = await call_next(request)
            
            # Update session after successful request
            if session and session_manager:
                await self._update_session_after_request(session, request, response, session_manager)
            
            # Add session headers to response
            if session:
                self._add_session_headers(response, session)
            
            return response
            
        except Exception as e:
            # Handle session on error
            if session and session_manager:
                await self._handle_session_error(session, request, e, session_manager)
            raise
        finally:
            # Log session tracking metrics
            processing_time = time.time() - start_time
            if session:
                logger.info(f"Session tracking: {session.id} - {processing_time:.3f}s")
    
    def _should_skip_tracking(self, request: Request) -> bool:
        """Check if request should skip session tracking"""
        path = request.url.path
        
        # Skip static files and health checks
        for excluded in self.excluded_paths:
            if path.startswith(excluded):
                return True
        
        # Skip OPTIONS requests
        if request.method == "OPTIONS":
            return True
            
        return False
    
    async def _extract_session_info(self, request: Request) -> Dict[str, Any]:
        """Extract session information from request"""
        headers = request.headers
        
        # Extract Claude-Code specific headers
        session_info = {
            'user_agent': headers.get('user-agent', ''),
            'claude_session_id': headers.get('x-claude-session-id'),
            'claude_conversation_id': headers.get('x-claude-conversation-id'),
            'claude_request_id': headers.get('x-claude-request-id'),
            'client_id': headers.get('x-client-id'),
            'request_path': request.url.path,
            'request_method': request.method,
            'remote_addr': request.client.host if request.client else None,
            'timestamp': time.time()
        }
        
        # Detect Claude-Code from User-Agent
        user_agent = session_info['user_agent'].lower()
        session_info['is_claude_code'] = any(indicator in user_agent for indicator in [
            'claude', 'anthropic', 'claude-code', 'ai-assistant'
        ])
        
        # Extract query parameters that might indicate session
        query_params = dict(request.query_params)
        session_info['query_session_id'] = query_params.get('session_id')
        session_info['query_conversation_id'] = query_params.get('conversation_id')
        
        return session_info
    
    async def _manage_session(self, session_info: Dict[str, Any], request: Request, session_manager: Any) -> Optional[Any]:
        """Initialize or continue session based on request"""
        if not session_manager:
            return None
        
        try:
            # Determine session ID from various sources
            session_id = (
                session_info.get('claude_session_id') or
                session_info.get('query_session_id') or
                self._generate_session_id_from_client(session_info)
            )
            
            if session_id:
                # Try to continue existing session
                session = await session_manager.get_or_create_session(
                    session_id=session_id,
                    metadata={
                        'user_agent': session_info['user_agent'],
                        'client_id': session_info.get('client_id'),
                        'is_claude_code': session_info['is_claude_code'],
                        'remote_addr': session_info.get('remote_addr'),
                        'started_via': 'middleware'
                    }
                )
                
                logger.info(f"Session managed: {session.id} ({'continued' if session.updated_at else 'created'})")
                return session
            else:
                # Create new session for Claude-Code requests
                if session_info['is_claude_code']:
                    from ..memory_system.api.schemas import SessionCreate
                    session_data = SessionCreate(
                        user_id="claude-code",
                        project_id=None,
                        session_metadata={
                            'user_agent': session_info['user_agent'],
                            'is_claude_code': True,
                            'remote_addr': session_info.get('remote_addr'),
                            'started_via': 'middleware_auto'
                        }
                    )
                    session = await session_manager.create_session(session_data)
                    logger.info(f"Auto-created session for Claude-Code: {session.id}")
                    return session
                    
        except Exception as e:
            logger.error(f"Session management failed: {e}")
            
        return None
    
    def _generate_session_id_from_client(self, session_info: Dict[str, Any]) -> Optional[str]:
        """Generate deterministic session ID from client info"""
        if not session_info.get('is_claude_code'):
            return None
            
        # Create session ID based on client characteristics
        client_id = session_info.get('client_id')
        remote_addr = session_info.get('remote_addr')
        
        if client_id:
            return f"claude-{client_id}"
        elif remote_addr:
            # Use IP-based session for same client
            return f"claude-ip-{remote_addr.replace('.', '-')}"
        else:
            return None
    
    async def _inject_context(self, request: Request, session: Any):
        """Inject relevant context into request"""
        try:
            from ..memory_system.services.context_service import build_context_for_request
            
            # Build context for this request
            context = await build_context_for_request(
                session_id=str(session.id),
                request_path=request.url.path,
                request_method=request.method,
                max_tokens=4000  # Reasonable limit for context injection
            )
            
            # Store context in request state for downstream use
            if not hasattr(request, 'state'):
                request.state = type('RequestState', (), {})()
            
            request.state.session = session
            request.state.memory_context = context
            request.state.session_id = str(session.id)
            
            logger.debug(f"Injected context for session {session.id}: {len(context.get('memories', []))} memories")
            
        except Exception as e:
            logger.warning(f"Context injection failed for session {session.id}: {e}")
    
    async def _update_session_after_request(self, session: Any, request: Request, response: Response, session_manager: Any):
        """Update session after successful request processing"""
        try:
            # Extract request/response data for memory
            request_data = {
                'method': request.method,
                'path': request.url.path,
                'query_params': dict(request.query_params),
                'status_code': response.status_code,
                'timestamp': time.time()
            }
            
            # Try to extract request body for memory (if reasonable size)
            try:
                if hasattr(request, '_body') and len(getattr(request, '_body', b'')) < 10000:
                    body = getattr(request, '_body', b'').decode('utf-8', errors='ignore')
                    if body:
                        request_data['body_preview'] = body[:500]  # First 500 chars
            except Exception:
                pass  # Skip body extraction if it fails
            
            # Update session activity
            await session_manager.update_session_activity(
                session_id=str(session.id),
                activity_data=request_data
            )
            
        except Exception as e:
            logger.warning(f"Session update failed for {session.id}: {e}")
    
    async def _handle_session_error(self, session: Any, request: Request, error: Exception, session_manager: Any):
        """Handle session when request fails"""
        try:
            error_data = {
                'method': request.method,
                'path': request.url.path,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'timestamp': time.time()
            }
            
            # Log error in session metadata
            await session_manager.update_session_metadata(
                session_id=str(session.id),
                metadata_update={'last_error': error_data}
            )
            
        except Exception as meta_error:
            logger.error(f"Failed to record session error: {meta_error}")
    
    def _add_session_headers(self, response: Response, session: Any):
        """Add session-related headers to response"""
        try:
            response.headers['X-Session-ID'] = str(session.id)
            response.headers['X-Session-Active'] = 'true'
            
            # Add context information if available
            if hasattr(session, 'memory_count'):
                response.headers['X-Memory-Count'] = str(session.memory_count)
            
        except Exception as e:
            logger.warning(f"Failed to add session headers: {e}")


class ContextInjectionError(Exception):
    """Exception raised when context injection fails"""
    pass


async def get_current_session(request: Request) -> Optional[Any]:
    """Helper function to get current session from request"""
    if hasattr(request, 'state') and hasattr(request.state, 'session'):
        return request.state.session
    return None


async def get_memory_context(request: Request) -> Dict[str, Any]:
    """Helper function to get memory context from request"""
    if hasattr(request, 'state') and hasattr(request.state, 'memory_context'):
        return request.state.memory_context
    return {}


async def get_session_id(request: Request) -> Optional[str]:
    """Helper function to get session ID from request"""
    if hasattr(request, 'state') and hasattr(request.state, 'session_id'):
        return request.state.session_id
    return None