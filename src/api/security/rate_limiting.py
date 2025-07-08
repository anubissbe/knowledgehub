"""
Advanced Rate Limiting and DDoS Protection

Implements multiple rate limiting strategies and DDoS protection
mechanisms to ensure API availability and prevent abuse.
"""

import time
import asyncio
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

import redis.asyncio as redis
from starlette.requests import Request
from starlette.responses import JSONResponse
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"


class ThreatLevel(str, Enum):
    """DDoS threat assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    enable_adaptive: bool = True
    enable_ddos_protection: bool = True
    blacklist_threshold: int = 1000
    blacklist_duration: int = 3600  # 1 hour


@dataclass
class ClientMetrics:
    """Client request metrics"""
    requests_last_minute: int = 0
    requests_last_hour: int = 0
    requests_last_day: int = 0
    burst_requests: int = 0
    suspicious_patterns: int = 0
    threat_score: float = 0.0
    first_seen: datetime = None
    last_request: datetime = None
    user_agent_changes: int = 0
    blocked_requests: int = 0


class AdvancedRateLimiter:
    """Advanced rate limiting and DDoS protection system"""
    
    def __init__(self, config: RateLimitConfig, redis_client: Optional[redis.Redis] = None):
        self.config = config
        self.redis_client = redis_client
        
        # In-memory fallback storage
        self.memory_storage: Dict[str, ClientMetrics] = {}
        self.request_timestamps: Dict[str, List[float]] = defaultdict(list)
        self.blacklisted_ips: Dict[str, datetime] = {}
        self.suspicious_patterns: Dict[str, int] = defaultdict(int)
        
        # Token bucket storage (for token bucket algorithm)
        self.token_buckets: Dict[str, Tuple[float, float]] = {}  # (tokens, last_refill)
        
        # Adaptive rate limiting
        self.global_request_rate: float = 0.0
        self.adaptive_multiplier: float = 1.0
        
        # Cleanup interval
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        
        logger.info(f"Advanced rate limiter initialized with strategy: {config.strategy}")
    
    def _get_client_id(self, request: Request) -> str:
        """Generate unique client identifier"""
        # Try to get real IP from headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            real_ip = request.headers.get("X-Real-IP")
            if real_ip:
                client_ip = real_ip.strip()
            else:
                client_ip = request.client.host if request.client else "unknown"
        
        # Include User-Agent for additional fingerprinting
        user_agent = request.headers.get("User-Agent", "")
        
        # Create composite identifier
        identifier_data = f"{client_ip}:{user_agent[:100]}"
        return hashlib.sha256(identifier_data.encode()).hexdigest()[:16]
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        return request.client.host if request.client else "unknown"
    
    async def _get_client_metrics(self, client_id: str) -> ClientMetrics:
        """Get client metrics from storage"""
        if self.redis_client:
            try:
                # Try to get from Redis
                metrics_data = await self.redis_client.hgetall(f"rate_limit:metrics:{client_id}")
                if metrics_data:
                    return ClientMetrics(
                        requests_last_minute=int(metrics_data.get(b"requests_last_minute", 0)),
                        requests_last_hour=int(metrics_data.get(b"requests_last_hour", 0)),
                        requests_last_day=int(metrics_data.get(b"requests_last_day", 0)),
                        burst_requests=int(metrics_data.get(b"burst_requests", 0)),
                        suspicious_patterns=int(metrics_data.get(b"suspicious_patterns", 0)),
                        threat_score=float(metrics_data.get(b"threat_score", 0.0)),
                        first_seen=datetime.fromisoformat(metrics_data.get(b"first_seen", datetime.now().isoformat()).decode()),
                        last_request=datetime.fromisoformat(metrics_data.get(b"last_request", datetime.now().isoformat()).decode()),
                        user_agent_changes=int(metrics_data.get(b"user_agent_changes", 0)),
                        blocked_requests=int(metrics_data.get(b"blocked_requests", 0))
                    )
            except Exception as e:
                logger.warning(f"Redis error getting metrics: {e}")
        
        # Fallback to memory storage
        return self.memory_storage.get(client_id, ClientMetrics(
            first_seen=datetime.now(),
            last_request=datetime.now()
        ))
    
    async def _update_client_metrics(self, client_id: str, metrics: ClientMetrics):
        """Update client metrics in storage"""
        metrics.last_request = datetime.now()
        
        if self.redis_client:
            try:
                # Store in Redis with expiration
                metrics_data = {
                    "requests_last_minute": metrics.requests_last_minute,
                    "requests_last_hour": metrics.requests_last_hour,
                    "requests_last_day": metrics.requests_last_day,
                    "burst_requests": metrics.burst_requests,
                    "suspicious_patterns": metrics.suspicious_patterns,
                    "threat_score": metrics.threat_score,
                    "first_seen": metrics.first_seen.isoformat(),
                    "last_request": metrics.last_request.isoformat(),
                    "user_agent_changes": metrics.user_agent_changes,
                    "blocked_requests": metrics.blocked_requests
                }
                await self.redis_client.hset(f"rate_limit:metrics:{client_id}", mapping=metrics_data)
                await self.redis_client.expire(f"rate_limit:metrics:{client_id}", 86400)  # 24 hours
            except Exception as e:
                logger.warning(f"Redis error updating metrics: {e}")
        
        # Always update memory storage as fallback
        self.memory_storage[client_id] = metrics
    
    def _sliding_window_check(self, client_id: str, window_seconds: int, limit: int) -> bool:
        """Check sliding window rate limit"""
        now = time.time()
        window_start = now - window_seconds
        
        # Get timestamps for this client
        timestamps = self.request_timestamps[client_id]
        
        # Remove old timestamps
        timestamps[:] = [ts for ts in timestamps if ts > window_start]
        
        # Check if limit exceeded
        return len(timestamps) >= limit
    
    def _token_bucket_check(self, client_id: str, rate: float, capacity: int) -> bool:
        """Check token bucket rate limit"""
        now = time.time()
        
        if client_id not in self.token_buckets:
            # Initialize bucket with full capacity
            self.token_buckets[client_id] = (capacity, now)
            return True
        
        tokens, last_refill = self.token_buckets[client_id]
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = now - last_refill
        tokens_to_add = time_elapsed * rate
        tokens = min(capacity, tokens + tokens_to_add)
        
        # Check if we have tokens available
        if tokens >= 1.0:
            # Consume one token
            self.token_buckets[client_id] = (tokens - 1.0, now)
            return True
        else:
            # No tokens available
            self.token_buckets[client_id] = (tokens, now)
            return False
    
    def _assess_threat_level(self, metrics: ClientMetrics, request: Request) -> ThreatLevel:
        """Assess DDoS threat level based on client behavior"""
        threat_score = 0.0
        
        # High request rate
        if metrics.requests_last_minute > self.config.requests_per_minute * 2:
            threat_score += 3.0
        elif metrics.requests_last_minute > self.config.requests_per_minute:
            threat_score += 1.0
        
        # Burst behavior
        if metrics.burst_requests > self.config.burst_limit * 2:
            threat_score += 2.0
        elif metrics.burst_requests > self.config.burst_limit:
            threat_score += 1.0
        
        # Suspicious patterns
        if metrics.suspicious_patterns > 5:
            threat_score += 2.0
        elif metrics.suspicious_patterns > 2:
            threat_score += 1.0
        
        # User agent changes (potential bot behavior)
        if metrics.user_agent_changes > 10:
            threat_score += 1.5
        elif metrics.user_agent_changes > 5:
            threat_score += 0.5
        
        # Previous blocked requests
        if metrics.blocked_requests > 50:
            threat_score += 2.0
        elif metrics.blocked_requests > 10:
            threat_score += 1.0
        
        # Check for suspicious headers or patterns
        user_agent = request.headers.get("User-Agent", "").lower()
        suspicious_agents = ["bot", "crawler", "spider", "scraper", "wget", "curl"]
        if any(agent in user_agent for agent in suspicious_agents):
            threat_score += 0.5
        
        # Missing common headers
        if not request.headers.get("Accept"):
            threat_score += 0.3
        if not request.headers.get("Accept-Language"):
            threat_score += 0.3
        
        # Update threat score in metrics
        metrics.threat_score = threat_score
        
        # Determine threat level
        if threat_score >= 6.0:
            return ThreatLevel.CRITICAL
        elif threat_score >= 4.0:
            return ThreatLevel.HIGH
        elif threat_score >= 2.0:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _detect_suspicious_patterns(self, request: Request, metrics: ClientMetrics):
        """Detect suspicious request patterns"""
        # Check for rapid-fire requests (more than 10 per second)
        now = datetime.now()
        if metrics.last_request and (now - metrics.last_request).total_seconds() < 0.1:
            metrics.suspicious_patterns += 1
        
        # Check for missing or unusual headers
        user_agent = request.headers.get("User-Agent", "")
        if not user_agent or len(user_agent) < 10:
            metrics.suspicious_patterns += 1
        
        # Check for automated tool signatures
        automated_signatures = [
            "python-requests", "http.rb", "node-fetch", "axios",
            "curl", "wget", "httpie", "postman"
        ]
        if any(sig in user_agent.lower() for sig in automated_signatures):
            metrics.suspicious_patterns += 1
        
        # Check for SQL injection attempts in query parameters
        query_string = str(request.url.query).lower()
        sql_patterns = ["union", "select", "insert", "delete", "drop", "script"]
        if any(pattern in query_string for pattern in sql_patterns):
            metrics.suspicious_patterns += 2
    
    async def _update_adaptive_limits(self):
        """Update adaptive rate limits based on global load"""
        try:
            # Calculate global request rate
            total_requests = sum(
                len(timestamps) for timestamps in self.request_timestamps.values()
            )
            
            self.global_request_rate = total_requests / 60.0  # requests per second
            
            # Adjust adaptive multiplier based on load
            if self.global_request_rate > 1000:  # High load
                self.adaptive_multiplier = 0.5
            elif self.global_request_rate > 500:  # Medium load
                self.adaptive_multiplier = 0.75
            else:  # Normal load
                self.adaptive_multiplier = 1.0
            
            logger.debug(f"Adaptive multiplier updated to {self.adaptive_multiplier}")
            
        except Exception as e:
            logger.error(f"Error updating adaptive limits: {e}")
    
    async def check_rate_limit(self, request: Request) -> Tuple[bool, Optional[JSONResponse]]:
        """
        Check if request should be rate limited
        
        Returns:
            (allowed, error_response): Tuple indicating if request is allowed
        """
        client_id = self._get_client_id(request)
        client_ip = self._get_client_ip(request)
        now = time.time()
        
        # Check if IP is blacklisted
        if client_ip in self.blacklisted_ips:
            if datetime.now() < self.blacklisted_ips[client_ip]:
                return False, JSONResponse(
                    status_code=429,
                    content={
                        "error": "IP_BLACKLISTED",
                        "message": "Your IP has been temporarily blacklisted due to suspicious activity",
                        "code": "BLACKLISTED"
                    }
                )
            else:
                # Remove expired blacklist entry
                del self.blacklisted_ips[client_ip]
        
        # Get client metrics
        metrics = await self._get_client_metrics(client_id)
        
        # Detect suspicious patterns
        self._detect_suspicious_patterns(request, metrics)
        
        # Assess threat level
        threat_level = self._assess_threat_level(metrics, request)
        
        # Apply strategy-specific rate limiting
        is_limited = False
        
        if self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            # Check multiple time windows
            is_limited = (
                self._sliding_window_check(client_id, 60, int(self.config.requests_per_minute * self.adaptive_multiplier)) or
                self._sliding_window_check(client_id, 3600, int(self.config.requests_per_hour * self.adaptive_multiplier)) or
                self._sliding_window_check(client_id, 86400, int(self.config.requests_per_day * self.adaptive_multiplier))
            )
        
        elif self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            # Token bucket algorithm
            rate = self.config.requests_per_minute / 60.0 * self.adaptive_multiplier
            is_limited = not self._token_bucket_check(client_id, rate, self.config.burst_limit)
        
        elif self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
            # Simple fixed window
            current_minute = int(now / 60)
            key = f"{client_id}:{current_minute}"
            
            if self.redis_client:
                try:
                    count = await self.redis_client.incr(key)
                    if count == 1:
                        await self.redis_client.expire(key, 60)
                    is_limited = count > self.config.requests_per_minute * self.adaptive_multiplier
                except:
                    # Fallback to sliding window
                    is_limited = self._sliding_window_check(client_id, 60, int(self.config.requests_per_minute * self.adaptive_multiplier))
            else:
                is_limited = self._sliding_window_check(client_id, 60, int(self.config.requests_per_minute * self.adaptive_multiplier))
        
        # Apply threat-based restrictions
        if threat_level == ThreatLevel.CRITICAL:
            is_limited = True
            # Add to blacklist
            self.blacklisted_ips[client_ip] = datetime.now() + timedelta(seconds=self.config.blacklist_duration)
            logger.warning(f"Blacklisted IP {client_ip} due to critical threat level")
        
        elif threat_level == ThreatLevel.HIGH:
            # Reduce limits significantly
            current_requests = len([ts for ts in self.request_timestamps[client_id] if ts > now - 60])
            if current_requests > self.config.requests_per_minute * 0.1:  # 10% of normal limit
                is_limited = True
        
        elif threat_level == ThreatLevel.MEDIUM:
            # Reduce limits moderately
            current_requests = len([ts for ts in self.request_timestamps[client_id] if ts > now - 60])
            if current_requests > self.config.requests_per_minute * 0.5:  # 50% of normal limit
                is_limited = True
        
        # Update metrics
        if is_limited:
            metrics.blocked_requests += 1
        else:
            # Record successful request
            self.request_timestamps[client_id].append(now)
            metrics.requests_last_minute = len([ts for ts in self.request_timestamps[client_id] if ts > now - 60])
            metrics.requests_last_hour = len([ts for ts in self.request_timestamps[client_id] if ts > now - 3600])
            metrics.requests_last_day = len([ts for ts in self.request_timestamps[client_id] if ts > now - 86400])
        
        # Save updated metrics
        await self._update_client_metrics(client_id, metrics)
        
        # Update adaptive limits periodically
        if self.config.enable_adaptive and now - self.last_cleanup > 60:
            await self._update_adaptive_limits()
            self.last_cleanup = now
        
        if is_limited:
            # Calculate retry after
            retry_after = 60  # Default
            if threat_level == ThreatLevel.CRITICAL:
                retry_after = self.config.blacklist_duration
            elif threat_level == ThreatLevel.HIGH:
                retry_after = 300  # 5 minutes
            elif threat_level == ThreatLevel.MEDIUM:
                retry_after = 120  # 2 minutes
            
            # Create rate limit response
            response = JSONResponse(
                status_code=429,
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Threat level: {threat_level.value}",
                    "threat_level": threat_level.value,
                    "retry_after": retry_after,
                    "requests_per_minute": int(self.config.requests_per_minute * self.adaptive_multiplier)
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(int(self.config.requests_per_minute * self.adaptive_multiplier)),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(now + retry_after)),
                    "X-Threat-Level": threat_level.value
                }
            )
            
            # Log security event
            from .monitoring import log_security_event, SecurityEventType, ThreatLevel as MonitoringThreatLevel
            await log_security_event(
                SecurityEventType.RATE_LIMIT_EXCEEDED,
                MonitoringThreatLevel.MEDIUM if threat_level == ThreatLevel.MEDIUM else MonitoringThreatLevel.HIGH,
                client_ip,
                request.headers.get("User-Agent", ""),
                str(request.url.path),
                request.method,
                f"Rate limit exceeded with threat level: {threat_level.value}"
            )
            
            return False, response
        
        return True, None
    
    async def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """Get statistics for a specific client"""
        metrics = await self._get_client_metrics(client_id)
        
        return {
            "client_id": client_id,
            "requests_last_minute": metrics.requests_last_minute,
            "requests_last_hour": metrics.requests_last_hour,
            "requests_last_day": metrics.requests_last_day,
            "threat_score": metrics.threat_score,
            "suspicious_patterns": metrics.suspicious_patterns,
            "blocked_requests": metrics.blocked_requests,
            "first_seen": metrics.first_seen.isoformat(),
            "last_request": metrics.last_request.isoformat()
        }
    
    async def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiting statistics"""
        total_clients = len(self.memory_storage)
        blacklisted_ips = len(self.blacklisted_ips)
        
        # Calculate average threat score
        avg_threat_score = 0.0
        if self.memory_storage:
            avg_threat_score = sum(m.threat_score for m in self.memory_storage.values()) / len(self.memory_storage)
        
        return {
            "total_clients": total_clients,
            "blacklisted_ips": blacklisted_ips,
            "global_request_rate": self.global_request_rate,
            "adaptive_multiplier": self.adaptive_multiplier,
            "average_threat_score": round(avg_threat_score, 2),
            "strategy": self.config.strategy.value,
            "limits": {
                "requests_per_minute": int(self.config.requests_per_minute * self.adaptive_multiplier),
                "requests_per_hour": int(self.config.requests_per_hour * self.adaptive_multiplier),
                "requests_per_day": int(self.config.requests_per_day * self.adaptive_multiplier)
            }
        }
    
    async def cleanup_expired_data(self):
        """Clean up expired data and timestamps"""
        now = time.time()
        
        # Clean up old timestamps
        for client_id in list(self.request_timestamps.keys()):
            # Keep only last 24 hours
            day_ago = now - 86400
            self.request_timestamps[client_id] = [
                ts for ts in self.request_timestamps[client_id] if ts > day_ago
            ]
            
            # Remove empty entries
            if not self.request_timestamps[client_id]:
                del self.request_timestamps[client_id]
        
        # Clean up expired blacklist entries
        expired_ips = [
            ip for ip, expiry in self.blacklisted_ips.items()
            if datetime.now() > expiry
        ]
        for ip in expired_ips:
            del self.blacklisted_ips[ip]
        
        # Clean up old metrics
        cutoff = datetime.now() - timedelta(hours=24)
        expired_clients = [
            client_id for client_id, metrics in self.memory_storage.items()
            if metrics.last_request < cutoff
        ]
        for client_id in expired_clients:
            del self.memory_storage[client_id]
        
        logger.debug(f"Cleaned up {len(expired_ips)} expired IPs and {len(expired_clients)} old clients")


# Global rate limiter instance
rate_limiter: Optional[AdvancedRateLimiter] = None


def get_rate_limiter() -> AdvancedRateLimiter:
    """Get global rate limiter instance"""
    global rate_limiter
    if rate_limiter is None:
        config = RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=2000,
            requests_per_day=20000,
            burst_limit=20,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            enable_adaptive=True,
            enable_ddos_protection=True
        )
        rate_limiter = AdvancedRateLimiter(config)
    return rate_limiter


async def check_rate_limit(request: Request) -> Tuple[bool, Optional[JSONResponse]]:
    """Check rate limit for request"""
    limiter = get_rate_limiter()
    return await limiter.check_rate_limit(request)


async def get_rate_limit_stats() -> Dict[str, Any]:
    """Get rate limiting statistics"""
    limiter = get_rate_limiter()
    return await limiter.get_global_stats()