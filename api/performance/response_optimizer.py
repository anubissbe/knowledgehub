"""
Response Optimization System

Provides response optimization including:
- Response compression
- Response caching with ETags
- Streaming responses for large data
- Response time monitoring
- Content optimization
"""

import asyncio
import gzip
import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import logging

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Supported compression types"""
    NONE = "none"
    GZIP = "gzip"
    DEFLATE = "deflate"
    BROTLI = "brotli"


class ResponseFormat(Enum):
    """Response format types"""
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    STREAM = "stream"


@dataclass
class ResponseMetrics:
    """Response performance metrics"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    response_size: int
    compression_ratio: Optional[float]
    cache_hit: bool
    timestamp: float


class ResponseCompressor:
    """Response compression handler"""
    
    def __init__(self, min_size: int = 1024, compression_level: int = 6):
        self.min_size = min_size
        self.compression_level = compression_level
        
    def should_compress(self, content: bytes, content_type: str) -> bool:
        """Determine if content should be compressed"""
        
        # Size check
        if len(content) < self.min_size:
            return False
        
        # Content type check
        compressible_types = [
            'application/json',
            'application/xml',
            'text/html',
            'text/plain',
            'text/css',
            'text/javascript',
            'application/javascript'
        ]
        
        return any(ct in content_type for ct in compressible_types)
    
    def get_accepted_encoding(self, request: Request) -> CompressionType:
        """Get the best accepted compression encoding"""
        accept_encoding = request.headers.get('accept-encoding', '').lower()
        
        if 'gzip' in accept_encoding:
            return CompressionType.GZIP
        elif 'deflate' in accept_encoding:
            return CompressionType.DEFLATE
        else:
            return CompressionType.NONE
    
    def compress_content(self, content: bytes, compression_type: CompressionType) -> bytes:
        """Compress content using specified algorithm"""
        
        if compression_type == CompressionType.GZIP:
            return gzip.compress(content, compresslevel=self.compression_level)
        elif compression_type == CompressionType.DEFLATE:
            # Using zlib for deflate
            import zlib
            return zlib.compress(content, level=self.compression_level)
        else:
            return content
    
    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio"""
        if original_size == 0:
            return 0.0
        return (original_size - compressed_size) / original_size


class ResponseCache:
    """Response caching with ETag support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        
    def generate_etag(self, content: Union[str, bytes, dict]) -> str:
        """Generate ETag for content"""
        if isinstance(content, dict):
            content_str = json.dumps(content, sort_keys=True, default=str)
        elif isinstance(content, str):
            content_str = content
        else:
            content_str = str(content)
        
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def get_cache_key(self, request: Request) -> str:
        """Generate cache key from request"""
        key_parts = [
            request.method,
            str(request.url.path),
            str(sorted(request.query_params.items())),
            request.headers.get('accept', ''),
            request.headers.get('accept-language', '')
        ]
        
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, cache_key: str, if_none_match: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        
        if cache_key not in self.cache:
            return None
        
        entry = self.cache[cache_key]
        
        # Check expiration
        if time.time() - entry['timestamp'] > entry['ttl']:
            del self.cache[cache_key]
            if cache_key in self.access_times:
                del self.access_times[cache_key]
            return None
        
        # Check ETag
        if if_none_match and entry['etag'] == if_none_match:
            return {'status': 'not_modified', 'etag': entry['etag']}
        
        # Update access time
        self.access_times[cache_key] = time.time()
        
        return entry
    
    def set(self, cache_key: str, content: Any, etag: str, ttl: Optional[int] = None):
        """Cache response"""
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[cache_key] = {
            'content': content,
            'etag': etag,
            'timestamp': time.time(),
            'ttl': ttl or self.default_ttl
        }
        self.access_times[cache_key] = time.time()
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times, key=self.access_times.get)
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'utilization': len(self.cache) / self.max_size
        }


class StreamingOptimizer:
    """Streaming response optimizer"""
    
    def __init__(self, chunk_size: int = 8192):
        self.chunk_size = chunk_size
    
    async def stream_json_array(self, items: AsyncGenerator[Any, None]) -> AsyncGenerator[str, None]:
        """Stream JSON array efficiently"""
        yield "["
        first = True
        
        async for item in items:
            if not first:
                yield ","
            else:
                first = False
            
            yield json.dumps(item, default=str)
        
        yield "]"
    
    async def stream_csv(self, items: AsyncGenerator[Dict[str, Any], None], 
                        headers: List[str]) -> AsyncGenerator[str, None]:
        """Stream CSV data efficiently"""
        
        # Yield headers
        yield ",".join(headers) + "\n"
        
        # Yield data rows
        async for item in items:
            row = []
            for header in headers:
                value = item.get(header, "")
                # Simple CSV escaping
                if isinstance(value, str) and ("," in value or '"' in value):
                    escaped_value = value.replace('"', '""')
                    value = f'"{escaped_value}"'
                row.append(str(value))
            
            yield ",".join(row) + "\n"
    
    async def stream_large_json(self, data: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
        """Stream large JSON responses in chunks"""
        json_str = json.dumps(data, default=str)
        
        for i in range(0, len(json_str), self.chunk_size):
            chunk = json_str[i:i + self.chunk_size]
            yield chunk.encode('utf-8')


class ResponseOptimizer:
    """Main response optimization system"""
    
    def __init__(self):
        self.compressor = ResponseCompressor()
        self.cache = ResponseCache()
        self.streaming = StreamingOptimizer()
        self.metrics: List[ResponseMetrics] = []
        self.max_metrics = 10000
        
    def should_cache_response(self, request: Request, response_data: Any) -> bool:
        """Determine if response should be cached"""
        
        # Only cache GET requests
        if request.method != "GET":
            return False
        
        # Don't cache responses with query parameters that indicate dynamic content
        dynamic_params = ['timestamp', 'random', 'nonce']
        if any(param in request.query_params for param in dynamic_params):
            return False
        
        # Don't cache very large responses
        if isinstance(response_data, (dict, list)):
            try:
                size = len(json.dumps(response_data, default=str))
                if size > 1024 * 1024:  # 1MB
                    return False
            except:
                pass
        
        return True
    
    async def optimize_response(self, 
                              request: Request, 
                              response_data: Any,
                              status_code: int = 200,
                              headers: Optional[Dict[str, str]] = None) -> Response:
        """Optimize response with caching, compression, and metrics"""
        
        start_time = time.time()
        cache_hit = False
        compression_ratio = None
        
        if headers is None:
            headers = {}
        
        # Try cache first
        if self.should_cache_response(request, response_data):
            cache_key = self.cache.get_cache_key(request)
            if_none_match = request.headers.get('if-none-match')
            
            cached_response = self.cache.get(cache_key, if_none_match)
            if cached_response:
                cache_hit = True
                
                if cached_response.get('status') == 'not_modified':
                    # Return 304 Not Modified
                    return Response(
                        status_code=304,
                        headers={'etag': cached_response['etag']}
                    )
                else:
                    # Return cached content
                    response_data = cached_response['content']
                    headers['etag'] = cached_response['etag']
        
        # Serialize response data
        if isinstance(response_data, (dict, list)):
            content = json.dumps(response_data, default=str).encode('utf-8')
            content_type = "application/json"
        elif isinstance(response_data, str):
            content = response_data.encode('utf-8')
            content_type = "text/plain"
        elif isinstance(response_data, bytes):
            content = response_data
            content_type = "application/octet-stream"
        else:
            content = str(response_data).encode('utf-8')
            content_type = "text/plain"
        
        original_size = len(content)
        
        # Generate ETag if not cached
        if not cache_hit:
            etag = self.cache.generate_etag(content)
            headers['etag'] = etag
            
            # Cache the response
            if self.should_cache_response(request, response_data):
                self.cache.set(self.cache.get_cache_key(request), response_data, etag)
        
        # Apply compression
        compression_type = self.compressor.get_accepted_encoding(request)
        if self.compressor.should_compress(content, content_type) and compression_type != CompressionType.NONE:
            compressed_content = self.compressor.compress_content(content, compression_type)
            
            if len(compressed_content) < len(content):
                content = compressed_content
                headers['content-encoding'] = compression_type.value
                compression_ratio = self.compressor.get_compression_ratio(original_size, len(content))
        
        # Set additional headers
        headers['content-length'] = str(len(content))
        headers['x-response-time'] = f"{(time.time() - start_time) * 1000:.2f}ms"
        headers['x-cache-status'] = 'hit' if cache_hit else 'miss'
        
        if compression_ratio:
            headers['x-compression-ratio'] = f"{compression_ratio:.2f}"
        
        # Record metrics
        self._record_metrics(
            request=request,
            status_code=status_code,
            response_time=time.time() - start_time,
            response_size=len(content),
            compression_ratio=compression_ratio,
            cache_hit=cache_hit
        )
        
        return Response(
            content=content,
            status_code=status_code,
            headers=headers,
            media_type=content_type
        )
    
    async def create_streaming_response(self,
                                      generator: AsyncGenerator[Any, None],
                                      format_type: ResponseFormat = ResponseFormat.JSON,
                                      headers: Optional[Dict[str, str]] = None) -> StreamingResponse:
        """Create optimized streaming response"""
        
        if headers is None:
            headers = {}
        
        if format_type == ResponseFormat.JSON:
            media_type = "application/json"
            content_generator = self.streaming.stream_json_array(generator)
        elif format_type == ResponseFormat.CSV:
            media_type = "text/csv"
            # Note: For CSV, you'd need to pass headers separately
            content_generator = generator
        else:
            media_type = "text/plain"
            content_generator = generator
        
        headers['x-content-type'] = 'streaming'
        
        return StreamingResponse(
            content_generator,
            media_type=media_type,
            headers=headers
        )
    
    def _record_metrics(self,
                       request: Request,
                       status_code: int,
                       response_time: float,
                       response_size: int,
                       compression_ratio: Optional[float],
                       cache_hit: bool):
        """Record response metrics"""
        
        metrics = ResponseMetrics(
            endpoint=str(request.url.path),
            method=request.method,
            status_code=status_code,
            response_time=response_time,
            response_size=response_size,
            compression_ratio=compression_ratio,
            cache_hit=cache_hit,
            timestamp=time.time()
        )
        
        self.metrics.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
    
    def get_performance_stats(self, 
                            endpoint: Optional[str] = None,
                            hours: int = 24) -> Dict[str, Any]:
        """Get response performance statistics"""
        
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        
        if endpoint:
            recent_metrics = [m for m in recent_metrics if m.endpoint == endpoint]
        
        if not recent_metrics:
            return {'message': 'No metrics available for the specified criteria'}
        
        # Calculate statistics
        response_times = [m.response_time for m in recent_metrics]
        response_sizes = [m.response_size for m in recent_metrics]
        compression_ratios = [m.compression_ratio for m in recent_metrics if m.compression_ratio is not None]
        
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        total_requests = len(recent_metrics)
        
        # Status code breakdown
        status_codes = {}
        for m in recent_metrics:
            status_codes[m.status_code] = status_codes.get(m.status_code, 0) + 1
        
        # Endpoint breakdown
        endpoint_stats = {}
        for m in recent_metrics:
            if m.endpoint not in endpoint_stats:
                endpoint_stats[m.endpoint] = {'count': 0, 'total_time': 0, 'avg_size': 0}
            
            endpoint_stats[m.endpoint]['count'] += 1
            endpoint_stats[m.endpoint]['total_time'] += m.response_time
            endpoint_stats[m.endpoint]['avg_size'] += m.response_size
        
        # Calculate averages for endpoints
        for endpoint_name, stats in endpoint_stats.items():
            stats['avg_time'] = stats['total_time'] / stats['count']
            stats['avg_size'] = stats['avg_size'] / stats['count']
        
        return {
            'summary': {
                'total_requests': total_requests,
                'cache_hit_rate': cache_hits / total_requests if total_requests > 0 else 0,
                'average_response_time': sum(response_times) / len(response_times),
                'median_response_time': sorted(response_times)[len(response_times) // 2],
                'p95_response_time': sorted(response_times)[int(len(response_times) * 0.95)],
                'average_response_size': sum(response_sizes) / len(response_sizes),
                'average_compression_ratio': sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0,
            },
            'status_codes': status_codes,
            'endpoints': endpoint_stats,
            'cache_stats': self.cache.get_stats(),
            'time_period': f'Last {hours} hours'
        }
    
    def get_slow_endpoints(self, threshold: float = 1.0, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest endpoints"""
        
        endpoint_times = {}
        for metric in self.metrics:
            if metric.endpoint not in endpoint_times:
                endpoint_times[metric.endpoint] = []
            endpoint_times[metric.endpoint].append(metric.response_time)
        
        slow_endpoints = []
        for endpoint, times in endpoint_times.items():
            avg_time = sum(times) / len(times)
            if avg_time > threshold:
                slow_endpoints.append({
                    'endpoint': endpoint,
                    'average_time': avg_time,
                    'request_count': len(times),
                    'max_time': max(times),
                    'min_time': min(times)
                })
        
        # Sort by average time descending
        slow_endpoints.sort(key=lambda x: x['average_time'], reverse=True)
        
        return slow_endpoints[:limit]


# Global response optimizer instance
response_optimizer: Optional[ResponseOptimizer] = None


def get_response_optimizer() -> ResponseOptimizer:
    """Get or create global response optimizer"""
    global response_optimizer
    if response_optimizer is None:
        response_optimizer = ResponseOptimizer()
    return response_optimizer