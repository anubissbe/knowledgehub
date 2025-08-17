"""
Consolidated Memory Services Module
Combines memory-related services for better performance and maintainability.
"""

from api.shared import *
from enum import Enum

class MemoryType(str, Enum):
    """Types of memory operations"""
    SESSION = "session"
    PERSISTENT = "persistent"
    CACHE = "cache"
    HYBRID = "hybrid"

class BaseMemoryService(ABC):
    """Base memory service interface"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def store(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Store data"""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data"""
        pass

class ConsolidatedMemoryService:
    """Unified memory service combining all memory operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._services: Dict[MemoryType, BaseMemoryService] = {}
        
    def register_service(self, memory_type: MemoryType, service: BaseMemoryService):
        """Register a memory service"""
        self._services[memory_type] = service
        self.logger.info(f"Registered {memory_type} memory service")
    
    async def store(self, memory_type: MemoryType, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Store data in appropriate memory service"""
        try:
            service = self._services.get(memory_type)
            if not service:
                raise ValueError(f"No service registered for {memory_type}")
            
            result = await service.store(key, data, ttl)
            self.logger.debug(f"Stored data for key: {key} in {memory_type}")
            return result
            
        except Exception as e:
            self.logger.error(f"Store operation failed: {e}")
            raise ServiceException(f"Store operation failed: {e}")
    
    async def retrieve(self, memory_type: MemoryType, key: str) -> Optional[Any]:
        """Retrieve data from appropriate memory service"""
        try:
            service = self._services.get(memory_type)
            if not service:
                raise ValueError(f"No service registered for {memory_type}")
            
            result = await service.retrieve(key)
            self.logger.debug(f"Retrieved data for key: {key} from {memory_type}")
            return result
            
        except Exception as e:
            self.logger.error(f"Retrieve operation failed: {e}")
            raise ServiceException(f"Retrieve operation failed: {e}")
    
    async def delete(self, memory_type: MemoryType, key: str) -> bool:
        """Delete data from appropriate memory service"""
        try:
            service = self._services.get(memory_type)
            if not service:
                raise ValueError(f"No service registered for {memory_type}")
            
            result = await service.delete(key)
            self.logger.debug(f"Deleted data for key: {key} from {memory_type}")
            return result
            
        except Exception as e:
            self.logger.error(f"Delete operation failed: {e}")
            raise ServiceException(f"Delete operation failed: {e}")
    
    async def sync_memories(self, source_type: MemoryType, target_type: MemoryType, key: str) -> bool:
        """Sync memory between services"""
        try:
            data = await self.retrieve(source_type, key)
            if data is not None:
                return await self.store(target_type, key, data)
            return False
            
        except Exception as e:
            self.logger.error(f"Memory sync failed: {e}")
            raise ServiceException(f"Memory sync failed: {e}")
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all memory services"""
        health_status = {}
        for memory_type, service in self._services.items():
            try:
                # Simple health check by attempting a test operation
                test_key = f"health_check_{memory_type}"
                await service.store(test_key, "test", ttl=1)
                await service.delete(test_key)
                health_status[memory_type] = True
            except Exception as e:
                self.logger.error(f"Health check failed for {memory_type}: {e}")
                health_status[memory_type] = False
        
        return health_status

# Global memory service instance
memory_service = ConsolidatedMemoryService()
