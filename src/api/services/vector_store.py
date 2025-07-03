"""Vector store service using Weaviate"""

import weaviate
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ..config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Weaviate vector store service"""
    
    def __init__(self, url: str, collection_name: str):
        self.url = url
        self.collection_name = collection_name
        self.client: Optional[weaviate.Client] = None
    
    async def initialize(self):
        """Initialize Weaviate client and create schema"""
        try:
            # Create Weaviate client
            self.client = weaviate.Client(
                url=self.url,
                timeout_config=(5, 30)  # (connect timeout, read timeout)
            )
            
            # Check if client is ready
            if self.client is None or not self.client.is_ready():
                raise Exception("Weaviate is not ready")
            
            logger.info("Weaviate connection established")
            
            # Create collection schema if it doesn't exist
            await self._create_schema()
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}")
            raise
    
    async def _create_schema(self):
        """Create Weaviate collection schema"""
        try:
            # Check if collection already exists
            if self.client is None:
                raise Exception("Weaviate client not initialized")
            schema = self.client.schema.get()
            existing_classes = [cls["class"] for cls in schema.get("classes", [])]
            
            if self.collection_name in existing_classes:
                logger.info(f"Collection {self.collection_name} already exists")
                return
            
            # Also check for the hardcoded class name
            if "Knowledge_chunks" in existing_classes:
                logger.info("Collection Knowledge_chunks already exists, skipping creation")
                return
            
            # Define collection schema
            collection_schema = {
                "class": self.collection_name,
                "description": "Knowledge chunks for AI Knowledge Hub",
                "properties": [
                    {
                        "name": "doc_id",
                        "dataType": ["string"],
                        "description": "Document ID"
                    },
                    {
                        "name": "source_id",
                        "dataType": ["string"],
                        "description": "Source ID"
                    },
                    {
                        "name": "url",
                        "dataType": ["string"],
                        "description": "Document URL"
                    },
                    {
                        "name": "chunk_index",
                        "dataType": ["int"],
                        "description": "Chunk index within document"
                    },
                    {
                        "name": "chunk_type",
                        "dataType": ["string"],
                        "description": "Type of chunk (text, code, table, etc.)"
                    },
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Chunk content"
                    },
                    {
                        "name": "parent_heading",
                        "dataType": ["string"],
                        "description": "Parent heading for context"
                    },
                    {
                        "name": "position",
                        "dataType": ["int"],
                        "description": "Position in document"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["string"],
                        "description": "Additional metadata as JSON"
                    },
                    {
                        "name": "created_at",
                        "dataType": ["date"],
                        "description": "Creation timestamp"
                    }
                ],
                "vectorizer": "none"  # We'll provide our own vectors
            }
            
            # Create collection
            if self.client is None:
                raise Exception("Weaviate client not initialized")
            self.client.schema.create_class(collection_schema)
            logger.info(f"Created collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if Weaviate is healthy"""
        try:
            if self.client and self.client.is_ready():
                return True
            return False
        except:
            return False
    
    async def insert_chunk(self, chunk_data: Dict[str, Any], vector: List[float]) -> str:
        """Insert a single chunk with its vector"""
        try:
            if self.client is None:
                raise Exception("Weaviate client not initialized")
            result = self.client.data_object.create(
                data_object=chunk_data,
                class_name=self.collection_name,
                vector=vector
            )
            return result
        except Exception as e:
            logger.error(f"Failed to insert chunk: {e}")
            raise
    
    async def batch_insert(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Batch insert multiple chunks"""
        try:
            if self.client is None:
                raise Exception("Weaviate client not initialized")
            with self.client.batch as batch:
                batch.batch_size = 100
                ids = []
                
                for chunk in chunks:
                    vector = chunk.pop("vector", None)
                    properties = chunk.get("properties", chunk)
                    
                    # Ensure date format
                    if "created_at" not in properties:
                        properties["created_at"] = datetime.utcnow().isoformat()
                    
                    batch.add_data_object(
                        data_object=properties,
                        class_name=self.collection_name,
                        vector=vector
                    )
                    
                    if chunk.get("id"):
                        ids.append(chunk["id"])
                
                # Flush remaining
                batch.create_objects()
            
            return ids
            
        except Exception as e:
            logger.error(f"Failed to batch insert: {e}")
            raise
    
    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        try:
            if self.client is None:
                raise Exception("Weaviate client not initialized")
            query = (
                self.client.query
                .get(self.collection_name, ["chunk_id", "document_id", "source_id", "content", 
                                           "chunk_type", "metadata"])
                .with_near_vector({"vector": query_vector})
                .with_limit(limit)
                .with_additional(["distance", "id"])
            )
            
            # Add filters if provided
            if filters:
                where_filter = self._build_where_filter(filters)
                if where_filter:
                    query = query.with_where(where_filter)
            
            result = query.do()
            
            if "errors" in result:
                logger.error(f"Search errors: {result['errors']}")
                return []
            
            chunks = result.get("data", {}).get("Get", {}).get(self.collection_name, [])
            
            # Format results
            formatted_results = []
            for chunk in chunks:
                formatted_results.append({
                    "id": chunk.get("_additional", {}).get("id"),
                    "distance": chunk.get("_additional", {}).get("distance", 0),
                    "score": 1 - float(chunk.get("_additional", {}).get("distance", 0)),
                    **chunk
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _build_where_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build Weaviate where filter from filters dict"""
        where_conditions = []
        
        if "source_id" in filters:
            where_conditions.append({
                "path": ["source_id"],
                "operator": "Equal",
                "valueString": filters["source_id"]
            })
        
        if "chunk_type" in filters:
            where_conditions.append({
                "path": ["chunk_type"],
                "operator": "Equal",
                "valueString": filters["chunk_type"]
            })
        
        if not where_conditions:
            return None
        
        if len(where_conditions) == 1:
            return where_conditions[0]
        
        return {
            "operator": "And",
            "operands": where_conditions
        }
    
    async def delete_by_source(self, source_id: str):
        """Delete all chunks for a source"""
        try:
            if self.client is None:
                raise Exception("Weaviate client not initialized")
            self.client.batch.delete_objects(
                class_name=self.collection_name,
                where={
                    "path": ["source_id"],
                    "operator": "Equal",
                    "valueString": source_id
                }
            )
            logger.info(f"Deleted chunks for source: {source_id}")
        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            raise
    
    async def close(self):
        """Close Weaviate connection"""
        # Weaviate client doesn't need explicit closing
        pass


# Global vector store instance
vector_store = VectorStore(
    url=settings.WEAVIATE_URL,
    collection_name=settings.WEAVIATE_COLLECTION_NAME
)