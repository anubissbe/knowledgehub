"""Entity extraction API router"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...services.entity_extraction import (
    extract_entities_from_text,
    enrich_chunks_with_entities,
    EntityType,
    SPACY_AVAILABLE
)

logger = logging.getLogger(__name__)
router = APIRouter()


class EntityExtractionRequest(BaseModel):
    """Request for entity extraction"""
    text: str = Field(..., description="Text to extract entities from", min_length=1)
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class EntityResponse(BaseModel):
    """Response for a single entity"""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str
    metadata: Dict[str, Any]


class EntityExtractionResponse(BaseModel):
    """Response for entity extraction"""
    entities: List[EntityResponse]
    total_entities: int
    processing_time_ms: float
    statistics: Dict[str, Any]
    spacy_available: bool


class ChunkEnrichmentRequest(BaseModel):
    """Request for chunk enrichment with entities"""
    chunks: List[Dict[str, Any]] = Field(..., description="Text chunks to enrich")


@router.get("/health")
async def entity_extraction_health():
    """Health check for entity extraction service"""
    return {
        "status": "healthy",
        "service": "entity_extraction",
        "spacy_available": SPACY_AVAILABLE,
        "features": [
            "named_entity_recognition",
            "technology_detection",
            "code_entity_extraction",
            "custom_entities",
            "chunk_enrichment",
            "confidence_scoring"
        ],
        "supported_entity_types": [entity_type.value for entity_type in EntityType]
    }


@router.post("/extract", response_model=EntityExtractionResponse)
async def extract_entities(request: EntityExtractionRequest):
    """
    Extract entities from text.
    
    This endpoint extracts various types of entities including:
    - People and organizations
    - Technologies (programming languages, frameworks, databases, tools)
    - Code elements (functions, classes, variables)
    - URLs, emails, file paths
    - Custom entities
    """
    import time
    start_time = time.time()
    
    try:
        entities = await extract_entities_from_text(request.text, request.context)
        
        # Convert to response format
        entity_responses = []
        for entity in entities:
            entity_response = EntityResponse(
                text=entity.text,
                entity_type=entity.entity_type.value,
                start_pos=entity.start_pos,
                end_pos=entity.end_pos,
                confidence=entity.confidence,
                context=entity.context,
                metadata=entity.metadata
            )
            entity_responses.append(entity_response)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate statistics
        by_type = {}
        total_confidence = 0
        
        for entity in entities:
            entity_type = entity.entity_type.value
            by_type[entity_type] = by_type.get(entity_type, 0) + 1
            total_confidence += entity.confidence
        
        statistics = {
            "by_type": by_type,
            "avg_confidence": total_confidence / len(entities) if entities else 0,
            "unique_entities": len(set(entity.text.lower() for entity in entities)),
            "entity_types_found": len(by_type),
            "text_length": len(request.text)
        }
        
        return EntityExtractionResponse(
            entities=entity_responses,
            total_entities=len(entities),
            processing_time_ms=processing_time,
            statistics=statistics,
            spacy_available=SPACY_AVAILABLE
        )
        
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")


@router.post("/enrich-chunks")
async def enrich_chunks(request: ChunkEnrichmentRequest):
    """
    Enrich text chunks with extracted entities.
    
    This endpoint takes text chunks (like those from the text chunking service)
    and adds entity information to each chunk.
    """
    try:
        enriched_chunks = await enrich_chunks_with_entities(request.chunks)
        
        total_entities = sum(len(chunk.get('entities', [])) for chunk in enriched_chunks)
        
        logger.info(f"Enriched {len(enriched_chunks)} chunks with {total_entities} total entities")
        
        return {
            "enriched_chunks": enriched_chunks,
            "total_chunks": len(enriched_chunks),
            "total_entities": total_entities,
            "chunks_with_entities": sum(1 for chunk in enriched_chunks if chunk.get('entities'))
        }
        
    except Exception as e:
        logger.error(f"Chunk enrichment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chunk enrichment failed: {str(e)}")


@router.get("/entity-types")
async def get_entity_types():
    """Get available entity types and their descriptions"""
    entity_type_info = {
        EntityType.PERSON.value: "People and individuals",
        EntityType.ORGANIZATION.value: "Organizations and companies",
        EntityType.TECHNOLOGY.value: "General technology terms",
        EntityType.PROJECT.value: "Projects and applications",
        EntityType.PROGRAMMING_LANGUAGE.value: "Programming languages",
        EntityType.FRAMEWORK.value: "Software frameworks and libraries",
        EntityType.DATABASE.value: "Database systems",
        EntityType.TOOL.value: "Development tools and utilities",
        EntityType.CONCEPT.value: "General concepts and ideas",
        EntityType.LOCATION.value: "Places and locations",
        EntityType.DATE.value: "Dates and times",
        EntityType.URL.value: "Web URLs and links",
        EntityType.EMAIL.value: "Email addresses",
        EntityType.FILE_PATH.value: "File and directory paths",
        EntityType.FUNCTION_NAME.value: "Function and method names",
        EntityType.CLASS_NAME.value: "Class and type names",
        EntityType.VARIABLE_NAME.value: "Variable and constant names",
        EntityType.ERROR_TYPE.value: "Error and exception types"
    }
    
    return {
        "entity_types": entity_type_info,
        "total_types": len(entity_type_info),
        "spacy_available": SPACY_AVAILABLE
    }


@router.post("/analyze-entities")
async def analyze_entity_content(text: str):
    """
    Analyze text for entity content without full extraction.
    
    Returns quick analysis of what types of entities might be present.
    """
    try:
        import re
        
        # Quick pattern-based analysis
        analysis = {
            "text_length": len(text),
            "estimated_entities": 0,
            "likely_types": []
        }
        
        # Check for common patterns
        if re.search(r'https?://[^\s]+', text):
            analysis["likely_types"].append("url")
            analysis["estimated_entities"] += len(re.findall(r'https?://[^\s]+', text))
        
        if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text):
            analysis["likely_types"].append("email")
            analysis["estimated_entities"] += len(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text))
        
        # Technology terms (basic check)
        tech_terms = ['python', 'javascript', 'react', 'django', 'postgresql', 'redis', 'docker']
        found_tech = [term for term in tech_terms if term.lower() in text.lower()]
        if found_tech:
            analysis["likely_types"].append("technology")
            analysis["estimated_entities"] += len(found_tech)
        
        # Code patterns
        if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\(\)', text):
            analysis["likely_types"].append("function_name")
            analysis["estimated_entities"] += len(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\(\)', text))
        
        # File paths
        if re.search(r'/[a-zA-Z0-9._/-]+|[A-Z]:\\[a-zA-Z0-9._\\-]+', text):
            analysis["likely_types"].append("file_path")
            analysis["estimated_entities"] += len(re.findall(r'/[a-zA-Z0-9._/-]+|[A-Z]:\\[a-zA-Z0-9._\\-]+', text))
        
        analysis["entity_density"] = analysis["estimated_entities"] / len(text.split()) if text.split() else 0
        
        return analysis
        
    except Exception as e:
        logger.error(f"Entity analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Entity analysis failed: {str(e)}")


@router.post("/extract-from-chunks")
async def extract_entities_from_chunks(chunks: List[Dict[str, Any]]):
    """
    Extract entities from multiple text chunks efficiently.
    
    This is optimized for processing multiple chunks from the text chunking service.
    """
    try:
        results = []
        total_entities = 0
        
        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '')
            if not content:
                continue
            
            entities = await extract_entities_from_text(content)
            
            chunk_result = {
                "chunk_index": i,
                "entities": [{
                    "text": entity.text,
                    "type": entity.entity_type.value,
                    "confidence": entity.confidence,
                    "start_pos": entity.start_pos,
                    "end_pos": entity.end_pos
                } for entity in entities],
                "entity_count": len(entities),
                "chunk_metadata": chunk.get('metadata', {})
            }
            
            results.append(chunk_result)
            total_entities += len(entities)
        
        return {
            "results": results,
            "total_chunks": len(chunks),
            "total_entities": total_entities,
            "chunks_with_entities": sum(1 for r in results if r["entity_count"] > 0)
        }
        
    except Exception as e:
        logger.error(f"Chunk entity extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chunk entity extraction failed: {str(e)}")