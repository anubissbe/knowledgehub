"""Fact extraction API router"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...services.fact_extraction import (
    extract_facts_from_text,
    enrich_chunks_with_facts,
    FactType,
    ExtractedFact
)

logger = logging.getLogger(__name__)
router = APIRouter()


class FactExtractionRequest(BaseModel):
    """Request for fact extraction"""
    text: str = Field(..., description="Text to extract facts from", min_length=1)
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    entities: Optional[List[str]] = Field(None, description="Pre-extracted entities to associate with facts")


class FactResponse(BaseModel):
    """Response for a single fact"""
    content: str
    fact_type: str
    confidence: float
    certainty_level: float
    importance: float
    source_span: List[int]  # [start, end]
    context: str
    entities: List[str]
    temporal_info: Optional[str]
    metadata: Dict[str, Any]


class FactExtractionResponse(BaseModel):
    """Response for fact extraction"""
    facts: List[FactResponse]
    total_facts: int
    processing_time_ms: float
    statistics: Dict[str, Any]


class ChunkFactEnrichmentRequest(BaseModel):
    """Request for chunk enrichment with facts"""
    chunks: List[Dict[str, Any]] = Field(..., description="Text chunks to enrich with facts")


@router.get("/health")
async def fact_extraction_health():
    """Health check for fact extraction service"""
    return {
        "status": "healthy",
        "service": "fact_extraction",
        "features": [
            "decision_extraction",
            "preference_extraction", 
            "action_item_extraction",
            "error_solution_extraction",
            "configuration_extraction",
            "observation_extraction",
            "temporal_information",
            "entity_association",
            "confidence_scoring",
            "importance_scoring",
            "chunk_enrichment"
        ],
        "supported_fact_types": [fact_type.value for fact_type in FactType]
    }


@router.post("/extract", response_model=FactExtractionResponse)
async def extract_facts(request: FactExtractionRequest):
    """
    Extract facts from conversation text.
    
    This endpoint extracts various types of facts including:
    - Decisions that were made
    - User preferences and choices
    - Action items and tasks
    - Requirements and constraints
    - Configuration settings
    - Error facts and solutions
    - Observations and findings
    - General factual statements
    """
    import time
    start_time = time.time()
    
    try:
        facts = await extract_facts_from_text(
            request.text, 
            request.context, 
            request.entities
        )
        
        # Convert to response format
        fact_responses = []
        for fact in facts:
            fact_response = FactResponse(
                content=fact.content,
                fact_type=fact.fact_type.value,
                confidence=fact.confidence,
                certainty_level=fact.certainty_level,
                importance=fact.importance,
                source_span=[fact.source_span[0], fact.source_span[1]],
                context=fact.context,
                entities=fact.entities,
                temporal_info=fact.temporal_info,
                metadata=fact.metadata
            )
            fact_responses.append(fact_response)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate statistics
        by_type = {}
        total_confidence = 0
        total_certainty = 0
        total_importance = 0
        facts_with_entities = 0
        facts_with_temporal = 0
        
        for fact in facts:
            fact_type = fact.fact_type.value
            by_type[fact_type] = by_type.get(fact_type, 0) + 1
            total_confidence += fact.confidence
            total_certainty += fact.certainty_level
            total_importance += fact.importance
            
            if fact.entities:
                facts_with_entities += 1
            if fact.temporal_info:
                facts_with_temporal += 1
        
        statistics = {
            "by_type": by_type,
            "avg_confidence": total_confidence / len(facts) if facts else 0,
            "avg_certainty": total_certainty / len(facts) if facts else 0,
            "avg_importance": total_importance / len(facts) if facts else 0,
            "facts_with_entities": facts_with_entities,
            "facts_with_temporal_info": facts_with_temporal,
            "unique_fact_types": len(by_type),
            "text_length": len(request.text)
        }
        
        return FactExtractionResponse(
            facts=fact_responses,
            total_facts=len(facts),
            processing_time_ms=processing_time,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"Fact extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Fact extraction failed: {str(e)}")


@router.post("/enrich-chunks")
async def enrich_chunks(request: ChunkFactEnrichmentRequest):
    """
    Enrich text chunks with extracted facts.
    
    This endpoint takes text chunks (like those from the text chunking service)
    and adds fact information to each chunk.
    """
    try:
        enriched_chunks = await enrich_chunks_with_facts(request.chunks)
        
        total_facts = sum(len(chunk.get('facts', [])) for chunk in enriched_chunks)
        
        logger.info(f"Enriched {len(enriched_chunks)} chunks with {total_facts} total facts")
        
        return {
            "enriched_chunks": enriched_chunks,
            "total_chunks": len(enriched_chunks),
            "total_facts": total_facts,
            "chunks_with_facts": sum(1 for chunk in enriched_chunks if chunk.get('facts'))
        }
        
    except Exception as e:
        logger.error(f"Chunk fact enrichment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chunk fact enrichment failed: {str(e)}")


@router.get("/fact-types")
async def get_fact_types():
    """Get available fact types and their descriptions"""
    fact_type_info = {
        FactType.STATEMENT.value: "General factual statements",
        FactType.DECISION.value: "Decisions that were made",
        FactType.PREFERENCE.value: "User preferences and choices",
        FactType.ACTION_ITEM.value: "Tasks or actions to be taken",
        FactType.REQUIREMENT.value: "Requirements or constraints",
        FactType.OBSERVATION.value: "Observations or findings",
        FactType.RULE.value: "Rules or guidelines",
        FactType.CONFIGURATION.value: "Configuration or settings",
        FactType.ERROR_FACT.value: "Facts about errors or issues",
        FactType.SOLUTION.value: "Solutions or fixes",
        FactType.GOAL.value: "Goals or objectives",
        FactType.CONSTRAINT.value: "Limitations or constraints"
    }
    
    return {
        "fact_types": fact_type_info,
        "total_types": len(fact_type_info)
    }


@router.post("/analyze-facts")
async def analyze_fact_content(text: str):
    """
    Analyze text for fact content without full extraction.
    
    Returns quick analysis of what types of facts might be present.
    """
    try:
        import re
        
        # Quick pattern-based analysis
        analysis = {
            "text_length": len(text),
            "estimated_facts": 0,
            "likely_types": []
        }
        
        text_lower = text.lower()
        
        # Check for decision patterns
        decision_indicators = ['decided', 'chose', 'selected', 'will use', 'going with']
        if any(indicator in text_lower for indicator in decision_indicators):
            analysis["likely_types"].append("decision")
            analysis["estimated_facts"] += len([i for i in decision_indicators if i in text_lower])
        
        # Check for preference patterns
        preference_indicators = ['prefer', 'like', 'favor', 'think', 'believe']
        if any(indicator in text_lower for indicator in preference_indicators):
            analysis["likely_types"].append("preference")
            analysis["estimated_facts"] += 1
        
        # Check for action patterns
        action_indicators = ['need to', 'should', 'must', 'will', 'todo', 'action']
        if any(indicator in text_lower for indicator in action_indicators):
            analysis["likely_types"].append("action_item")
            analysis["estimated_facts"] += len([i for i in action_indicators if i in text_lower])
        
        # Check for error patterns
        error_indicators = ['error', 'issue', 'problem', 'bug', 'failure', 'exception']
        if any(indicator in text_lower for indicator in error_indicators):
            analysis["likely_types"].append("error_fact")
            analysis["estimated_facts"] += 1
        
        # Check for solution patterns
        solution_indicators = ['solution', 'fix', 'resolved', 'workaround']
        if any(indicator in text_lower for indicator in solution_indicators):
            analysis["likely_types"].append("solution")
            analysis["estimated_facts"] += 1
        
        # Check for configuration patterns
        config_indicators = ['config', 'setting', 'parameter', 'environment', 'configured']
        if any(indicator in text_lower for indicator in config_indicators):
            analysis["likely_types"].append("configuration")
            analysis["estimated_facts"] += 1
        
        # Check for observation patterns
        observation_indicators = ['noticed', 'observed', 'found', 'discovered', 'appears', 'seems']
        if any(indicator in text_lower for indicator in observation_indicators):
            analysis["likely_types"].append("observation")
            analysis["estimated_facts"] += 1
        
        analysis["fact_density"] = analysis["estimated_facts"] / len(text.split()) if text.split() else 0
        
        return analysis
        
    except Exception as e:
        logger.error(f"Fact analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Fact analysis failed: {str(e)}")


@router.post("/extract-by-type")
async def extract_facts_by_type(
    text: str,
    fact_types: List[str],
    context: Optional[Dict[str, Any]] = None,
    entities: Optional[List[str]] = None
):
    """
    Extract facts of specific types only.
    
    This endpoint allows filtering fact extraction to only specific types,
    which can be useful for targeted fact extraction.
    """
    try:
        # Validate fact types
        valid_types = {fact_type.value for fact_type in FactType}
        invalid_types = [ft for ft in fact_types if ft not in valid_types]
        
        if invalid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid fact types: {invalid_types}. Valid types: {list(valid_types)}"
            )
        
        # Extract all facts first
        all_facts = await extract_facts_from_text(text, context, entities)
        
        # Filter to requested types
        filtered_facts = [
            fact for fact in all_facts 
            if fact.fact_type.value in fact_types
        ]
        
        # Convert to response format
        fact_responses = []
        for fact in filtered_facts:
            fact_response = FactResponse(
                content=fact.content,
                fact_type=fact.fact_type.value,
                confidence=fact.confidence,
                certainty_level=fact.certainty_level,
                importance=fact.importance,
                source_span=[fact.source_span[0], fact.source_span[1]],
                context=fact.context,
                entities=fact.entities,
                temporal_info=fact.temporal_info,
                metadata=fact.metadata
            )
            fact_responses.append(fact_response)
        
        return {
            "facts": fact_responses,
            "total_facts": len(filtered_facts),
            "requested_types": fact_types,
            "facts_found_by_type": {
                ft: len([f for f in filtered_facts if f.fact_type.value == ft])
                for ft in fact_types
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Type-filtered fact extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Type-filtered fact extraction failed: {str(e)}")


@router.post("/extract-from-chunks")
async def extract_facts_from_chunks(chunks: List[Dict[str, Any]]):
    """
    Extract facts from multiple text chunks efficiently.
    
    This is optimized for processing multiple chunks from the text chunking service.
    """
    try:
        results = []
        total_facts = 0
        
        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '')
            if not content:
                continue
            
            # Get entities from chunk if available
            entities = chunk.get('entities', [])
            
            facts = await extract_facts_from_text(
                content, 
                context=chunk.get('metadata', {}),
                entities=entities
            )
            
            chunk_result = {
                "chunk_index": i,
                "facts": [{
                    "content": fact.content,
                    "type": fact.fact_type.value,
                    "confidence": fact.confidence,
                    "certainty_level": fact.certainty_level,
                    "importance": fact.importance,
                    "source_span": [fact.source_span[0], fact.source_span[1]],
                    "entities": fact.entities,
                    "temporal_info": fact.temporal_info
                } for fact in facts],
                "fact_count": len(facts),
                "chunk_metadata": chunk.get('metadata', {})
            }
            
            results.append(chunk_result)
            total_facts += len(facts)
        
        return {
            "results": results,
            "total_chunks": len(chunks),
            "total_facts": total_facts,
            "chunks_with_facts": sum(1 for r in results if r["fact_count"] > 0)
        }
        
    except Exception as e:
        logger.error(f"Chunk fact extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chunk fact extraction failed: {str(e)}")


@router.post("/fact-importance")
async def calculate_fact_importance(
    facts: List[str], 
    context: Optional[Dict[str, Any]] = None
):
    """
    Calculate importance scores for a list of fact statements.
    
    This endpoint can be used to rank facts by importance without full extraction.
    """
    try:
        from ...services.fact_extraction import IntelligentFactExtractor
        
        extractor = IntelligentFactExtractor()
        results = []
        
        for fact_text in facts:
            # Create a simple fact object for importance calculation
            temp_facts = await extractor.extract_facts(fact_text, context or {})
            
            if temp_facts:
                # Take the most important fact found
                best_fact = max(temp_facts, key=lambda f: f.importance)
                results.append({
                    "fact": fact_text,
                    "importance": best_fact.importance,
                    "confidence": best_fact.confidence,
                    "fact_type": best_fact.fact_type.value
                })
            else:
                # Fallback scoring for facts that don't match patterns
                base_importance = 0.3
                if any(keyword in fact_text.lower() for keyword in ['decision', 'decided', 'chose']):
                    base_importance = 0.8
                elif any(keyword in fact_text.lower() for keyword in ['need', 'must', 'should']):
                    base_importance = 0.7
                
                results.append({
                    "fact": fact_text,
                    "importance": base_importance,
                    "confidence": 0.5,
                    "fact_type": "statement"
                })
        
        # Sort by importance
        results.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            "ranked_facts": results,
            "total_facts": len(results),
            "avg_importance": sum(r['importance'] for r in results) / len(results) if results else 0
        }
        
    except Exception as e:
        logger.error(f"Fact importance calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Fact importance calculation failed: {str(e)}")