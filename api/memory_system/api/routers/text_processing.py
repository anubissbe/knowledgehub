"""Text processing API router for intelligent chunking"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...services.text_chunking import (
    chunk_conversation_text, 
    process_conversation_chunks,
    ChunkType,
    TextChunk
)

logger = logging.getLogger(__name__)
router = APIRouter()


class ChunkRequest(BaseModel):
    """Request for text chunking"""
    text: str = Field(..., description="Text to chunk", min_length=1)
    max_chunk_size: int = Field(512, description="Maximum chunk size", gt=0, le=2048)
    min_chunk_size: int = Field(50, description="Minimum chunk size", gt=0, le=1024)
    overlap_size: int = Field(50, description="Overlap size between chunks", ge=0, le=512)
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ConversationProcessRequest(BaseModel):
    """Request for conversation processing"""
    text: str = Field(..., description="Conversation text to process", min_length=1)
    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    max_chunk_size: int = Field(512, description="Maximum chunk size", gt=0, le=2048)
    min_chunk_size: int = Field(50, description="Minimum chunk size", gt=0, le=1024)


class ChunkResponse(BaseModel):
    """Response for a single chunk"""
    content: str
    chunk_type: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any]
    importance: float
    contains_code: bool
    language: Optional[str]
    entities: List[str]
    semantic_boundary: bool


class ChunkingResponse(BaseModel):
    """Response for text chunking"""
    chunks: List[ChunkResponse]
    total_chunks: int
    total_characters: int
    processing_time_ms: float
    statistics: Dict[str, Any]


@router.get("/health")
async def text_processing_health():
    """Health check for text processing service"""
    return {
        "status": "healthy",
        "service": "text_processing",
        "features": [
            "intelligent_chunking",
            "code_block_preservation", 
            "semantic_boundaries",
            "importance_scoring",
            "chunk_type_detection",
            "overlap_handling"
        ],
        "supported_chunk_types": [chunk_type.value for chunk_type in ChunkType]
    }


@router.post("/chunk", response_model=ChunkingResponse)
async def chunk_text(request: ChunkRequest):
    """
    Chunk text into intelligent segments.
    
    This endpoint takes raw text and splits it into meaningful chunks while:
    - Preserving code blocks and commands
    - Maintaining semantic boundaries
    - Detecting chunk types (conversation, code, question, etc.)
    - Scoring importance
    - Adding metadata and analysis
    """
    import time
    start_time = time.time()
    
    try:
        # Import chunker here to use custom settings
        from ...services.text_chunking import IntelligentTextChunker
        
        chunker = IntelligentTextChunker(
            max_chunk_size=request.max_chunk_size,
            min_chunk_size=request.min_chunk_size,
            overlap_size=request.overlap_size
        )
        
        chunks = chunker.chunk_conversation(request.text, request.context)
        
        # Convert to response format
        chunk_responses = []
        for chunk in chunks:
            chunk_response = ChunkResponse(
                content=chunk.content,
                chunk_type=chunk.chunk_type.value,
                start_index=chunk.start_index,
                end_index=chunk.end_index,
                metadata=chunk.metadata,
                importance=chunk.importance,
                contains_code=chunk.contains_code,
                language=chunk.language,
                entities=chunk.entities,
                semantic_boundary=chunk.semantic_boundary
            )
            chunk_responses.append(chunk_response)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate statistics
        chunk_types = [chunk.chunk_type.value for chunk in chunks]
        type_counts = {chunk_type.value: chunk_types.count(chunk_type.value) for chunk_type in ChunkType}
        
        statistics = {
            "chunk_type_distribution": type_counts,
            "avg_chunk_size": sum(len(chunk.content) for chunk in chunks) / len(chunks) if chunks else 0,
            "avg_importance": sum(chunk.importance for chunk in chunks) / len(chunks) if chunks else 0,
            "code_chunks": sum(1 for chunk in chunks if chunk.contains_code),
            "languages_detected": list(set(chunk.language for chunk in chunks if chunk.language)),
            "semantic_boundaries": sum(1 for chunk in chunks if chunk.semantic_boundary)
        }
        
        return ChunkingResponse(
            chunks=chunk_responses,
            total_chunks=len(chunks),
            total_characters=len(request.text),
            processing_time_ms=processing_time,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"Text chunking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")


@router.post("/process-conversation", response_model=List[Dict[str, Any]])
async def process_conversation(request: ConversationProcessRequest):
    """
    Process conversation text into chunks ready for memory storage.
    
    This endpoint is optimized for conversation processing and returns
    chunks in a format ready for direct storage in the memory system.
    """
    try:
        chunks = await process_conversation_chunks(
            text=request.text,
            session_id=request.session_id,
            user_id=request.user_id
        )
        
        logger.info(f"Processed conversation into {len(chunks)} chunks for session {request.session_id}")
        return chunks
        
    except Exception as e:
        logger.error(f"Conversation processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Conversation processing failed: {str(e)}")


@router.get("/chunk-types")
async def get_chunk_types():
    """Get available chunk types and their descriptions"""
    chunk_type_info = {
        ChunkType.CONVERSATION.value: "General conversation content",
        ChunkType.CODE_BLOCK.value: "Code blocks and programming content",
        ChunkType.COMMAND.value: "Command line instructions and terminal commands",
        ChunkType.ERROR_MESSAGE.value: "Error messages and exceptions",
        ChunkType.QUESTION.value: "Questions and inquiries",
        ChunkType.ANSWER.value: "Answers and responses",
        ChunkType.INSTRUCTION.value: "Instructions and directives",
        ChunkType.FACT.value: "Factual statements and information",
        ChunkType.DECISION.value: "Decisions and choices made"
    }
    
    return {
        "chunk_types": chunk_type_info,
        "total_types": len(chunk_type_info)
    }


@router.post("/analyze-text")
async def analyze_text_content(text: str):
    """
    Analyze text content without chunking.
    
    Returns analysis of the text including:
    - Estimated chunk count
    - Content type detection
    - Code block detection
    - Complexity metrics
    """
    try:
        from ...services.text_chunking import IntelligentTextChunker
        import re
        
        chunker = IntelligentTextChunker()
        
        # Quick analysis without full chunking
        code_blocks = len(re.findall(r'```.*?```', text, re.DOTALL))
        inline_code = len(re.findall(r'`[^`]+`', text))
        commands = len(re.findall(r'(?:^|\n)\s*(?:\$|>|#)\s*[^\n]+', text, re.MULTILINE))
        questions = len(re.findall(r'\?', text))
        
        # Estimate chunks
        estimated_chunks = max(1, len(text) // 512)
        
        # Detect primary content type
        if code_blocks > 0:
            primary_type = "technical_with_code"
        elif commands > 0:
            primary_type = "instructional"
        elif questions > 2:
            primary_type = "conversational"
        else:
            primary_type = "general"
        
        analysis = {
            "text_length": len(text),
            "estimated_chunks": estimated_chunks,
            "primary_content_type": primary_type,
            "code_blocks": code_blocks,
            "inline_code": inline_code,
            "commands": commands,
            "questions": questions,
            "lines": len(text.split('\n')),
            "words": len(text.split()),
            "complexity_score": min(1.0, (code_blocks * 0.3 + commands * 0.2 + questions * 0.1) / len(text) * 1000)
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")


@router.post("/preview-chunks")
async def preview_chunks(request: ChunkRequest):
    """
    Preview how text would be chunked without full processing.
    
    Returns a lightweight preview showing chunk boundaries and types
    without full metadata processing.
    """
    try:
        from ...services.text_chunking import IntelligentTextChunker
        
        chunker = IntelligentTextChunker(
            max_chunk_size=request.max_chunk_size,
            min_chunk_size=request.min_chunk_size,
            overlap_size=request.overlap_size
        )
        
        chunks = chunker.chunk_conversation(request.text, request.context)
        
        # Create lightweight preview
        preview = []
        for i, chunk in enumerate(chunks):
            preview.append({
                "index": i,
                "type": chunk.chunk_type.value,
                "length": len(chunk.content),
                "preview": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
                "importance": round(chunk.importance, 2),
                "contains_code": chunk.contains_code,
                "language": chunk.language
            })
        
        return {
            "preview": preview,
            "total_chunks": len(chunks),
            "settings": {
                "max_chunk_size": request.max_chunk_size,
                "min_chunk_size": request.min_chunk_size,
                "overlap_size": request.overlap_size
            }
        }
        
    except Exception as e:
        logger.error(f"Chunk preview failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chunk preview failed: {str(e)}")