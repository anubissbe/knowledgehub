"""
Code embeddings API router.

Provides endpoints for generating code-specific embeddings and similarity search.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..services.code_embeddings_simple import CodeEmbeddingService, CodeLanguage
from ..dependencies import get_current_user
from ...shared.logging import setup_logging

logger = setup_logging("api.code_embeddings")

router = APIRouter(prefix="/api/code-embeddings", tags=["code-embeddings"])


class CodeEmbeddingRequest(BaseModel):
    """Request for generating code embedding"""
    code: str = Field(..., description="Source code to embed")
    language: Optional[str] = Field(None, description="Programming language")
    model: Optional[str] = Field("graphcodebert", description="Embedding model to use")


class CodeEmbeddingResponse(BaseModel):
    """Response with code embedding"""
    embedding: List[float] = Field(..., description="Embedding vector")
    dimension: int = Field(..., description="Embedding dimension")
    model: str = Field(..., description="Model used")
    language: str = Field(..., description="Detected/specified language")


class CodeSimilarityRequest(BaseModel):
    """Request for code similarity search"""
    query_code: str = Field(..., description="Code to search for")
    code_corpus: List[Dict[str, Any]] = Field(..., description="Code snippets to search in")
    language: Optional[str] = Field(None, description="Programming language")
    top_k: int = Field(5, description="Number of results")
    threshold: float = Field(0.7, description="Similarity threshold")


class CodeSimilarityResponse(BaseModel):
    """Response with similar code snippets"""
    results: List[Dict[str, Any]] = Field(..., description="Similar code snippets")
    query_language: str = Field(..., description="Detected query language")


class BatchCodeEmbeddingRequest(BaseModel):
    """Request for batch code embedding"""
    snippets: List[Dict[str, str]] = Field(..., description="Code snippets with optional language")
    model: Optional[str] = Field("graphcodebert", description="Embedding model to use")


class ModelInfoResponse(BaseModel):
    """Response with model information"""
    models: Dict[str, Dict[str, Any]] = Field(..., description="Available models and their info")
    default_model: str = Field(..., description="Default model")


# Initialize services
code_embedding_service = CodeEmbeddingService()


@router.post("/generate", response_model=CodeEmbeddingResponse)
async def generate_code_embedding(
    request: CodeEmbeddingRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate embedding for source code using specialized models.
    
    Available models:
    - codebert: Microsoft's CodeBERT for general code understanding
    - graphcodebert: GraphCodeBERT with data flow understanding
    - codet5: Salesforce's CodeT5 for code generation/understanding
    - unixcoder: UniXcoder for unified code representation
    - codegen: CodeGen for Python-specific embeddings
    """
    try:
        # Parse language
        language = CodeLanguage.GENERAL
        if request.language:
            try:
                language = CodeLanguage(request.language.lower())
            except ValueError:
                logger.warning(f"Unknown language: {request.language}, using general")
        
        # Generate embedding
        embedding = code_embedding_service.generate_embedding(
            request.code,
            language,
            request.model
        )
        
        # Get model info
        model_info = code_embedding_service.get_model_info(request.model)
        
        return CodeEmbeddingResponse(
            embedding=embedding.tolist(),
            dimension=len(embedding),
            model=request.model or code_embedding_service.default_model,
            language=language.value
        )
        
    except Exception as e:
        logger.error(f"Error generating code embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similarity", response_model=CodeSimilarityResponse)
async def find_similar_code(
    request: CodeSimilarityRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Find similar code snippets using code-specific embeddings.
    
    The code corpus should be a list of dictionaries with:
    - code: The source code
    - language: Optional programming language
    - metadata: Optional metadata dict
    """
    try:
        # Parse language
        language = CodeLanguage.GENERAL
        if request.language:
            try:
                language = CodeLanguage(request.language.lower())
            except ValueError:
                pass
        
        # Find similar code
        results = code_embedding_service.find_similar_code(
            request.query_code,
            request.code_corpus,
            language,
            request.top_k,
            request.threshold
        )
        
        return CodeSimilarityResponse(
            results=results,
            query_language=language.value
        )
        
    except Exception as e:
        logger.error(f"Error finding similar code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=List[CodeEmbeddingResponse])
async def generate_batch_embeddings(
    request: BatchCodeEmbeddingRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate embeddings for multiple code snippets.
    
    Each snippet should have:
    - code: The source code
    - language: Optional programming language
    """
    try:
        # Prepare snippets
        code_snippets = []
        for snippet in request.snippets:
            lang = snippet.get('language', 'general')
            try:
                lang = CodeLanguage(lang.lower())
            except ValueError:
                lang = CodeLanguage.GENERAL
            
            code_snippets.append({
                'code': snippet['code'],
                'language': lang
            })
        
        # Generate embeddings
        embeddings = code_embedding_service.generate_embeddings_batch(
            code_snippets,
            request.model
        )
        
        # Build responses
        responses = []
        for snippet, embedding in zip(code_snippets, embeddings):
            responses.append(CodeEmbeddingResponse(
                embedding=embedding.tolist(),
                dimension=len(embedding),
                model=request.model or code_embedding_service.default_model,
                language=snippet['language'].value
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelInfoResponse)
async def get_available_models(
    current_user: dict = Depends(get_current_user)
):
    """Get information about available code embedding models."""
    try:
        models_info = {}
        
        for model_name in code_embedding_service.MODELS:
            models_info[model_name] = code_embedding_service.get_model_info(model_name)
        
        return ModelInfoResponse(
            models=models_info,
            default_model=code_embedding_service.default_model
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fingerprint")
async def generate_code_fingerprint(
    request: CodeEmbeddingRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate a unique fingerprint for code deduplication.
    
    Returns a hash that can be used to identify similar code.
    """
    try:
        # Parse language
        language = CodeLanguage.GENERAL
        if request.language:
            try:
                language = CodeLanguage(request.language.lower())
            except ValueError:
                pass
        
        # Generate fingerprint
        fingerprint = code_embedding_service.generate_code_fingerprint(
            request.code,
            language
        )
        
        return {
            "fingerprint": fingerprint,
            "language": language.value
        }
        
    except Exception as e:
        logger.error(f"Error generating fingerprint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-language")
async def detect_code_language(
    code: str,
    current_user: dict = Depends(get_current_user)
):
    """Detect the programming language of code."""
    try:
        # Use basic language detection
        language = code_embedding_service.detect_language(code)
        
        return {
            "detected_language": language.value if language else None,
            "is_code": language is not None
        }
        
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))