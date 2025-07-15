"""
Code-specific embedding generation using specialized models.

This module provides embedding generation specifically optimized for code,
using models like CodeBERT, GraphCodeBERT, and language-specific embeddings.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CodeLanguage(Enum):
    """Supported programming languages for embeddings."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    GENERAL = "general"


@dataclass
class CodeEmbeddingModel:
    """Configuration for a code embedding model."""
    name: str
    model_id: str
    languages: List[CodeLanguage]
    dimension: int
    max_length: int


class CodeEmbeddingService:
    """
    Service for generating code-specific embeddings using specialized models.
    
    Features:
    - Multiple model support (CodeBERT, GraphCodeBERT, CodeT5)
    - Language-specific embeddings
    - Code structure awareness
    - Semantic code understanding
    """
    
    # Available models for code embeddings
    MODELS = {
        "codebert": CodeEmbeddingModel(
            name="CodeBERT",
            model_id="microsoft/codebert-base",
            languages=[lang for lang in CodeLanguage],
            dimension=768,
            max_length=512
        ),
        "graphcodebert": CodeEmbeddingModel(
            name="GraphCodeBERT",
            model_id="microsoft/graphcodebert-base",
            languages=[lang for lang in CodeLanguage],
            dimension=768,
            max_length=512
        ),
        "codet5": CodeEmbeddingModel(
            name="CodeT5",
            model_id="Salesforce/codet5-base",
            languages=[lang for lang in CodeLanguage],
            dimension=768,
            max_length=512
        ),
        "unixcoder": CodeEmbeddingModel(
            name="UniXcoder",
            model_id="microsoft/unixcoder-base",
            languages=[lang for lang in CodeLanguage],
            dimension=768,
            max_length=512
        ),
        "codegen": CodeEmbeddingModel(
            name="CodeGen",
            model_id="Salesforce/codegen-350M-mono",
            languages=[CodeLanguage.PYTHON],
            dimension=1024,
            max_length=2048
        )
    }
    
    def __init__(self, 
                 default_model: str = "graphcodebert",
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize the code embedding service.
        
        Args:
            default_model: Default model to use for embeddings
            device: Device to run models on (cuda/cpu)
            cache_dir: Directory to cache models
        """
        self.default_model = default_model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        
        # Model cache
        self._models: Dict[str, Tuple[Any, Any]] = {}
        
        # Fallback to general-purpose model
        self._fallback_model = None
        
        logger.info(f"Initialized CodeEmbeddingService with device: {self.device}")
    
    def _load_model(self, model_name: str) -> Tuple[Any, Any]:
        """Load a specific model and tokenizer."""
        if model_name in self._models:
            return self._models[model_name]
        
        try:
            model_config = self.MODELS[model_name]
            logger.info(f"Loading {model_config.name} model...")
            
            if model_name in ["codebert", "graphcodebert"]:
                tokenizer = RobertaTokenizer.from_pretrained(
                    model_config.model_id,
                    cache_dir=self.cache_dir
                )
                model = RobertaModel.from_pretrained(
                    model_config.model_id,
                    cache_dir=self.cache_dir
                ).to(self.device)
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config.model_id,
                    cache_dir=self.cache_dir
                )
                model = AutoModel.from_pretrained(
                    model_config.model_id,
                    cache_dir=self.cache_dir
                ).to(self.device)
            
            model.eval()
            self._models[model_name] = (tokenizer, model)
            
            logger.info(f"Successfully loaded {model_config.name}")
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {str(e)}")
            return self._get_fallback_model()
    
    def _get_fallback_model(self) -> Tuple[Any, Any]:
        """Get fallback model for general embeddings."""
        if self._fallback_model is None:
            logger.info("Loading fallback model...")
            self._fallback_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device=self.device
            )
        return self._fallback_model, None
    
    def _preprocess_code(self, code: str, language: CodeLanguage) -> str:
        """
        Preprocess code for better embedding quality.
        
        Args:
            code: Source code
            language: Programming language
            
        Returns:
            Preprocessed code
        """
        # Remove excessive whitespace
        lines = code.strip().split('\n')
        processed_lines = []
        
        for line in lines:
            # Keep indentation structure
            stripped = line.rstrip()
            if stripped:
                processed_lines.append(stripped)
        
        # Add language tag for better context
        if language != CodeLanguage.GENERAL:
            return f"<{language.value}>\n{chr(10).join(processed_lines)}\n</{language.value}>"
        
        return '\n'.join(processed_lines)
    
    def generate_embedding(self,
                         code: str,
                         language: CodeLanguage = CodeLanguage.GENERAL,
                         model_name: Optional[str] = None) -> np.ndarray:
        """
        Generate embedding for a single code snippet.
        
        Args:
            code: Source code to embed
            language: Programming language
            model_name: Specific model to use
            
        Returns:
            Embedding vector
        """
        model_name = model_name or self.default_model
        
        # Check if model supports the language
        if model_name in self.MODELS:
            model_config = self.MODELS[model_name]
            if language not in model_config.languages:
                logger.warning(f"{model_config.name} doesn't support {language.value}, using default")
                model_name = self.default_model
        
        try:
            # Preprocess code
            processed_code = self._preprocess_code(code, language)
            
            # Load model
            tokenizer, model = self._load_model(model_name)
            
            if tokenizer is None:  # Fallback model
                return model.encode(processed_code)
            
            # Tokenize
            inputs = tokenizer(
                processed_code,
                return_tensors='pt',
                max_length=self.MODELS[model_name].max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                
                # Use CLS token embedding or mean pooling
                if hasattr(outputs, 'pooler_output'):
                    embeddings = outputs.pooler_output
                else:
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                
                return embeddings.cpu().numpy()[0]
                
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Fallback to basic model
            fallback_model, _ = self._get_fallback_model()
            return fallback_model.encode(code)
    
    def generate_embeddings_batch(self,
                                code_snippets: List[Dict[str, Any]],
                                model_name: Optional[str] = None,
                                batch_size: int = 8) -> List[np.ndarray]:
        """
        Generate embeddings for multiple code snippets.
        
        Args:
            code_snippets: List of dicts with 'code' and 'language' keys
            model_name: Specific model to use
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        model_name = model_name or self.default_model
        embeddings = []
        
        # Group by language for better batching
        language_groups = {}
        for i, snippet in enumerate(code_snippets):
            lang = snippet.get('language', CodeLanguage.GENERAL)
            if isinstance(lang, str):
                lang = CodeLanguage(lang) if lang in [l.value for l in CodeLanguage] else CodeLanguage.GENERAL
            
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append((i, snippet['code']))
        
        # Process each language group
        for language, code_items in language_groups.items():
            logger.info(f"Processing {len(code_items)} {language.value} snippets")
            
            # Process in batches
            for i in range(0, len(code_items), batch_size):
                batch = code_items[i:i + batch_size]
                batch_embeddings = []
                
                for idx, code in batch:
                    embedding = self.generate_embedding(code, language, model_name)
                    batch_embeddings.append((idx, embedding))
                
                # Store in correct order
                for idx, embedding in batch_embeddings:
                    embeddings.append((idx, embedding))
        
        # Sort by original index and return
        embeddings.sort(key=lambda x: x[0])
        return [emb for _, emb in embeddings]
    
    def get_similarity(self, 
                      embedding1: np.ndarray, 
                      embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Ensure result is between 0 and 1
        return (similarity + 1) / 2
    
    def find_similar_code(self,
                         query_code: str,
                         code_corpus: List[Dict[str, Any]],
                         language: CodeLanguage = CodeLanguage.GENERAL,
                         top_k: int = 5,
                         threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find similar code snippets in a corpus.
        
        Args:
            query_code: Code to search for
            code_corpus: List of code snippets with metadata
            language: Programming language
            top_k: Number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of similar code snippets with scores
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query_code, language)
        
        # Generate corpus embeddings
        corpus_embeddings = self.generate_embeddings_batch([
            {'code': item['code'], 'language': item.get('language', language)}
            for item in code_corpus
        ])
        
        # Calculate similarities
        similarities = []
        for i, (item, embedding) in enumerate(zip(code_corpus, corpus_embeddings)):
            similarity = self.get_similarity(query_embedding, embedding)
            if similarity >= threshold:
                similarities.append({
                    'code': item['code'],
                    'metadata': item.get('metadata', {}),
                    'similarity': float(similarity),
                    'language': item.get('language', language.value)
                })
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def generate_code_fingerprint(self, 
                                code: str, 
                                language: CodeLanguage = CodeLanguage.GENERAL) -> str:
        """
        Generate a unique fingerprint for code deduplication.
        
        Args:
            code: Source code
            language: Programming language
            
        Returns:
            Hex string fingerprint
        """
        # Generate embedding
        embedding = self.generate_embedding(code, language)
        
        # Quantize to reduce dimensionality
        quantized = np.round(embedding * 100).astype(np.int32)
        
        # Hash the quantized vector
        fingerprint = hashlib.sha256(quantized.tobytes()).hexdigest()
        
        return fingerprint
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Model name (uses default if None)
            
        Returns:
            Model information dict
        """
        model_name = model_name or self.default_model
        
        if model_name not in self.MODELS:
            return {
                "error": f"Model {model_name} not found",
                "available_models": list(self.MODELS.keys())
            }
        
        model_config = self.MODELS[model_name]
        return {
            "name": model_config.name,
            "model_id": model_config.model_id,
            "supported_languages": [lang.value for lang in model_config.languages],
            "embedding_dimension": model_config.dimension,
            "max_input_length": model_config.max_length,
            "device": self.device
        }