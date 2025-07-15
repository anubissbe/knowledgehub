"""
Simplified code embeddings service for basic functionality.

This module provides basic code analysis without heavy ML dependencies,
focusing on structural analysis and basic similarity detection.
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
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    GENERAL = "general"


@dataclass
class CodeEmbeddingModel:
    """Information about a code embedding model"""
    name: str
    model_id: str
    dimension: int
    description: str


class CodeEmbeddingService:
    """
    Simplified service for basic code analysis and similarity detection.
    
    Uses lightweight methods for code analysis without requiring heavy ML models.
    """
    
    MODELS = {
        "simple": CodeEmbeddingModel(
            name="Simple Code Analyzer",
            model_id="builtin/simple",
            dimension=128,
            description="Basic structural code analysis"
        )
    }
    
    def __init__(self):
        self.default_model = "simple"
        logger.info("Initialized simplified CodeEmbeddingService")
    
    def generate_embedding(self, code: str, language: CodeLanguage = CodeLanguage.GENERAL, model: str = None) -> np.ndarray:
        """
        Generate a simple embedding based on code structure.
        
        Args:
            code: Source code to analyze
            language: Programming language 
            model: Model to use (currently only 'simple')
            
        Returns:
            128-dimensional embedding vector
        """
        try:
            # Extract basic features
            features = self._extract_code_features(code, language)
            
            # Convert to fixed-size embedding
            embedding = self._features_to_embedding(features)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector on error
            return np.zeros(128, dtype=np.float32)
    
    def _extract_code_features(self, code: str, language: CodeLanguage) -> Dict[str, float]:
        """Extract basic structural features from code"""
        features = {}
        
        # Basic metrics
        features['line_count'] = len(code.split('\n'))
        features['char_count'] = len(code)
        features['word_count'] = len(code.split())
        
        # Language-specific patterns
        if language == CodeLanguage.PYTHON:
            features['def_count'] = len(re.findall(r'\bdef\s+\w+', code))
            features['class_count'] = len(re.findall(r'\bclass\s+\w+', code))
            features['import_count'] = len(re.findall(r'\bimport\s+\w+', code))
        elif language == CodeLanguage.JAVASCRIPT:
            features['function_count'] = len(re.findall(r'\bfunction\s+\w+', code))
            features['const_count'] = len(re.findall(r'\bconst\s+\w+', code))
            features['let_count'] = len(re.findall(r'\blet\s+\w+', code))
        elif language == CodeLanguage.JAVA:
            features['class_count'] = len(re.findall(r'\bclass\s+\w+', code))
            features['method_count'] = len(re.findall(r'\bpublic\s+\w+\s+\w+\s*\(', code))
            features['import_count'] = len(re.findall(r'\bimport\s+[\w.]+', code))
        
        # General patterns
        features['brace_count'] = code.count('{') + code.count('}')
        features['paren_count'] = code.count('(') + code.count(')')
        features['bracket_count'] = code.count('[') + code.count(']')
        features['semicolon_count'] = code.count(';')
        features['comment_count'] = len(re.findall(r'//.*|/\*.*?\*/|#.*', code))
        
        # Complexity indicators
        features['if_count'] = len(re.findall(r'\bif\s*\(', code))
        features['for_count'] = len(re.findall(r'\bfor\s*\(', code))
        features['while_count'] = len(re.findall(r'\bwhile\s*\(', code))
        
        # Normalize features
        total_chars = max(features['char_count'], 1)
        for key, value in features.items():
            if key != 'char_count':
                features[key] = value / total_chars
        
        return features
    
    def _features_to_embedding(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to fixed-size embedding vector"""
        # Define feature order for consistency
        feature_keys = [
            'line_count', 'char_count', 'word_count', 'def_count', 'class_count',
            'import_count', 'function_count', 'const_count', 'let_count', 
            'method_count', 'brace_count', 'paren_count', 'bracket_count',
            'semicolon_count', 'comment_count', 'if_count', 'for_count', 'while_count'
        ]
        
        # Create base vector from features
        base_vector = []
        for key in feature_keys:
            base_vector.append(features.get(key, 0.0))
        
        # Pad or trim to exactly 128 dimensions
        while len(base_vector) < 128:
            # Add derived features
            if len(base_vector) < 32:
                # Add squared features
                base_vector.extend([x**2 for x in base_vector[:min(16, 32-len(base_vector))]])
            elif len(base_vector) < 64:
                # Add cross-products
                for i in range(min(8, len(feature_keys))):
                    for j in range(i+1, min(8, len(feature_keys))):
                        if len(base_vector) >= 64:
                            break
                        base_vector.append(features.get(feature_keys[i], 0) * features.get(feature_keys[j], 0))
            else:
                # Add random walk features based on hash
                hash_val = hash(str(features)) % 1000
                base_vector.append((hash_val + len(base_vector)) / 1000.0)
        
        return np.array(base_vector[:128], dtype=np.float32)
    
    def find_similar_code(self, query_code: str, code_corpus: List[Dict[str, Any]], 
                         language: CodeLanguage = CodeLanguage.GENERAL,
                         top_k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find similar code snippets using cosine similarity"""
        try:
            query_embedding = self.generate_embedding(query_code, language)
            
            similarities = []
            for i, snippet in enumerate(code_corpus):
                snippet_code = snippet.get('code', '')
                snippet_lang = CodeLanguage(snippet.get('language', 'general'))
                
                snippet_embedding = self.generate_embedding(snippet_code, snippet_lang)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, snippet_embedding)
                
                if similarity >= threshold:
                    result = snippet.copy()
                    result['similarity'] = float(similarity)
                    result['index'] = i
                    similarities.append(result)
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar code: {e}")
            return []
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
        except:
            return 0.0
    
    def generate_embeddings_batch(self, code_snippets: List[Dict[str, Any]], 
                                 model: str = None) -> List[np.ndarray]:
        """Generate embeddings for multiple code snippets"""
        embeddings = []
        for snippet in code_snippets:
            code = snippet.get('code', '')
            language = snippet.get('language', CodeLanguage.GENERAL)
            if isinstance(language, str):
                try:
                    language = CodeLanguage(language.lower())
                except ValueError:
                    language = CodeLanguage.GENERAL
            
            embedding = self.generate_embedding(code, language, model)
            embeddings.append(embedding)
        
        return embeddings
    
    def generate_code_fingerprint(self, code: str, language: CodeLanguage = CodeLanguage.GENERAL) -> str:
        """Generate a unique fingerprint for code deduplication"""
        try:
            # Normalize code (remove whitespace, comments)
            normalized = re.sub(r'\s+', ' ', code.strip())
            normalized = re.sub(r'//.*|/\*.*?\*/|#.*', '', normalized)
            
            # Create hash
            return hashlib.md5(normalized.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating fingerprint: {e}")
            return hashlib.md5(code.encode()).hexdigest()
    
    def detect_language(self, code: str) -> Optional[CodeLanguage]:
        """Detect programming language from code"""
        code_lower = code.lower()
        
        # Simple heuristic-based detection
        if 'def ' in code and 'import ' in code:
            return CodeLanguage.PYTHON
        elif 'function ' in code and ('var ' in code or 'let ' in code or 'const ' in code):
            return CodeLanguage.JAVASCRIPT
        elif 'public class ' in code or 'private class ' in code:
            return CodeLanguage.JAVA
        elif 'func ' in code and 'package ' in code:
            return CodeLanguage.GO
        elif 'fn ' in code and ('let ' in code or 'mut ' in code):
            return CodeLanguage.RUST
        elif '#include' in code and ('int main' in code or 'void main' in code):
            return CodeLanguage.CPP if '::' in code else CodeLanguage.C
        else:
            return CodeLanguage.GENERAL
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_name in self.MODELS:
            model = self.MODELS[model_name]
            return {
                'name': model.name,
                'model_id': model.model_id,
                'dimension': model.dimension,
                'description': model.description,
                'available': True
            }
        else:
            return {
                'name': model_name,
                'model_id': 'unknown',
                'dimension': 0,
                'description': 'Unknown model',
                'available': False
            }