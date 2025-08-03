"""
Query Decomposer for Multi-Agent System
Breaks down complex queries into sub-tasks
"""

from typing import Dict, List, Any, Optional
import re
from dataclasses import dataclass

import logging
logger = logging.getLogger(__name__)


@dataclass
class SubQuery:
    """Represents a decomposed sub-query"""
    id: str
    text: str
    type: str  # documentation, code, performance, style, testing
    keywords: List[str]
    dependencies: List[str] = None
    priority: int = 1


class QueryDecomposer:
    """
    Decomposes complex queries into manageable sub-queries
    Uses pattern matching and NLP techniques
    """
    
    def __init__(self):
        self.logger = logger
        
        # Keywords for query type classification
        self.type_keywords = {
            "documentation": [
                "docs", "documentation", "guide", "tutorial", "example",
                "how to", "explain", "what is", "reference", "api"
            ],
            "code": [
                "implement", "code", "function", "class", "method",
                "algorithm", "pattern", "structure", "architecture"
            ],
            "performance": [
                "performance", "optimize", "speed", "fast", "slow",
                "benchmark", "latency", "throughput", "scale", "efficient"
            ],
            "style": [
                "style", "convention", "best practice", "clean", "readable",
                "maintainable", "format", "lint", "standard"
            ],
            "testing": [
                "test", "testing", "unit test", "integration", "e2e",
                "coverage", "mock", "assert", "verify", "validate"
            ]
        }
        
        # Conjunctions that indicate multiple sub-queries
        self.conjunctions = ["and", "also", "plus", "with", "including", "as well as"]
        
        # Dependency indicators
        self.dependency_indicators = ["then", "after", "before", "first", "finally", "next"]
    
    async def decompose(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Decompose a complex query into sub-queries
        
        Args:
            query: The original query
            context: Additional context
            
        Returns:
            Decomposition result with sub-queries and metadata
        """
        try:
            # Clean and normalize query
            normalized_query = self._normalize_query(query)
            
            # Extract query intents
            intents = self._extract_intents(normalized_query)
            
            # Create sub-queries from intents
            sub_queries = self._create_sub_queries(intents, normalized_query)
            
            # Identify dependencies
            sub_queries = self._identify_dependencies(sub_queries, normalized_query)
            
            # Calculate complexity
            complexity = self._calculate_complexity(sub_queries)
            
            self.logger.info(
                f"Decomposed query into {len(sub_queries)} sub-queries",
                extra={
                    "original_query": query,
                    "sub_query_count": len(sub_queries),
                    "complexity": complexity
                }
            )
            
            return {
                "original_query": query,
                "sub_queries": [self._serialize_sub_query(sq) for sq in sub_queries],
                "complexity": complexity,
                "query_type": self._determine_primary_type(sub_queries),
                "estimated_agents": len(set(sq.type for sq in sub_queries))
            }
            
        except Exception as e:
            self.logger.error(f"Error decomposing query: {str(e)}")
            # Return simple decomposition on error
            return {
                "original_query": query,
                "sub_queries": [{
                    "id": "main",
                    "text": query,
                    "type": "general",
                    "keywords": [],
                    "priority": 1
                }],
                "complexity": 1.0,
                "query_type": "general",
                "estimated_agents": 1
            }
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for analysis"""
        # Convert to lowercase and clean whitespace
        normalized = query.lower().strip()
        
        # Expand common abbreviations
        abbreviations = {
            "impl": "implement",
            "perf": "performance",
            "docs": "documentation",
            "e2e": "end to end",
            "db": "database",
            "api": "api",
            "ui": "user interface"
        }
        
        for abbr, full in abbreviations.items():
            normalized = re.sub(r'\b' + abbr + r'\b', full, normalized)
        
        return normalized
    
    def _extract_intents(self, query: str) -> List[Dict[str, Any]]:
        """Extract multiple intents from the query"""
        intents = []
        
        # Split by conjunctions to find multiple intents
        parts = [query]
        for conjunction in self.conjunctions:
            new_parts = []
            for part in parts:
                splits = part.split(f" {conjunction} ")
                new_parts.extend(splits)
            parts = new_parts
        
        # Analyze each part for intent
        for i, part in enumerate(parts):
            if len(part.strip()) > 5:  # Minimum meaningful length
                intent_type = self._classify_intent(part)
                keywords = self._extract_keywords(part)
                
                intents.append({
                    "text": part.strip(),
                    "type": intent_type,
                    "keywords": keywords,
                    "position": i
                })
        
        # If no intents found, use the whole query
        if not intents:
            intents.append({
                "text": query,
                "type": self._classify_intent(query),
                "keywords": self._extract_keywords(query),
                "position": 0
            })
        
        return intents
    
    def _classify_intent(self, text: str) -> str:
        """Classify the intent type of a text segment"""
        text_lower = text.lower()
        
        # Count keyword matches for each type
        type_scores = {}
        for intent_type, keywords in self.type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            type_scores[intent_type] = score
        
        # Return type with highest score
        if max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        
        # Default based on common patterns
        if "?" in text:
            return "documentation"  # Questions often need documentation
        elif any(word in text_lower for word in ["create", "build", "make"]):
            return "code"
        else:
            return "general"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Remove common words
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "been", "be",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "about", "into", "through", "during", "before", "after",
            "i", "me", "my", "you", "your", "it", "its", "this", "that"
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter keywords
        keywords = []
        for word in words:
            if len(word) > 2 and word not in stop_words:
                keywords.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:10]  # Top 10 keywords
    
    def _create_sub_queries(
        self, 
        intents: List[Dict[str, Any]], 
        original_query: str
    ) -> List[SubQuery]:
        """Create SubQuery objects from intents"""
        sub_queries = []
        
        for i, intent in enumerate(intents):
            sub_query = SubQuery(
                id=f"sq_{i}",
                text=intent["text"],
                type=intent["type"],
                keywords=intent["keywords"],
                priority=self._calculate_priority(intent, i, len(intents))
            )
            sub_queries.append(sub_query)
        
        # Add synthesis query if multiple sub-queries
        if len(sub_queries) > 1:
            sub_queries.append(SubQuery(
                id="sq_synthesis",
                text=f"Synthesize results for: {original_query}",
                type="synthesis",
                keywords=["synthesize", "combine", "summarize"],
                dependencies=[sq.id for sq in sub_queries],
                priority=0  # Lowest priority (runs last)
            ))
        
        return sub_queries
    
    def _identify_dependencies(
        self, 
        sub_queries: List[SubQuery], 
        query: str
    ) -> List[SubQuery]:
        """Identify dependencies between sub-queries"""
        # Look for explicit ordering in query
        for indicator in self.dependency_indicators:
            if indicator in query:
                # Simple sequential dependency for now
                for i in range(1, len(sub_queries)):
                    if sub_queries[i].id != "sq_synthesis":
                        sub_queries[i].dependencies = [sub_queries[i-1].id]
                break
        
        # Code queries often depend on documentation queries
        doc_queries = [sq for sq in sub_queries if sq.type == "documentation"]
        code_queries = [sq for sq in sub_queries if sq.type == "code"]
        
        for code_sq in code_queries:
            for doc_sq in doc_queries:
                # If they share keywords, create dependency
                shared_keywords = set(code_sq.keywords) & set(doc_sq.keywords)
                if len(shared_keywords) >= 2:
                    if not code_sq.dependencies:
                        code_sq.dependencies = []
                    code_sq.dependencies.append(doc_sq.id)
        
        return sub_queries
    
    def _calculate_priority(
        self, 
        intent: Dict[str, Any], 
        position: int, 
        total: int
    ) -> int:
        """Calculate priority for a sub-query"""
        # Higher priority for:
        # 1. Documentation queries (often needed first)
        # 2. Queries appearing earlier in the original
        
        base_priority = 5  # Medium priority
        
        if intent["type"] == "documentation":
            base_priority += 2
        elif intent["type"] == "performance":
            base_priority += 1
        elif intent["type"] == "testing":
            base_priority -= 1  # Usually comes after implementation
        
        # Earlier position = higher priority
        position_bonus = (total - position) / total * 2
        
        return int(base_priority + position_bonus)
    
    def _calculate_complexity(self, sub_queries: List[SubQuery]) -> float:
        """Calculate query complexity score"""
        # Factors:
        # - Number of sub-queries
        # - Number of different types
        # - Dependency chains
        # - Total keywords
        
        num_queries = len(sub_queries)
        num_types = len(set(sq.type for sq in sub_queries))
        
        # Calculate dependency depth
        max_depth = 0
        for sq in sub_queries:
            depth = self._get_dependency_depth(sq, sub_queries)
            max_depth = max(max_depth, depth)
        
        total_keywords = sum(len(sq.keywords) for sq in sub_queries)
        
        # Weighted complexity score
        complexity = (
            num_queries * 0.3 +
            num_types * 0.2 +
            max_depth * 0.3 +
            (total_keywords / 10) * 0.2
        )
        
        # Normalize to 0-10 scale
        return min(10.0, max(1.0, complexity))
    
    def _get_dependency_depth(
        self, 
        sub_query: SubQuery, 
        all_queries: List[SubQuery]
    ) -> int:
        """Get the dependency depth for a sub-query"""
        if not sub_query.dependencies:
            return 0
        
        # Map for quick lookup
        query_map = {sq.id: sq for sq in all_queries}
        
        max_parent_depth = 0
        for dep_id in sub_query.dependencies:
            if dep_id in query_map:
                parent_depth = self._get_dependency_depth(
                    query_map[dep_id], 
                    all_queries
                )
                max_parent_depth = max(max_parent_depth, parent_depth)
        
        return max_parent_depth + 1
    
    def _determine_primary_type(self, sub_queries: List[SubQuery]) -> str:
        """Determine the primary type of the overall query"""
        # Count occurrences of each type
        type_counts = {}
        for sq in sub_queries:
            if sq.type != "synthesis":  # Exclude synthesis
                type_counts[sq.type] = type_counts.get(sq.type, 0) + 1
        
        if type_counts:
            return max(type_counts, key=type_counts.get)
        return "general"
    
    def _serialize_sub_query(self, sub_query: SubQuery) -> Dict[str, Any]:
        """Serialize SubQuery for API response"""
        return {
            "id": sub_query.id,
            "text": sub_query.text,
            "type": sub_query.type,
            "keywords": sub_query.keywords,
            "dependencies": sub_query.dependencies or [],
            "priority": sub_query.priority
        }