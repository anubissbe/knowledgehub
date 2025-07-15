"""Correction Processor Service

This service specializes in processing user corrections, extracting patterns,
and applying learned corrections to future responses.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from uuid import UUID, uuid4
from difflib import SequenceMatcher
import json

from sqlalchemy.orm import Session
from sqlalchemy import select, and_, func
from pydantic import BaseModel, Field

from ..models.user_feedback import UserFeedback, FeedbackType
from ..models.learning_pattern import LearningPattern, PatternType
from ...memory_system.models.memory import Memory
from ...services.cache import redis_client

logger = logging.getLogger(__name__)


from enum import Enum

class CorrectionType(str, Enum):
    """Types of corrections"""
    SPELLING = "spelling"
    GRAMMAR = "grammar"
    FACTUAL = "factual"
    STYLE = "style"
    TECHNICAL = "technical"
    EXPANSION = "expansion"
    CLARIFICATION = "clarification"
    FORMATTING = "formatting"


class CorrectionPattern(BaseModel):
    """A pattern extracted from corrections"""
    pattern_id: UUID = Field(default_factory=uuid4)
    correction_type: str
    original_pattern: str
    corrected_pattern: str
    context_clues: List[str] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    frequency: int = 1
    examples: List[Dict[str, str]] = Field(default_factory=list)


class CorrectionResult(BaseModel):
    """Result of applying corrections"""
    original_text: str
    corrected_text: str
    corrections_applied: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CorrectionProcessor:
    """Advanced correction processing and pattern extraction"""
    
    def __init__(self, db: Session):
        """Initialize the correction processor"""
        self.db = db
        
        # Configuration
        self.min_similarity_threshold = 0.7
        self.pattern_confidence_threshold = 0.6
        self.max_pattern_cache_size = 1000
        
        # Pattern cache for fast lookup
        self._pattern_cache = {}
        self._correction_rules = self._initialize_correction_rules()
    
    async def process_correction(
        self,
        original: str,
        corrected: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a user correction and extract patterns
        
        Args:
            original: Original text
            corrected: Corrected text
            context: Optional context information
            
        Returns:
            Processing results including extracted patterns
        """
        try:
            # Analyze the correction
            analysis = await self._analyze_correction(original, corrected)
            
            # Extract patterns
            patterns = await self._extract_correction_patterns(
                original, corrected, analysis, context
            )
            
            # Store patterns
            stored_patterns = []
            for pattern in patterns:
                stored = await self._store_correction_pattern(pattern)
                stored_patterns.append(stored)
            
            # Update pattern cache
            await self._update_pattern_cache(stored_patterns)
            
            # Generate learning insights
            insights = await self._generate_correction_insights(analysis, patterns)
            
            return {
                "analysis": analysis,
                "patterns_extracted": len(patterns),
                "patterns": [p.dict() for p in patterns],
                "insights": insights,
                "correction_type": analysis['primary_type']
            }
            
        except Exception as e:
            logger.error(f"Error processing correction: {e}")
            return {
                "error": str(e),
                "patterns_extracted": 0
            }
    
    async def apply_learned_corrections(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> CorrectionResult:
        """Apply learned correction patterns to text
        
        Args:
            text: Text to potentially correct
            context: Optional context for better corrections
            
        Returns:
            Correction result with applied changes
        """
        try:
            # Get applicable patterns
            patterns = await self._find_applicable_patterns(text, context)
            
            # Sort by confidence and priority
            patterns.sort(key=lambda p: (p.confidence, p.frequency), reverse=True)
            
            # Apply corrections
            corrected_text = text
            corrections_applied = []
            
            for pattern in patterns:
                result = self._apply_pattern(corrected_text, pattern)
                if result['applied']:
                    corrected_text = result['text']
                    corrections_applied.append({
                        "pattern_id": str(pattern.pattern_id),
                        "type": pattern.correction_type,
                        "original": result['original_segment'],
                        "corrected": result['corrected_segment'],
                        "confidence": pattern.confidence
                    })
            
            # Calculate overall confidence
            overall_confidence = self._calculate_correction_confidence(
                text, corrected_text, corrections_applied
            )
            
            return CorrectionResult(
                original_text=text,
                corrected_text=corrected_text,
                corrections_applied=corrections_applied,
                confidence=overall_confidence,
                metadata={
                    "patterns_checked": len(patterns),
                    "corrections_made": len(corrections_applied)
                }
            )
            
        except Exception as e:
            logger.error(f"Error applying corrections: {e}")
            return CorrectionResult(
                original_text=text,
                corrected_text=text,
                corrections_applied=[],
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    async def get_correction_statistics(
        self,
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """Get statistics about corrections
        
        Args:
            time_period_days: Days to look back
            
        Returns:
            Correction statistics
        """
        try:
            # Get correction feedback
            corrections = await self._get_recent_corrections(time_period_days)
            
            # Calculate statistics
            stats = {
                "total_corrections": len(corrections),
                "correction_types": {},
                "most_common_patterns": [],
                "improvement_rate": 0.0,
                "average_similarity": 0.0,
                "top_corrected_words": [],
                "correction_complexity": {}
            }
            
            if not corrections:
                return stats
            
            # Analyze corrections
            type_counts = {}
            similarities = []
            word_corrections = {}
            
            for correction in corrections:
                # Analyze type
                analysis = await self._analyze_correction(
                    correction.original_content,
                    correction.corrected_content
                )
                
                correction_type = analysis['primary_type']
                type_counts[correction_type] = type_counts.get(correction_type, 0) + 1
                
                # Track similarity
                similarities.append(analysis['similarity'])
                
                # Extract word-level corrections
                word_changes = self._extract_word_changes(
                    correction.original_content,
                    correction.corrected_content
                )
                for original, corrected in word_changes:
                    key = f"{original} → {corrected}"
                    word_corrections[key] = word_corrections.get(key, 0) + 1
            
            # Fill statistics
            stats["correction_types"] = type_counts
            stats["average_similarity"] = sum(similarities) / len(similarities)
            
            # Top corrected words
            sorted_words = sorted(
                word_corrections.items(),
                key=lambda x: x[1],
                reverse=True
            )
            stats["top_corrected_words"] = [
                {"correction": word, "count": count}
                for word, count in sorted_words[:10]
            ]
            
            # Get most common patterns
            patterns = await self._get_most_common_patterns(10)
            stats["most_common_patterns"] = [
                {
                    "pattern": p.original_pattern + " → " + p.corrected_pattern,
                    "type": p.correction_type,
                    "frequency": p.frequency,
                    "confidence": p.confidence
                }
                for p in patterns
            ]
            
            # Calculate improvement rate
            stats["improvement_rate"] = await self._calculate_improvement_rate(corrections)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting correction statistics: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    def _initialize_correction_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize common correction rules"""
        return {
            CorrectionType.TECHNICAL: [
                {
                    "pattern": r'\bsync\s+def\b',
                    "replacement": "async def",
                    "description": "Missing async keyword"
                },
                {
                    "pattern": r'\.append\s*\(\s*\)',
                    "replacement": ".append(item)",
                    "description": "Empty append call"
                }
            ],
            CorrectionType.SPELLING: [
                # Common misspellings would be loaded from a dictionary
            ],
            CorrectionType.STYLE: [
                {
                    "pattern": r'if\s+\w+\s*==\s*True',
                    "replacement": "if {var}",
                    "description": "Redundant boolean comparison"
                }
            ]
        }
    
    async def _analyze_correction(
        self,
        original: str,
        corrected: str
    ) -> Dict[str, Any]:
        """Analyze a correction to determine its type and characteristics"""
        analysis = {
            "primary_type": CorrectionType.TECHNICAL,
            "similarity": 0.0,
            "changes": [],
            "complexity": "simple"
        }
        
        # Calculate similarity
        matcher = SequenceMatcher(None, original, corrected)
        analysis["similarity"] = matcher.ratio()
        
        # Determine correction type
        if len(corrected) > len(original) * 1.5:
            analysis["primary_type"] = CorrectionType.EXPANSION
        elif len(corrected) < len(original) * 0.7:
            analysis["primary_type"] = CorrectionType.CLARIFICATION
        else:
            # Check for specific patterns
            if self._is_spelling_correction(original, corrected):
                analysis["primary_type"] = CorrectionType.SPELLING
            elif self._is_grammar_correction(original, corrected):
                analysis["primary_type"] = CorrectionType.GRAMMAR
            elif self._is_technical_correction(original, corrected):
                analysis["primary_type"] = CorrectionType.TECHNICAL
            elif self._is_style_correction(original, corrected):
                analysis["primary_type"] = CorrectionType.STYLE
            else:
                analysis["primary_type"] = CorrectionType.FACTUAL
        
        # Extract specific changes
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                analysis["changes"].append({
                    "type": tag,
                    "original": original[i1:i2],
                    "corrected": corrected[j1:j2],
                    "position": i1
                })
        
        # Determine complexity
        if len(analysis["changes"]) > 5:
            analysis["complexity"] = "complex"
        elif len(analysis["changes"]) > 2:
            analysis["complexity"] = "moderate"
        
        return analysis
    
    async def _extract_correction_patterns(
        self,
        original: str,
        corrected: str,
        analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[CorrectionPattern]:
        """Extract reusable patterns from a correction"""
        patterns = []
        
        # Extract change-based patterns
        for change in analysis["changes"]:
            if change["type"] in ["replace", "delete", "insert"]:
                pattern = CorrectionPattern(
                    correction_type=analysis["primary_type"],
                    original_pattern=change["original"],
                    corrected_pattern=change["corrected"],
                    context_clues=self._extract_context_clues(
                        original, change["position"]
                    ),
                    confidence=0.5,  # Initial confidence
                    examples=[{
                        "original": original,
                        "corrected": corrected
                    }]
                )
                patterns.append(pattern)
        
        # Extract regex-based patterns
        if analysis["primary_type"] == CorrectionType.TECHNICAL:
            tech_patterns = self._extract_technical_patterns(original, corrected)
            patterns.extend(tech_patterns)
        
        # Extract semantic patterns
        if context:
            semantic_patterns = await self._extract_semantic_patterns(
                original, corrected, context
            )
            patterns.extend(semantic_patterns)
        
        return patterns
    
    def _extract_context_clues(
        self,
        text: str,
        position: int,
        window: int = 20
    ) -> List[str]:
        """Extract context clues around a change position"""
        clues = []
        
        # Get surrounding text
        start = max(0, position - window)
        end = min(len(text), position + window)
        context_text = text[start:end]
        
        # Extract word-level context
        words = context_text.split()
        if words:
            clues.extend(words[:3])  # First few words
        
        # Extract patterns
        if "async" in context_text:
            clues.append("async_context")
        if "def" in context_text:
            clues.append("function_context")
        if "class" in context_text:
            clues.append("class_context")
        
        return clues
    
    def _extract_technical_patterns(
        self,
        original: str,
        corrected: str
    ) -> List[CorrectionPattern]:
        """Extract technical correction patterns"""
        patterns = []
        
        # Check for async/await patterns
        if "async" in corrected and "async" not in original:
            if "def" in original:
                pattern = CorrectionPattern(
                    correction_type=CorrectionType.TECHNICAL,
                    original_pattern=r"^\s*def\s+",
                    corrected_pattern="async def ",
                    context_clues=["function_definition"],
                    confidence=0.9
                )
                patterns.append(pattern)
        
        # Check for type hint additions
        if "->" in corrected and "->" not in original:
            pattern = CorrectionPattern(
                correction_type=CorrectionType.TECHNICAL,
                original_pattern=r"\):\s*$",
                corrected_pattern=") -> ReturnType:",
                context_clues=["function_signature"],
                confidence=0.7
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _extract_semantic_patterns(
        self,
        original: str,
        corrected: str,
        context: Dict[str, Any]
    ) -> List[CorrectionPattern]:
        """Extract semantic correction patterns"""
        patterns = []
        
        # This would use NLP to extract semantic patterns
        # For now, simple implementation
        
        if context.get("intent") == "code_generation":
            if "return" in corrected and "return" not in original:
                pattern = CorrectionPattern(
                    correction_type=CorrectionType.TECHNICAL,
                    original_pattern="end_of_function",
                    corrected_pattern="return result",
                    context_clues=["missing_return"],
                    confidence=0.6
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _store_correction_pattern(
        self,
        pattern: CorrectionPattern
    ) -> CorrectionPattern:
        """Store or update a correction pattern"""
        # Check if similar pattern exists
        existing = await self._find_similar_pattern(pattern)
        
        if existing:
            # Update existing pattern
            existing.frequency += 1
            existing.confidence = min(
                1.0,
                existing.confidence + 0.05  # Increase confidence
            )
            existing.examples.extend(pattern.examples)
            # Keep only recent examples
            existing.examples = existing.examples[-10:]
            
            return existing
        else:
            # Store new pattern
            # In production, this would store to database
            self._pattern_cache[pattern.pattern_id] = pattern
            return pattern
    
    async def _find_similar_pattern(
        self,
        pattern: CorrectionPattern
    ) -> Optional[CorrectionPattern]:
        """Find similar existing pattern"""
        for cached_pattern in self._pattern_cache.values():
            if (
                cached_pattern.correction_type == pattern.correction_type and
                cached_pattern.original_pattern == pattern.original_pattern and
                cached_pattern.corrected_pattern == pattern.corrected_pattern
            ):
                return cached_pattern
        return None
    
    async def _update_pattern_cache(self, patterns: List[CorrectionPattern]):
        """Update the pattern cache"""
        for pattern in patterns:
            self._pattern_cache[pattern.pattern_id] = pattern
        
        # Limit cache size
        if len(self._pattern_cache) > self.max_pattern_cache_size:
            # Remove least frequently used patterns
            sorted_patterns = sorted(
                self._pattern_cache.items(),
                key=lambda x: (x[1].frequency, x[1].confidence)
            )
            # Keep top patterns
            self._pattern_cache = dict(
                sorted_patterns[-self.max_pattern_cache_size:]
            )
    
    async def _generate_correction_insights(
        self,
        analysis: Dict[str, Any],
        patterns: List[CorrectionPattern]
    ) -> Dict[str, Any]:
        """Generate insights from correction analysis"""
        insights = {
            "correction_type": analysis["primary_type"],
            "complexity": analysis["complexity"],
            "patterns_found": len(patterns),
            "recommendations": []
        }
        
        # Generate recommendations based on correction type
        if analysis["primary_type"] == CorrectionType.TECHNICAL:
            insights["recommendations"].append(
                "Review technical accuracy, especially async/await patterns"
            )
        elif analysis["primary_type"] == CorrectionType.EXPANSION:
            insights["recommendations"].append(
                "Provide more detailed and comprehensive responses"
            )
        elif analysis["primary_type"] == CorrectionType.STYLE:
            insights["recommendations"].append(
                "Follow established coding style guidelines"
            )
        
        # Add pattern-specific recommendations
        for pattern in patterns:
            if pattern.confidence > 0.7:
                insights["recommendations"].append(
                    f"Apply pattern: {pattern.original_pattern} → {pattern.corrected_pattern}"
                )
        
        return insights
    
    async def _find_applicable_patterns(
        self,
        text: str,
        context: Optional[Dict[str, Any]]
    ) -> List[CorrectionPattern]:
        """Find patterns applicable to the given text"""
        applicable = []
        
        for pattern in self._pattern_cache.values():
            if pattern.confidence < self.pattern_confidence_threshold:
                continue
            
            # Check if pattern applies
            if self._pattern_applies(text, pattern, context):
                applicable.append(pattern)
        
        return applicable
    
    def _pattern_applies(
        self,
        text: str,
        pattern: CorrectionPattern,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if a pattern applies to text"""
        # Check for literal match
        if pattern.original_pattern in text:
            return True
        
        # Check for regex match
        try:
            if re.search(pattern.original_pattern, text):
                return True
        except re.error:
            pass
        
        # Check context clues
        if context and pattern.context_clues:
            context_matches = any(
                clue in str(context).lower()
                for clue in pattern.context_clues
            )
            if context_matches:
                return True
        
        return False
    
    def _apply_pattern(
        self,
        text: str,
        pattern: CorrectionPattern
    ) -> Dict[str, Any]:
        """Apply a correction pattern to text"""
        result = {
            "applied": False,
            "text": text,
            "original_segment": "",
            "corrected_segment": ""
        }
        
        # Try literal replacement
        if pattern.original_pattern in text:
            result["applied"] = True
            result["original_segment"] = pattern.original_pattern
            result["corrected_segment"] = pattern.corrected_pattern
            result["text"] = text.replace(
                pattern.original_pattern,
                pattern.corrected_pattern,
                1  # Replace only first occurrence
            )
            return result
        
        # Try regex replacement
        try:
            match = re.search(pattern.original_pattern, text)
            if match:
                result["applied"] = True
                result["original_segment"] = match.group(0)
                result["corrected_segment"] = pattern.corrected_pattern
                result["text"] = re.sub(
                    pattern.original_pattern,
                    pattern.corrected_pattern,
                    text,
                    count=1
                )
        except re.error:
            pass
        
        return result
    
    def _calculate_correction_confidence(
        self,
        original: str,
        corrected: str,
        corrections: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence in corrections"""
        if not corrections:
            return 1.0  # No corrections needed
        
        # Average confidence of applied corrections
        confidences = [c["confidence"] for c in corrections]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Adjust based on text similarity
        similarity = SequenceMatcher(None, original, corrected).ratio()
        
        # Higher similarity with corrections indicates higher confidence
        return avg_confidence * (0.5 + similarity * 0.5)
    
    def _is_spelling_correction(self, original: str, corrected: str) -> bool:
        """Check if correction is primarily spelling"""
        # Simple heuristic: high similarity with small changes
        similarity = SequenceMatcher(None, original, corrected).ratio()
        return similarity > 0.9 and len(original) == len(corrected)
    
    def _is_grammar_correction(self, original: str, corrected: str) -> bool:
        """Check if correction is primarily grammar"""
        # Check for common grammar patterns
        grammar_indicators = [
            ("a ", "an "),
            ("an ", "a "),
            (" is ", " are "),
            (" are ", " is "),
            (" was ", " were "),
            (" were ", " was ")
        ]
        
        for orig, corr in grammar_indicators:
            if orig in original and corr in corrected:
                return True
        
        return False
    
    def _is_technical_correction(self, original: str, corrected: str) -> bool:
        """Check if correction is technical"""
        technical_keywords = [
            "async", "await", "def", "class", "import",
            "return", "yield", "lambda", "->", "self"
        ]
        
        # Check if technical keywords are added/modified
        for keyword in technical_keywords:
            if keyword in corrected and keyword not in original:
                return True
        
        return False
    
    def _is_style_correction(self, original: str, corrected: str) -> bool:
        """Check if correction is style-related"""
        # Check for formatting changes
        if original.strip() == corrected.strip():
            return True
        
        # Check for quote style changes
        if original.replace('"', "'") == corrected or original.replace("'", '"') == corrected:
            return True
        
        return False
    
    def _extract_word_changes(
        self,
        original: str,
        corrected: str
    ) -> List[Tuple[str, str]]:
        """Extract word-level changes"""
        changes = []
        
        original_words = original.split()
        corrected_words = corrected.split()
        
        matcher = SequenceMatcher(None, original_words, corrected_words)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                for i in range(i2 - i1):
                    if i1 + i < len(original_words) and j1 + i < len(corrected_words):
                        changes.append((
                            original_words[i1 + i],
                            corrected_words[j1 + i]
                        ))
        
        return changes
    
    async def _get_recent_corrections(
        self,
        days: int
    ) -> List[UserFeedback]:
        """Get recent correction feedback"""
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        result = await self.db.execute(
            select(UserFeedback).where(
                and_(
                    UserFeedback.feedback_type == FeedbackType.CORRECTION.value,
                    UserFeedback.created_at >= start_date
                )
            )
        )
        
        return result.scalars().all()
    
    async def _get_most_common_patterns(
        self,
        limit: int
    ) -> List[CorrectionPattern]:
        """Get most commonly used patterns"""
        # Sort patterns by frequency and confidence
        sorted_patterns = sorted(
            self._pattern_cache.values(),
            key=lambda p: (p.frequency, p.confidence),
            reverse=True
        )
        
        return sorted_patterns[:limit]
    
    async def _calculate_improvement_rate(
        self,
        corrections: List[UserFeedback]
    ) -> float:
        """Calculate improvement rate based on corrections"""
        if len(corrections) < 10:
            return 0.0
        
        # Sort by date
        sorted_corrections = sorted(corrections, key=lambda c: c.created_at)
        
        # Compare first and last quarters
        quarter_size = len(sorted_corrections) // 4
        first_quarter = sorted_corrections[:quarter_size]
        last_quarter = sorted_corrections[-quarter_size:]
        
        # Calculate average complexity for each quarter
        first_complexity = 0
        last_complexity = 0
        
        for correction in first_quarter:
            analysis = await self._analyze_correction(
                correction.original_content,
                correction.corrected_content
            )
            first_complexity += len(analysis["changes"])
        
        for correction in last_quarter:
            analysis = await self._analyze_correction(
                correction.original_content,
                correction.corrected_content
            )
            last_complexity += len(analysis["changes"])
        
        # Lower complexity in recent corrections indicates improvement
        if first_complexity > 0:
            improvement = 1.0 - (last_complexity / first_complexity)
            return max(0.0, min(1.0, improvement))
        
        return 0.0