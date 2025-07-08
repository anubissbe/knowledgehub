"""
Importance scoring algorithm for memory prioritization.

This service analyzes memories, facts, and conversation content to calculate
importance scores that help prioritize which information should be retained,
retrieved, and emphasized in the memory system.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import math

logger = logging.getLogger(__name__)


class ImportanceFactors(Enum):
    """Factors that contribute to importance scoring"""
    REPETITION = "repetition"  # How often something is mentioned
    RECENCY = "recency"  # How recent the information is
    EXPLICITNESS = "explicitness"  # Explicit importance markers
    USER_EMPHASIS = "user_emphasis"  # User explicitly emphasizes something
    CONTEXTUAL_RELEVANCE = "contextual_relevance"  # Relevance to current context
    TECHNICAL_COMPLEXITY = "technical_complexity"  # Technical depth/complexity
    DECISION_WEIGHT = "decision_weight"  # Decisions are inherently important
    ACTION_URGENCY = "action_urgency"  # Urgency of action items
    ERROR_SEVERITY = "error_severity"  # Severity of errors/issues
    ENTITY_DENSITY = "entity_density"  # Number of important entities
    TEMPORAL_PROXIMITY = "temporal_proximity"  # Time-sensitive information
    FREQUENCY_PATTERN = "frequency_pattern"  # Patterns in frequency of mention


@dataclass
class ImportanceScore:
    """Represents an importance score with breakdown"""
    total_score: float
    factor_scores: Dict[ImportanceFactors, float]
    confidence: float
    reasoning: List[str]
    metadata: Dict[str, Any]
    normalized_score: float = None
    
    def __post_init__(self):
        if self.normalized_score is None:
            self.normalized_score = min(1.0, max(0.0, self.total_score))


class ImportancePatterns:
    """Patterns that indicate importance"""
    
    # Explicit importance indicators
    EXPLICIT_IMPORTANCE = [
        r'\b(?:important|critical|crucial|essential|vital|key|significant|major)\b',
        r'\b(?:must|required|necessary|urgent|priority|high.priority)\b',
        r'\b(?:note|remember|don\'t forget|keep in mind|pay attention)\b',
        r'\b(?:warning|caution|attention|alert)\b'
    ]
    
    # User emphasis patterns
    USER_EMPHASIS = [
        r'\b(?:I|we)\s+(?:really|definitely|absolutely|strongly)\s+(?:need|want|think|believe)\b',
        r'\b(?:this is|that\'s|it\'s)\s+(?:very|extremely|really|quite)\s+(?:important|significant)\b',
        r'\*\*([^*]+)\*\*',  # Bold text
        r'__([^_]+)__',  # Underlined text
        r'!!+',  # Multiple exclamation marks
        r'(?:TODO|FIXME|IMPORTANT|NOTE):\s*',  # Explicit markers
    ]
    
    # Technical complexity indicators
    TECHNICAL_COMPLEXITY = [
        r'\b(?:algorithm|architecture|implementation|optimization|performance)\b',
        r'\b(?:database|query|index|transaction|concurrency)\b',
        r'\b(?:security|authentication|authorization|encryption)\b',
        r'\b(?:scalability|reliability|availability|consistency)\b',
        r'\b(?:api|endpoint|protocol|framework|library)\b'
    ]
    
    # Time-sensitive indicators
    TIME_SENSITIVE = [
        r'\b(?:deadline|due|expire|urgent|asap|immediately)\b',
        r'\b(?:today|tomorrow|this week|next week|by \w+)\b',
        r'\b(?:before|after|during|while|until)\s+\w+\b',
        r'\b(?:schedule|timeline|milestone|release)\b'
    ]
    
    # Decision indicators
    DECISION_INDICATORS = [
        r'\b(?:decided|chose|selected|picked|determined)\b',
        r'\b(?:decision|choice|selection|option)\b',
        r'\b(?:will|shall|going to|plan to)\s+(?:use|implement|adopt)\b',
        r'\b(?:approved|accepted|agreed|confirmed)\b'
    ]
    
    # Error severity indicators
    ERROR_SEVERITY = [
        r'\b(?:critical|fatal|severe|major)\s+(?:error|issue|problem|bug)\b',
        r'\b(?:crash|failure|exception|stack trace)\b',
        r'\b(?:security|vulnerability|exploit|breach)\b',
        r'\b(?:data loss|corruption|integrity)\b'
    ]


class IntelligentImportanceScorer:
    """
    Intelligent importance scoring service that analyzes content and context
    to determine the relative importance of memories and information.
    """
    
    def __init__(self):
        self.patterns = ImportancePatterns()
        
        # Compile regex patterns for performance
        self.compiled_patterns = {
            ImportanceFactors.EXPLICITNESS: [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in self.patterns.EXPLICIT_IMPORTANCE
            ],
            ImportanceFactors.USER_EMPHASIS: [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in self.patterns.USER_EMPHASIS
            ],
            ImportanceFactors.TECHNICAL_COMPLEXITY: [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in self.patterns.TECHNICAL_COMPLEXITY
            ],
            ImportanceFactors.ACTION_URGENCY: [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in self.patterns.TIME_SENSITIVE
            ],
            ImportanceFactors.DECISION_WEIGHT: [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in self.patterns.DECISION_INDICATORS
            ],
            ImportanceFactors.ERROR_SEVERITY: [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in self.patterns.ERROR_SEVERITY
            ]
        }
        
        # Weight configuration for different factors
        self.factor_weights = {
            ImportanceFactors.EXPLICITNESS: 0.25,
            ImportanceFactors.USER_EMPHASIS: 0.20,
            ImportanceFactors.ERROR_SEVERITY: 0.18,
            ImportanceFactors.DECISION_WEIGHT: 0.15,
            ImportanceFactors.ACTION_URGENCY: 0.15,
            ImportanceFactors.TEMPORAL_PROXIMITY: 0.10,
            ImportanceFactors.TECHNICAL_COMPLEXITY: 0.08,
            ImportanceFactors.CONTEXTUAL_RELEVANCE: 0.08,
            ImportanceFactors.REPETITION: 0.05,
            ImportanceFactors.RECENCY: 0.05,
            ImportanceFactors.ENTITY_DENSITY: 0.03,
            ImportanceFactors.FREQUENCY_PATTERN: 0.03
        }
    
    async def calculate_importance(self, content: str, context: Dict[str, Any] = None, 
                                 entities: List[str] = None, facts: List[Dict[str, Any]] = None,
                                 history: List[Dict[str, Any]] = None) -> ImportanceScore:
        """
        Calculate comprehensive importance score for content.
        
        Args:
            content: Text content to score
            context: Additional context information
            entities: Extracted entities from the content
            facts: Extracted facts from the content
            history: Historical content for repetition analysis
            
        Returns:
            ImportanceScore object with detailed breakdown
        """
        if not content or not content.strip():
            return ImportanceScore(
                total_score=0.0,
                factor_scores={},
                confidence=1.0,
                reasoning=["Empty content has no importance"],
                metadata={}
            )
        
        context = context or {}
        entities = entities or []
        facts = facts or []
        history = history or []
        
        logger.info(f"Calculating importance for {len(content)} characters of content")
        
        # Calculate individual factor scores
        factor_scores = {}
        reasoning = []
        
        # Pattern-based factors
        factor_scores[ImportanceFactors.EXPLICITNESS] = self._calculate_explicitness_score(content, reasoning)
        factor_scores[ImportanceFactors.USER_EMPHASIS] = self._calculate_user_emphasis_score(content, reasoning)
        factor_scores[ImportanceFactors.TECHNICAL_COMPLEXITY] = self._calculate_technical_complexity_score(content, reasoning)
        factor_scores[ImportanceFactors.ACTION_URGENCY] = self._calculate_action_urgency_score(content, reasoning)
        factor_scores[ImportanceFactors.DECISION_WEIGHT] = self._calculate_decision_weight_score(content, reasoning)
        factor_scores[ImportanceFactors.ERROR_SEVERITY] = self._calculate_error_severity_score(content, reasoning)
        
        # Context-based factors
        factor_scores[ImportanceFactors.CONTEXTUAL_RELEVANCE] = self._calculate_contextual_relevance_score(content, context, reasoning)
        factor_scores[ImportanceFactors.RECENCY] = self._calculate_recency_score(context, reasoning)
        factor_scores[ImportanceFactors.TEMPORAL_PROXIMITY] = self._calculate_temporal_proximity_score(content, reasoning)
        
        # Entity and fact-based factors
        factor_scores[ImportanceFactors.ENTITY_DENSITY] = self._calculate_entity_density_score(content, entities, reasoning)
        
        # Historical factors
        factor_scores[ImportanceFactors.REPETITION] = self._calculate_repetition_score(content, history, reasoning)
        factor_scores[ImportanceFactors.FREQUENCY_PATTERN] = self._calculate_frequency_pattern_score(content, history, reasoning)
        
        # Calculate weighted total score
        total_score = 0.0
        for factor, score in factor_scores.items():
            weight = self.factor_weights.get(factor, 0.0)
            total_score += score * weight
        
        # Calculate confidence based on available information
        confidence = self._calculate_confidence(content, context, entities, facts, history)
        
        # Add metadata
        metadata = {
            'content_length': len(content),
            'word_count': len(content.split()),
            'entity_count': len(entities),
            'fact_count': len(facts),
            'has_context': bool(context),
            'has_history': bool(history),
            'calculation_factors': len([f for f, s in factor_scores.items() if s > 0])
        }
        
        # Enhance reasoning with top contributing factors
        top_factors = sorted(
            [(factor, score * self.factor_weights.get(factor, 0)) 
             for factor, score in factor_scores.items()], 
            key=lambda x: x[1], reverse=True
        )[:3]
        
        if top_factors:
            reasoning.append(f"Top contributing factors: {', '.join([f.value for f, _ in top_factors])}")
        
        return ImportanceScore(
            total_score=total_score,
            factor_scores=factor_scores,
            confidence=confidence,
            reasoning=reasoning,
            metadata=metadata
        )
    
    def _calculate_explicitness_score(self, content: str, reasoning: List[str]) -> float:
        """Calculate score based on explicit importance markers"""
        score = 0.0
        matches = 0
        
        patterns = self.compiled_patterns.get(ImportanceFactors.EXPLICITNESS, [])
        for pattern in patterns:
            pattern_matches = len(pattern.findall(content))
            matches += pattern_matches
            if pattern_matches > 0:
                score += min(0.3, pattern_matches * 0.1)
        
        if matches > 0:
            reasoning.append(f"Found {matches} explicit importance markers")
        
        return min(1.0, score)
    
    def _calculate_user_emphasis_score(self, content: str, reasoning: List[str]) -> float:
        """Calculate score based on user emphasis patterns"""
        score = 0.0
        emphasis_count = 0
        
        patterns = self.compiled_patterns.get(ImportanceFactors.USER_EMPHASIS, [])
        for pattern in patterns:
            pattern_matches = len(pattern.findall(content))
            emphasis_count += pattern_matches
            if pattern_matches > 0:
                score += min(0.25, pattern_matches * 0.1)
        
        # Check for caps (but not excessive)
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', content))
        if caps_words > 0 and caps_words < 5:  # Not all caps
            score += min(0.1, caps_words * 0.02)
            emphasis_count += caps_words
        
        if emphasis_count > 0:
            reasoning.append(f"Found {emphasis_count} user emphasis indicators")
        
        return min(1.0, score)
    
    def _calculate_technical_complexity_score(self, content: str, reasoning: List[str]) -> float:
        """Calculate score based on technical complexity"""
        score = 0.0
        tech_terms = 0
        
        patterns = self.compiled_patterns.get(ImportanceFactors.TECHNICAL_COMPLEXITY, [])
        for pattern in patterns:
            pattern_matches = len(pattern.findall(content))
            tech_terms += pattern_matches
            if pattern_matches > 0:
                score += min(0.2, pattern_matches * 0.05)
        
        # Check for code patterns
        code_patterns = [
            r'```[\s\S]*?```',  # Code blocks
            r'`[^`]+`',  # Inline code
            r'\b\w+\(\)',  # Function calls
            r'\b[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*\b'  # CamelCase
        ]
        
        for pattern in code_patterns:
            code_matches = len(re.findall(pattern, content))
            if code_matches > 0:
                score += min(0.15, code_matches * 0.03)
                tech_terms += code_matches
        
        if tech_terms > 0:
            reasoning.append(f"Found {tech_terms} technical complexity indicators")
        
        return min(1.0, score)
    
    def _calculate_action_urgency_score(self, content: str, reasoning: List[str]) -> float:
        """Calculate score based on action urgency"""
        score = 0.0
        urgent_indicators = 0
        
        patterns = self.compiled_patterns.get(ImportanceFactors.ACTION_URGENCY, [])
        for pattern in patterns:
            pattern_matches = len(pattern.findall(content))
            urgent_indicators += pattern_matches
            if pattern_matches > 0:
                score += min(0.3, pattern_matches * 0.1)
        
        # Check for action words
        action_words = ['must', 'need', 'should', 'required', 'todo', 'fixme']
        for word in action_words:
            if word.lower() in content.lower():
                score += 0.05
                urgent_indicators += 1
        
        if urgent_indicators > 0:
            reasoning.append(f"Found {urgent_indicators} urgency indicators")
        
        return min(1.0, score)
    
    def _calculate_decision_weight_score(self, content: str, reasoning: List[str]) -> float:
        """Calculate score based on decision content"""
        score = 0.0
        decision_indicators = 0
        
        patterns = self.compiled_patterns.get(ImportanceFactors.DECISION_WEIGHT, [])
        for pattern in patterns:
            pattern_matches = len(pattern.findall(content))
            decision_indicators += pattern_matches
            if pattern_matches > 0:
                score += min(0.4, pattern_matches * 0.15)
        
        if decision_indicators > 0:
            reasoning.append(f"Found {decision_indicators} decision indicators")
        
        return min(1.0, score)
    
    def _calculate_error_severity_score(self, content: str, reasoning: List[str]) -> float:
        """Calculate score based on error severity"""
        score = 0.0
        error_indicators = 0
        
        patterns = self.compiled_patterns.get(ImportanceFactors.ERROR_SEVERITY, [])
        for pattern in patterns:
            pattern_matches = len(pattern.findall(content))
            error_indicators += pattern_matches
            if pattern_matches > 0:
                score += min(0.4, pattern_matches * 0.2)
        
        # Check for error-related keywords
        error_keywords = ['error', 'exception', 'failure', 'bug', 'issue', 'problem']
        for keyword in error_keywords:
            if keyword.lower() in content.lower():
                score += 0.03
                error_indicators += 1
        
        if error_indicators > 0:
            reasoning.append(f"Found {error_indicators} error severity indicators")
        
        return min(1.0, score)
    
    def _calculate_contextual_relevance_score(self, content: str, context: Dict[str, Any], reasoning: List[str]) -> float:
        """Calculate score based on contextual relevance"""
        score = 0.5  # Base relevance
        
        if not context:
            return score
        
        # Project context
        if context.get('project_id') or context.get('project'):
            score += 0.1
            reasoning.append("Content has project context")
        
        # Session context
        if context.get('session_id'):
            score += 0.05
        
        # User context
        if context.get('user_id'):
            score += 0.05
        
        # Topic relevance
        if context.get('topic') or context.get('category'):
            score += 0.1
            reasoning.append("Content has topic/category context")
        
        # Critical context
        if context.get('is_critical', False):
            score += 0.2
            reasoning.append("Marked as critical content")
        
        return min(1.0, score)
    
    def _calculate_recency_score(self, context: Dict[str, Any], reasoning: List[str]) -> float:
        """Calculate score based on recency"""
        if not context.get('timestamp') and not context.get('created_at'):
            return 0.5  # Neutral if no timestamp
        
        try:
            # Try to parse timestamp
            timestamp_str = context.get('timestamp') or context.get('created_at')
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = timestamp_str
            
            now = datetime.now(timestamp.tzinfo) if timestamp.tzinfo else datetime.now()
            age = now - timestamp
            
            # Score based on age (higher for more recent)
            if age.total_seconds() < 3600:  # Less than 1 hour
                score = 1.0
                reasoning.append("Very recent content (< 1 hour)")
            elif age.total_seconds() < 86400:  # Less than 1 day
                score = 0.8
                reasoning.append("Recent content (< 1 day)")
            elif age.days < 7:  # Less than 1 week
                score = 0.6
                reasoning.append("Relatively recent content (< 1 week)")
            elif age.days < 30:  # Less than 1 month
                score = 0.4
                reasoning.append("Moderately recent content (< 1 month)")
            else:
                score = 0.2
                reasoning.append("Older content (> 1 month)")
            
            return score
            
        except Exception as e:
            logger.warning(f"Could not parse timestamp for recency calculation: {e}")
            return 0.5
    
    def _calculate_temporal_proximity_score(self, content: str, reasoning: List[str]) -> float:
        """Calculate score based on temporal proximity indicators"""
        score = 0.0
        temporal_indicators = 0
        
        # Look for time-sensitive content
        time_patterns = [
            r'\b(?:today|tomorrow|yesterday|now|currently|at the moment)\b',
            r'\b(?:this|next|last)\s+(?:week|month|year|quarter)\b',
            r'\b(?:deadline|due|expire|schedule)\b',
            r'\b(?:by|before|after|until|during)\s+\w+\b'
        ]
        
        for pattern in time_patterns:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            if matches > 0:
                score += min(0.2, matches * 0.1)
                temporal_indicators += matches
        
        if temporal_indicators > 0:
            reasoning.append(f"Found {temporal_indicators} temporal proximity indicators")
        
        return min(1.0, score)
    
    def _calculate_entity_density_score(self, content: str, entities: List[str], reasoning: List[str]) -> float:
        """Calculate score based on entity density"""
        if not entities:
            return 0.0
        
        word_count = len(content.split())
        if word_count == 0:
            return 0.0
        
        entity_density = len(entities) / word_count
        score = min(1.0, entity_density * 10)  # Scale entity density
        
        if len(entities) > 0:
            reasoning.append(f"Entity density: {len(entities)} entities in {word_count} words")
        
        return score
    
    def _calculate_repetition_score(self, content: str, history: List[Dict[str, Any]], reasoning: List[str]) -> float:
        """Calculate score based on repetition in history"""
        if not history:
            return 0.0
        
        # Simple repetition check - count how many times similar content appears
        content_words = set(content.lower().split())
        repetition_count = 0
        
        for historical_item in history[-20:]:  # Check last 20 items
            hist_content = historical_item.get('content', '')
            hist_words = set(hist_content.lower().split())
            
            # Calculate word overlap
            overlap = len(content_words.intersection(hist_words))
            total_words = len(content_words.union(hist_words))
            
            if total_words > 0:
                similarity = overlap / total_words
                if similarity > 0.3:  # 30% word overlap threshold
                    repetition_count += 1
        
        if repetition_count > 0:
            score = min(1.0, repetition_count * 0.2)
            reasoning.append(f"Found {repetition_count} similar items in history")
            return score
        
        return 0.0
    
    def _calculate_frequency_pattern_score(self, content: str, history: List[Dict[str, Any]], reasoning: List[str]) -> float:
        """Calculate score based on frequency patterns"""
        if not history:
            return 0.0
        
        # Extract key terms from content
        key_terms = set()
        words = content.lower().split()
        for word in words:
            if len(word) > 4 and word.isalpha():  # Significant words only
                key_terms.add(word)
        
        if not key_terms:
            return 0.0
        
        # Count frequency of key terms in history
        term_frequencies = {}
        for term in key_terms:
            count = 0
            for historical_item in history[-50:]:  # Check last 50 items
                hist_content = historical_item.get('content', '').lower()
                count += hist_content.count(term)
            term_frequencies[term] = count
        
        # Calculate score based on frequency patterns
        max_frequency = max(term_frequencies.values()) if term_frequencies else 0
        if max_frequency > 2:  # Term appears more than twice
            score = min(1.0, max_frequency * 0.1)
            reasoning.append(f"Key terms appear frequently in history (max: {max_frequency} times)")
            return score
        
        return 0.0
    
    def _calculate_confidence(self, content: str, context: Dict[str, Any], entities: List[str], 
                           facts: List[Dict[str, Any]], history: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the importance score"""
        confidence = 0.5  # Base confidence
        
        # More data points increase confidence
        if context:
            confidence += 0.1
        if entities:
            confidence += 0.1
        if facts:
            confidence += 0.1
        if history:
            confidence += 0.1
        
        # Content length affects confidence
        word_count = len(content.split())
        if word_count > 10:
            confidence += 0.1
        if word_count > 50:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    async def score_memory_batch(self, memories: List[Dict[str, Any]], 
                                context: Dict[str, Any] = None) -> List[Tuple[Dict[str, Any], ImportanceScore]]:
        """
        Score a batch of memories for importance.
        
        Args:
            memories: List of memory objects to score
            context: Shared context for all memories
            
        Returns:
            List of (memory, ImportanceScore) tuples
        """
        results = []
        
        # Build history from all memories for repetition analysis
        history = [{'content': m.get('content', '')} for m in memories]
        
        for i, memory in enumerate(memories):
            content = memory.get('content', '')
            entities = memory.get('entities', [])
            facts = memory.get('facts', [])
            
            # Use other memories as history context (excluding current)
            memory_history = history[:i] + history[i+1:]
            
            # Merge memory context with shared context
            memory_context = {**(context or {}), **memory.get('metadata', {})}
            
            score = await self.calculate_importance(
                content=content,
                context=memory_context,
                entities=entities,
                facts=facts,
                history=memory_history
            )
            
            results.append((memory, score))
        
        return results
    
    def get_importance_statistics(self, scores: List[ImportanceScore]) -> Dict[str, Any]:
        """Get statistics about importance scores"""
        if not scores:
            return {'total': 0}
        
        total_scores = [score.total_score for score in scores]
        normalized_scores = [score.normalized_score for score in scores]
        confidences = [score.confidence for score in scores]
        
        # Factor contribution analysis
        factor_contributions = {}
        for factor in ImportanceFactors:
            factor_scores = [score.factor_scores.get(factor, 0) for score in scores]
            if any(score > 0 for score in factor_scores):
                factor_contributions[factor.value] = {
                    'avg_score': sum(factor_scores) / len(factor_scores),
                    'max_score': max(factor_scores),
                    'count_nonzero': sum(1 for score in factor_scores if score > 0)
                }
        
        return {
            'total_scores': len(scores),
            'avg_importance': sum(total_scores) / len(total_scores),
            'max_importance': max(total_scores),
            'min_importance': min(total_scores),
            'avg_normalized': sum(normalized_scores) / len(normalized_scores),
            'avg_confidence': sum(confidences) / len(confidences),
            'high_importance_count': sum(1 for score in normalized_scores if score >= 0.8),
            'medium_importance_count': sum(1 for score in normalized_scores if 0.5 <= score < 0.8),
            'low_importance_count': sum(1 for score in normalized_scores if score < 0.5),
            'factor_contributions': factor_contributions
        }


# Global service instance
importance_scorer = IntelligentImportanceScorer()


async def calculate_content_importance(content: str, context: Dict[str, Any] = None, 
                                     entities: List[str] = None, facts: List[Dict[str, Any]] = None,
                                     history: List[Dict[str, Any]] = None) -> ImportanceScore:
    """
    Convenience function to calculate importance score for content.
    
    Args:
        content: Text content to score
        context: Additional context
        entities: Extracted entities
        facts: Extracted facts
        history: Historical content
        
    Returns:
        ImportanceScore object
    """
    return await importance_scorer.calculate_importance(content, context, entities, facts, history)


async def score_memories_by_importance(memories: List[Dict[str, Any]], 
                                     context: Dict[str, Any] = None) -> List[Tuple[Dict[str, Any], ImportanceScore]]:
    """
    Score multiple memories for importance.
    
    Args:
        memories: List of memory objects
        context: Shared context
        
    Returns:
        List of (memory, ImportanceScore) tuples sorted by importance
    """
    results = await importance_scorer.score_memory_batch(memories, context)
    
    # Sort by importance score (descending)
    results.sort(key=lambda x: x[1].normalized_score, reverse=True)
    
    return results