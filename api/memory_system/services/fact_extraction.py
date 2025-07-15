"""
Fact extraction service for extracting factual statements, decisions, preferences, 
and action items from conversation text.

This service analyzes conversation text to identify and extract key facts, decisions,
user preferences, and actionable items that should be preserved in the memory system.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class FactType(Enum):
    """Types of facts that can be extracted"""
    STATEMENT = "statement"  # General factual statements
    DECISION = "decision"    # Decisions that were made
    PREFERENCE = "preference"  # User preferences and choices
    ACTION_ITEM = "action_item"  # Tasks or actions to be taken
    REQUIREMENT = "requirement"  # Requirements or constraints
    OBSERVATION = "observation"  # Observations or findings
    RULE = "rule"  # Rules or guidelines
    CONFIGURATION = "configuration"  # Configuration or settings
    ERROR_FACT = "error_fact"  # Facts about errors or issues
    SOLUTION = "solution"  # Solutions or fixes
    GOAL = "goal"  # Goals or objectives
    CONSTRAINT = "constraint"  # Limitations or constraints


@dataclass
class ExtractedFact:
    """Represents an extracted fact with metadata"""
    content: str
    fact_type: FactType
    confidence: float
    source_span: Tuple[int, int]  # Start and end positions in original text
    context: str
    entities: List[str]
    temporal_info: Optional[str] = None
    certainty_level: float = 0.8  # How certain is this fact (0.0-1.0)
    importance: float = 0.5  # How important is this fact (0.0-1.0)
    metadata: Dict[str, Any] = None
    related_facts: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.related_facts is None:
            self.related_facts = []


class FactExtractionPatterns:
    """Predefined patterns for fact extraction"""
    
    # Decision patterns
    DECISION_PATTERNS = [
        r'(?:I|we|team|project)\s+(?:decided|chose|selected|picked|went with|settled on)\s+(.+?)(?:\.|$|because|since)',
        r'(?:decision|choice)\s+(?:is|was|will be)\s+to\s+(.+?)(?:\.|$)',
        r'(?:we|I)\s+(?:will|should|must)\s+(?:use|implement|go with|choose)\s+(.+?)(?:\.|$|for|because)',
        r'(?:after|following)\s+(?:discussion|consideration|analysis|review),?\s+(?:we|I|team)\s+(?:decided|chose)\s+(.+?)(?:\.|$)',
        r'(?:final|ultimate|main|primary)\s+(?:decision|choice|selection)\s+(?:is|was)\s+(.+?)(?:\.|$)'
    ]
    
    # Preference patterns
    PREFERENCE_PATTERNS = [
        r'(?:I|we|team)\s+(?:prefer|like|favor|would rather|want)\s+(.+?)(?:\.|$|over|than|because)',
        r'(?:preference|choice)\s+(?:is|was)\s+(?:for|to use|to go with)\s+(.+?)(?:\.|$)',
        r'(?:I|we)\s+(?:think|believe|feel)\s+(.+?)\s+(?:is|would be)\s+(?:better|best|ideal|preferred)(?:\.|$)',
        r'(?:I|we|team)\s+(?:don\'t|do not)\s+(?:like|want|prefer)\s+(.+?)(?:\.|$|because)',
        r'(?:personally|in my opinion|I think),?\s+(.+?)\s+(?:is|would be)\s+(?:better|preferable|ideal)(?:\.|$)'
    ]
    
    # Action item patterns
    ACTION_PATTERNS = [
        r'(?:need to|should|must|have to|will|plan to|going to)\s+(.+?)(?:\.|$|before|after|by|when)',
        r'(?:action|task|todo|next step)\s+(?:is|was|will be)\s+(?:to\s+)?(.+?)(?:\.|$)',
        r'(?:I|we|team)\s+(?:will|should|need to|must|have to)\s+(.+?)(?:\.|$|by|before|after)',
        r'(?:remember to|don\'t forget to|make sure to)\s+(.+?)(?:\.|$)',
        r'(?:TODO|FIXME|NOTE):\s*(.+?)(?:\.|$|$)',
        r'(?:next|upcoming|planned)\s+(?:step|action|task)\s+(?:is|will be)\s+(?:to\s+)?(.+?)(?:\.|$)'
    ]
    
    # Requirement patterns
    REQUIREMENT_PATTERNS = [
        r'(?:requirement|constraint|condition)\s+(?:is|was)\s+(?:that\s+)?(.+?)(?:\.|$)',
        r'(?:must|required to|necessary to|essential to)\s+(.+?)(?:\.|$|in order|to ensure)',
        r'(?:system|application|code|project)\s+(?:must|should|needs to|requires)\s+(.+?)(?:\.|$)',
        r'(?:it|this|that)\s+(?:is|was)\s+(?:required|mandatory|necessary)\s+(?:that\s+|to\s+)?(.+?)(?:\.|$)',
        r'(?:business|functional|technical)\s+requirement:\s*(.+?)(?:\.|$)'
    ]
    
    # Configuration patterns
    CONFIG_PATTERNS = [
        r'(?:configuration|config|setting|parameter)\s+(?:is|was|will be)\s+(?:set to\s+)?(.+?)(?:\.|$)',
        r'(?:configured|set up|setup|configured)\s+(.+?)\s+(?:to|as|with)\s+(.+?)(?:\.|$)',
        r'(?:environment|env)\s+(?:variable|var)\s+(.+?)\s+(?:is|was|set to)\s+(.+?)(?:\.|$)',
        r'(?:database|db|server|host)\s+(?:is|was|configured|set)\s+(?:to|as|at)\s+(.+?)(?:\.|$)',
        r'(?:port|url|endpoint|address)\s+(?:is|was|set to)\s+(.+?)(?:\.|$)'
    ]
    
    # Error/problem patterns
    ERROR_PATTERNS = [
        r'(?:error|issue|problem|bug|failure)\s+(?:is|was|occurs|happens)\s+(.+?)(?:\.|$)',
        r'(?:getting|seeing|encountering|facing)\s+(?:an?\s+)?(?:error|issue|problem)\s+(?:with|in|when)\s+(.+?)(?:\.|$)',
        r'(?:fails|failing|broken|not working)\s+(?:when|if|because)\s+(.+?)(?:\.|$)',
        r'(?:exception|error message|stack trace):\s*(.+?)(?:\.|$)',
        r'(?:root cause|cause|reason)\s+(?:is|was|seems to be)\s+(.+?)(?:\.|$)'
    ]
    
    # Solution patterns
    SOLUTION_PATTERNS = [
        r'(?:solution|fix|workaround|resolution)\s+(?:is|was)\s+(?:to\s+)?(.+?)(?:\.|$)',
        r'(?:fixed|resolved|solved)\s+(?:by|through|using|with)\s+(.+?)(?:\.|$)',
        r'(?:to fix|to resolve|to solve)\s+(?:this|the issue|the problem),?\s+(.+?)(?:\.|$)',
        r'(?:way to|method to|approach to)\s+(?:fix|resolve|solve)\s+(?:this|it)\s+(?:is|was)\s+(.+?)(?:\.|$)',
        r'(?:workaround|temporary fix)\s+(?:is|was)\s+(?:to\s+)?(.+?)(?:\.|$)'
    ]
    
    # Observation patterns
    OBSERVATION_PATTERNS = [
        r'(?:noticed|observed|found|discovered)\s+(?:that\s+)?(.+?)(?:\.|$)',
        r'(?:it|this|that)\s+(?:appears|seems)\s+(?:that\s+|to\s+)?(.+?)(?:\.|$)',
        r'(?:based on|according to|from)\s+(.+?),?\s+(?:it|we|I)\s+(?:can see|observed|found)\s+(?:that\s+)?(.+?)(?:\.|$)',
        r'(?:evidence|data|results)\s+(?:shows?|indicates?|suggests?)\s+(?:that\s+)?(.+?)(?:\.|$)',
        r'(?:analysis|investigation|research)\s+(?:reveals?|shows?)\s+(?:that\s+)?(.+?)(?:\.|$)'
    ]


class IntelligentFactExtractor:
    """
    Intelligent fact extraction service that identifies and extracts key facts,
    decisions, preferences, and actionable items from conversation text.
    """
    
    def __init__(self):
        self.patterns = FactExtractionPatterns()
        
        # Compile regex patterns for performance
        self.compiled_patterns = {
            FactType.DECISION: [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns.DECISION_PATTERNS],
            FactType.PREFERENCE: [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns.PREFERENCE_PATTERNS],
            FactType.ACTION_ITEM: [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns.ACTION_PATTERNS],
            FactType.REQUIREMENT: [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns.REQUIREMENT_PATTERNS],
            FactType.CONFIGURATION: [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns.CONFIG_PATTERNS],
            FactType.ERROR_FACT: [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns.ERROR_PATTERNS],
            FactType.SOLUTION: [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns.SOLUTION_PATTERNS],
            FactType.OBSERVATION: [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns.OBSERVATION_PATTERNS]
        }
        
        # Keywords that indicate high certainty
        self.certainty_indicators = {
            'high': ['definitely', 'certainly', 'absolutely', 'confirmed', 'verified', 'established'],
            'medium': ['probably', 'likely', 'seems', 'appears', 'suggests'],
            'low': ['maybe', 'perhaps', 'possibly', 'might', 'could', 'uncertain']
        }
        
        # Temporal indicators
        self.temporal_patterns = [
            r'(?:yesterday|today|tomorrow|next week|last week|next month|last month)',
            r'(?:in|after|before|during|by)\s+(?:\d+\s+)?(?:minutes?|hours?|days?|weeks?|months?|years?)',
            r'(?:at|on|in)\s+\d{1,2}(?::\d{2})?(?:\s*(?:AM|PM))?',
            r'(?:january|february|march|april|may|june|july|august|september|october|november|december)',
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        ]
        
        self.compiled_temporal = [re.compile(pattern, re.IGNORECASE) for pattern in self.temporal_patterns]
    
    async def extract_facts(self, text: str, context: Dict[str, Any] = None, entities: List[str] = None) -> List[ExtractedFact]:
        """
        Extract facts from conversation text.
        
        Args:
            text: Text to extract facts from
            context: Additional context about the conversation
            entities: Pre-extracted entities to associate with facts
            
        Returns:
            List of ExtractedFact objects
        """
        if not text or not text.strip():
            return []
        
        context = context or {}
        entities = entities or []
        facts = []
        
        logger.info(f"Starting fact extraction from {len(text)} characters of text")
        
        # Extract facts using pattern matching
        for fact_type, patterns in self.compiled_patterns.items():
            type_facts = await self._extract_facts_by_type(text, fact_type, patterns, entities, context)
            facts.extend(type_facts)
        
        # Extract general statements (fallback for unmatched content)
        statement_facts = await self._extract_general_statements(text, facts, entities, context)
        facts.extend(statement_facts)
        
        # Post-process facts
        facts = self._deduplicate_facts(facts)
        facts = self._enhance_facts_with_context(facts, text, context)
        facts = self._calculate_importance_scores(facts, context)
        
        # Sort by importance and confidence
        facts.sort(key=lambda f: (f.importance, f.confidence), reverse=True)
        
        logger.info(f"Extracted {len(facts)} facts from conversation")
        return facts
    
    async def _extract_facts_by_type(self, text: str, fact_type: FactType, patterns: List[re.Pattern], 
                                   entities: List[str], context: Dict[str, Any]) -> List[ExtractedFact]:
        """Extract facts of a specific type using regex patterns"""
        facts = []
        
        for pattern in patterns:
            for match in pattern.finditer(text):
                if match.groups():
                    # Extract the main fact content from the first group
                    fact_content = match.group(1).strip()
                    
                    # Skip if too short or just punctuation
                    if len(fact_content) < 5 or not any(c.isalnum() for c in fact_content):
                        continue
                    
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_pattern_confidence(pattern, match, text)
                    
                    # Extract temporal information
                    temporal_info = self._extract_temporal_info(match.group(0))
                    
                    # Calculate certainty level
                    certainty_level = self._calculate_certainty_level(match.group(0))
                    
                    # Get context around the match
                    context_text = self._get_fact_context(text, match.start(), match.end())
                    
                    # Find related entities
                    fact_entities = self._find_related_entities(fact_content, entities)
                    
                    fact = ExtractedFact(
                        content=fact_content,
                        fact_type=fact_type,
                        confidence=confidence,
                        source_span=(match.start(), match.end()),
                        context=context_text,
                        entities=fact_entities,
                        temporal_info=temporal_info,
                        certainty_level=certainty_level,
                        metadata={'pattern_match': True, 'original_text': match.group(0)}
                    )
                    
                    facts.append(fact)
        
        return facts
    
    async def _extract_general_statements(self, text: str, existing_facts: List[ExtractedFact], 
                                        entities: List[str], context: Dict[str, Any]) -> List[ExtractedFact]:
        """Extract general factual statements that weren't caught by specific patterns"""
        facts = []
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        # Get spans of existing facts to avoid duplication
        existing_spans = set()
        for fact in existing_facts:
            start, end = fact.source_span
            existing_spans.add((start, end))
        
        current_pos = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                current_pos += len(sentence) + 1
                continue
            
            # Find position in original text
            sentence_start = text.find(sentence, current_pos)
            sentence_end = sentence_start + len(sentence)
            
            # Skip if this sentence overlaps with existing facts
            overlaps = any(
                self._ranges_overlap((sentence_start, sentence_end), span)
                for span in existing_spans
            )
            
            if not overlaps and self._is_factual_statement(sentence):
                # Calculate confidence for general statements
                confidence = self._calculate_statement_confidence(sentence)
                
                # Only include if confidence is reasonable
                if confidence >= 0.3:
                    fact_entities = self._find_related_entities(sentence, entities)
                    temporal_info = self._extract_temporal_info(sentence)
                    certainty_level = self._calculate_certainty_level(sentence)
                    
                    fact = ExtractedFact(
                        content=sentence,
                        fact_type=FactType.STATEMENT,
                        confidence=confidence,
                        source_span=(sentence_start, sentence_end),
                        context=self._get_fact_context(text, sentence_start, sentence_end),
                        entities=fact_entities,
                        temporal_info=temporal_info,
                        certainty_level=certainty_level,
                        metadata={'general_statement': True}
                    )
                    
                    facts.append(fact)
            
            current_pos = sentence_end + 1
        
        return facts
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple rules"""
        # Simple sentence splitting - can be enhanced with NLP libraries
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_factual_statement(self, sentence: str) -> bool:
        """Determine if a sentence contains factual content"""
        # Skip questions
        if sentence.strip().endswith('?'):
            return False
        
        # Skip very short sentences
        if len(sentence.split()) < 4:
            return False
        
        # Skip sentences that are primarily code
        if sentence.count('(') + sentence.count('{') + sentence.count('[') > 2:
            return False
        
        # Look for factual indicators
        factual_indicators = [
            'is', 'are', 'was', 'were', 'has', 'have', 'contains', 'includes',
            'provides', 'supports', 'enables', 'allows', 'requires', 'uses',
            'implemented', 'configured', 'designed', 'built', 'created'
        ]
        
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in factual_indicators)
    
    def _calculate_pattern_confidence(self, pattern: re.Pattern, match: re.Match, text: str) -> float:
        """Calculate confidence based on pattern specificity and context"""
        base_confidence = 0.7
        
        # Increase confidence for more specific patterns
        pattern_str = pattern.pattern
        if 'decided' in pattern_str or 'chose' in pattern_str:
            base_confidence = 0.9
        elif 'prefer' in pattern_str or 'like' in pattern_str:
            base_confidence = 0.8
        elif 'need to' in pattern_str or 'must' in pattern_str:
            base_confidence = 0.85
        
        # Adjust based on match quality
        matched_text = match.group(0)
        if len(matched_text) > 50:  # Longer matches are often more reliable
            base_confidence += 0.05
        
        # Check for certainty indicators
        certainty_bonus = self._get_certainty_bonus(matched_text)
        
        return min(1.0, base_confidence + certainty_bonus)
    
    def _calculate_statement_confidence(self, sentence: str) -> float:
        """Calculate confidence for general statements"""
        base_confidence = 0.4
        
        sentence_lower = sentence.lower()
        
        # Boost confidence for statements with strong indicators
        strong_indicators = ['is', 'are', 'was', 'were', 'has', 'have']
        if any(f' {indicator} ' in sentence_lower for indicator in strong_indicators):
            base_confidence += 0.2
        
        # Boost for technical content
        technical_indicators = ['system', 'application', 'database', 'server', 'code', 'function']
        if any(indicator in sentence_lower for indicator in technical_indicators):
            base_confidence += 0.15
        
        # Reduce confidence for uncertain language
        uncertainty_indicators = ['maybe', 'perhaps', 'might', 'could', 'possibly']
        if any(indicator in sentence_lower for indicator in uncertainty_indicators):
            base_confidence -= 0.2
        
        return max(0.1, min(1.0, base_confidence))
    
    def _calculate_certainty_level(self, text: str) -> float:
        """Calculate how certain/confident the fact is"""
        text_lower = text.lower()
        
        # Check for high certainty indicators
        for indicator in self.certainty_indicators['high']:
            if indicator in text_lower:
                return 0.9
        
        # Check for medium certainty indicators
        for indicator in self.certainty_indicators['medium']:
            if indicator in text_lower:
                return 0.6
        
        # Check for low certainty indicators
        for indicator in self.certainty_indicators['low']:
            if indicator in text_lower:
                return 0.3
        
        # Default certainty level
        return 0.7
    
    def _get_certainty_bonus(self, text: str) -> float:
        """Get confidence bonus based on certainty indicators"""
        text_lower = text.lower()
        
        if any(indicator in text_lower for indicator in self.certainty_indicators['high']):
            return 0.1
        elif any(indicator in text_lower for indicator in self.certainty_indicators['low']):
            return -0.1
        
        return 0.0
    
    def _extract_temporal_info(self, text: str) -> Optional[str]:
        """Extract temporal/time information from text"""
        for pattern in self.compiled_temporal:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return None
    
    def _get_fact_context(self, text: str, start: int, end: int, context_size: int = 100) -> str:
        """Get surrounding context for a fact"""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        return text[context_start:context_end]
    
    def _find_related_entities(self, fact_content: str, entities: List[str]) -> List[str]:
        """Find entities that are mentioned in the fact content"""
        fact_lower = fact_content.lower()
        related = []
        
        for entity in entities:
            if entity.lower() in fact_lower:
                related.append(entity)
        
        return related
    
    def _deduplicate_facts(self, facts: List[ExtractedFact]) -> List[ExtractedFact]:
        """Remove duplicate facts"""
        if not facts:
            return facts
        
        deduplicated = []
        seen_contents = set()
        
        for fact in facts:
            # Normalize content for comparison
            normalized = re.sub(r'\s+', ' ', fact.content.lower().strip())
            
            if normalized not in seen_contents:
                seen_contents.add(normalized)
                deduplicated.append(fact)
            else:
                # If we've seen this content, merge with existing fact if confidence is higher
                for existing in deduplicated:
                    existing_normalized = re.sub(r'\s+', ' ', existing.content.lower().strip())
                    if existing_normalized == normalized and fact.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(fact)
                        break
        
        return deduplicated
    
    def _enhance_facts_with_context(self, facts: List[ExtractedFact], text: str, context: Dict[str, Any]) -> List[ExtractedFact]:
        """Enhance facts with additional context information"""
        for fact in facts:
            # Add session context
            if 'session_id' in context:
                fact.metadata['session_id'] = context['session_id']
            if 'user_id' in context:
                fact.metadata['user_id'] = context['user_id']
            
            # Add surrounding sentence context
            start, end = fact.source_span
            sentence_context = self._get_sentence_context(text, start, end)
            fact.metadata['sentence_context'] = sentence_context
            
            # Add fact length
            fact.metadata['fact_length'] = len(fact.content)
            fact.metadata['word_count'] = len(fact.content.split())
        
        return facts
    
    def _get_sentence_context(self, text: str, start: int, end: int) -> str:
        """Get the full sentence containing the fact"""
        # Find sentence boundaries around the fact
        before = text[:start]
        after = text[end:]
        
        # Find last sentence start before the fact
        sentence_start = 0
        for match in re.finditer(r'[.!?]\s+', before):
            sentence_start = match.end()
        
        # Find next sentence end after the fact
        sentence_end = len(text)
        match = re.search(r'[.!?]', after)
        if match:
            sentence_end = end + match.end()
        
        return text[sentence_start:sentence_end].strip()
    
    def _calculate_importance_scores(self, facts: List[ExtractedFact], context: Dict[str, Any]) -> List[ExtractedFact]:
        """Calculate importance scores for facts"""
        for fact in facts:
            importance = 0.5  # Base importance
            
            # Fact type importance
            type_importance = {
                FactType.DECISION: 0.9,
                FactType.REQUIREMENT: 0.85,
                FactType.ACTION_ITEM: 0.8,
                FactType.ERROR_FACT: 0.75,
                FactType.SOLUTION: 0.8,
                FactType.PREFERENCE: 0.7,
                FactType.CONFIGURATION: 0.75,
                FactType.GOAL: 0.8,
                FactType.CONSTRAINT: 0.75,
                FactType.RULE: 0.7,
                FactType.OBSERVATION: 0.6,
                FactType.STATEMENT: 0.5
            }
            
            importance = type_importance.get(fact.fact_type, 0.5)
            
            # Boost importance based on confidence and certainty
            importance += (fact.confidence - 0.5) * 0.2
            importance += (fact.certainty_level - 0.5) * 0.1
            
            # Boost for facts with entities
            if fact.entities:
                importance += min(0.1, len(fact.entities) * 0.02)
            
            # Boost for facts with temporal information
            if fact.temporal_info:
                importance += 0.05
            
            # Boost for longer, more detailed facts
            word_count = len(fact.content.split())
            if word_count > 10:
                importance += min(0.1, (word_count - 10) * 0.01)
            
            # Context-based importance
            if context.get('is_critical', False):
                importance += 0.15
            
            fact.importance = min(1.0, max(0.0, importance))
        
        return facts
    
    def _ranges_overlap(self, range1: Tuple[int, int], range2: Tuple[int, int]) -> bool:
        """Check if two ranges overlap"""
        start1, end1 = range1
        start2, end2 = range2
        return start1 < end2 and start2 < end1
    
    def get_fact_statistics(self, facts: List[ExtractedFact]) -> Dict[str, Any]:
        """Get statistics about extracted facts"""
        if not facts:
            return {'total': 0}
        
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
        
        return {
            'total': len(facts),
            'by_type': by_type,
            'average_confidence': total_confidence / len(facts),
            'average_certainty': total_certainty / len(facts),
            'average_importance': total_importance / len(facts),
            'facts_with_entities': facts_with_entities,
            'facts_with_temporal_info': facts_with_temporal,
            'unique_fact_types': len(by_type)
        }
    
    async def extract_facts_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract facts from text chunks and add to metadata.
        
        Args:
            chunks: List of text chunks with content
            
        Returns:
            Updated chunks with facts added to metadata
        """
        updated_chunks = []
        
        for chunk in chunks:
            chunk_copy = chunk.copy()
            
            # Get entities from chunk if available
            entities = chunk.get('entities', [])
            
            # Extract facts from chunk content
            facts = await self.extract_facts(
                chunk['content'], 
                context=chunk.get('metadata', {}),
                entities=entities
            )
            
            # Convert facts to serializable format
            fact_data = []
            for fact in facts:
                fact_data.append({
                    'content': fact.content,
                    'type': fact.fact_type.value,
                    'confidence': fact.confidence,
                    'certainty_level': fact.certainty_level,
                    'importance': fact.importance,
                    'source_span': fact.source_span,
                    'context': fact.context,
                    'entities': fact.entities,
                    'temporal_info': fact.temporal_info,
                    'metadata': fact.metadata
                })
            
            # Update chunk with facts
            chunk_copy['facts'] = [fact.content for fact in facts]  # Simple list for compatibility
            chunk_copy['fact_details'] = fact_data  # Detailed fact information
            
            # Add fact statistics to metadata
            if 'metadata' not in chunk_copy:
                chunk_copy['metadata'] = {}
            
            chunk_copy['metadata']['fact_stats'] = self.get_fact_statistics(facts)
            
            updated_chunks.append(chunk_copy)
        
        return updated_chunks


# Global service instance
fact_extractor = IntelligentFactExtractor()


async def extract_facts_from_text(text: str, context: Dict[str, Any] = None, entities: List[str] = None) -> List[ExtractedFact]:
    """
    Convenience function to extract facts from text.
    
    Args:
        text: Text to extract facts from
        context: Additional context
        entities: Pre-extracted entities
        
    Returns:
        List of ExtractedFact objects
    """
    return await fact_extractor.extract_facts(text, context, entities)


async def enrich_chunks_with_facts(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich text chunks with extracted facts.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Chunks enriched with fact information
    """
    return await fact_extractor.extract_facts_from_chunks(chunks)