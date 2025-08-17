"""
Advanced Semantic Analysis Engine - Phase 2.2
Created by Tinne Smets - Expert in Weight Sharing & Advanced Semantic Understanding

This system implements advanced semantic analysis that goes beyond simple embedding 
similarity, featuring context-aware entity linking, semantic role labeling, and 
intent recognition with efficient weight sharing across tasks.

Key Features:
- Context-aware entity linking and disambiguation
- Semantic role labeling with argument structure analysis
- Intent recognition and context-dependent meaning resolution
- Cross-lingual semantic understanding with shared representations
- Real-time semantic relationship extraction and validation
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter
from enum import Enum
import json
import re
import hashlib

# Import spaCy for NLP processing
try:
    import spacy
    from spacy.tokens import Doc, Token, Span
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. NLP features will be limited.")

# Import our weight sharing components
from .weight_sharing_semantic_engine import WeightSharingSemanticEngine, ContextLevel
from .context_hierarchy_engine import ContextNode, ContextRelation, ContextRelationType

logger = logging.getLogger(__name__)

class SemanticRole(Enum):
    """Semantic roles for predicate-argument structure."""
    AGENT = "ARG0"        # The doer of an action
    PATIENT = "ARG1"      # The thing affected by an action
    INSTRUMENT = "ARG2"   # The instrument or means
    LOCATION = "ARGM-LOC" # Location
    TIME = "ARGM-TMP"     # Temporal
    MANNER = "ARGM-MNR"   # Manner
    PURPOSE = "ARGM-PRP"  # Purpose
    CAUSE = "ARGM-CAU"    # Cause
    BENEFICIARY = "ARG3"  # Beneficiary
    THEME = "ARG1"        # Theme (alternative to patient)

class EntityType(Enum):
    """Enhanced entity types for context-aware linking."""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "GPE"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    TECHNOLOGY = "TECHNOLOGY"
    CONCEPT = "CONCEPT"
    TEMPORAL = "DATE"
    NUMERICAL = "CARDINAL"
    MONETARY = "MONEY"

@dataclass
class SemanticEntity:
    """Enhanced entity with contextual information."""
    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float
    
    # Contextual linking
    canonical_form: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    
    # Semantic properties
    semantic_class: Optional[str] = None
    domain: Optional[str] = None
    
    # Context information
    context_words: List[str] = field(default_factory=list)
    context_entities: List[str] = field(default_factory=list)
    disambiguation_score: float = 0.0
    
    # Relations
    relations: Dict[str, str] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticTriple:
    """Semantic relationship triple (subject, predicate, object)."""
    subject: SemanticEntity
    predicate: str
    object_: Union[SemanticEntity, str]
    confidence: float
    
    # Semantic role information
    predicate_roles: Dict[SemanticRole, str] = field(default_factory=dict)
    
    # Context
    source_sentence: str = ""
    context_window: List[str] = field(default_factory=list)
    
    # Validation
    is_validated: bool = False
    validation_score: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IntentAnalysis:
    """Intent recognition and understanding."""
    primary_intent: str
    intent_confidence: float
    secondary_intents: List[Tuple[str, float]] = field(default_factory=list)
    
    # Intent components
    entities: List[SemanticEntity] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)
    
    # Context-dependent meaning
    contextual_meaning: str = ""
    ambiguity_score: float = 0.0
    
    # Resolution
    resolved_entities: Dict[str, str] = field(default_factory=dict)
    coreference_chains: List[List[str]] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContextAwareEntityLinker:
    """
    Advanced entity linking with context awareness and disambiguation.
    
    Uses context information and cross-document knowledge to accurately
    link entities to their canonical representations.
    """
    
    def __init__(
        self,
        knowledge_bases: Dict[str, str] = None,
        use_context_window: int = 10,
        disambiguation_threshold: float = 0.7
    ):
        self.knowledge_bases = knowledge_bases or {
            'wikidata': 'https://wikidata.org',
            'freebase': 'https://freebase.com'
        }
        self.use_context_window = use_context_window
        self.disambiguation_threshold = disambiguation_threshold
        
        # Entity caches for performance
        self.entity_cache = {}
        self.canonical_forms = {}
        self.disambiguation_cache = {}
        
        # Context patterns for disambiguation
        self.context_patterns = self._load_context_patterns()
        
        logger.info("ContextAwareEntityLinker initialized")
    
    def _load_context_patterns(self) -> Dict[str, List[str]]:
        """Load context patterns for entity disambiguation."""
        return {
            'PERSON': [
                r'\b(?:Mr|Ms|Dr|Prof|CEO|President|Director)\s+{}',
                r'{}\s+(?:said|announced|stated|declared)',
                r'(?:by|from|with)\s+{}'
            ],
            'ORGANIZATION': [
                r'{}\s+(?:Inc|Corp|LLC|Ltd|Company)',
                r'(?:at|in|for)\s+{}',
                r'{}\s+(?:announced|reported|stated)'
            ],
            'LOCATION': [
                r'(?:in|at|from|to)\s+{}',
                r'{}\s+(?:is|was|has)',
                r'located\s+in\s+{}'
            ]
        }
    
    async def link_entities(
        self,
        entities: List[SemanticEntity],
        context: str,
        document_entities: List[SemanticEntity] = None
    ) -> List[SemanticEntity]:
        """Link entities to their canonical forms with context awareness."""
        linked_entities = []
        
        for entity in entities:
            try:
                # Get context around entity
                context_words = self._extract_context_words(entity, context)
                
                # Find canonical form
                canonical_form = await self._find_canonical_form(
                    entity, context_words, document_entities
                )
                
                # Perform disambiguation if needed
                if self._needs_disambiguation(entity, canonical_form):
                    disambiguation_result = await self._disambiguate_entity(
                        entity, context_words, document_entities
                    )
                    entity.disambiguation_score = disambiguation_result['score']
                    if disambiguation_result['score'] > self.disambiguation_threshold:
                        canonical_form = disambiguation_result['canonical_form']
                
                # Update entity with linking information
                entity.canonical_form = canonical_form
                entity.context_words = context_words
                entity.knowledge_base_id = await self._get_kb_id(canonical_form, entity.entity_type)
                
                linked_entities.append(entity)
                
            except Exception as e:
                logger.warning(f"Failed to link entity {entity.text}: {e}")
                linked_entities.append(entity)  # Return original entity
        
        return linked_entities
    
    def _extract_context_words(self, entity: SemanticEntity, context: str) -> List[str]:
        """Extract relevant context words around entity."""
        words = context.split()
        
        # Find entity position in context
        entity_words = entity.text.split()
        start_idx = -1
        
        for i in range(len(words) - len(entity_words) + 1):
            if words[i:i+len(entity_words)] == entity_words:
                start_idx = i
                break
        
        if start_idx == -1:
            return words[:self.use_context_window]  # Fallback to beginning
        
        # Extract context window
        window_start = max(0, start_idx - self.use_context_window // 2)
        window_end = min(len(words), start_idx + len(entity_words) + self.use_context_window // 2)
        
        return words[window_start:window_end]
    
    async def _find_canonical_form(
        self,
        entity: SemanticEntity,
        context_words: List[str],
        document_entities: List[SemanticEntity]
    ) -> str:
        """Find canonical form of entity using context."""
        
        # Check cache first
        cache_key = f"{entity.text}_{entity.entity_type.value}"
        if cache_key in self.canonical_forms:
            return self.canonical_forms[cache_key]
        
        # Simple canonical form resolution (would use knowledge bases in production)
        canonical_form = entity.text.strip().title()
        
        # Handle common variations
        if entity.entity_type == EntityType.PERSON:
            canonical_form = self._canonicalize_person_name(entity.text, context_words)
        elif entity.entity_type == EntityType.ORGANIZATION:
            canonical_form = self._canonicalize_organization_name(entity.text, context_words)
        elif entity.entity_type == EntityType.LOCATION:
            canonical_form = self._canonicalize_location_name(entity.text, context_words)
        
        # Cache result
        self.canonical_forms[cache_key] = canonical_form
        
        return canonical_form
    
    def _canonicalize_person_name(self, name: str, context: List[str]) -> str:
        """Canonicalize person name using context."""
        # Remove titles and clean up
        titles = ['Mr.', 'Ms.', 'Dr.', 'Prof.', 'CEO', 'President']
        clean_name = name
        
        for title in titles:
            clean_name = clean_name.replace(title, '').strip()
        
        # Handle first name / last name variations
        parts = clean_name.split()
        if len(parts) >= 2:
            # Assume "Last, First" or "First Last" format
            if ',' in clean_name:
                last, first = clean_name.split(',', 1)
                return f"{first.strip()} {last.strip()}"
            else:
                return clean_name.title()
        
        return clean_name.title()
    
    def _canonicalize_organization_name(self, name: str, context: List[str]) -> str:
        """Canonicalize organization name using context."""
        # Remove common suffixes for canonical form
        suffixes = ['Inc.', 'Corp.', 'LLC', 'Ltd.', 'Company', 'Co.']
        canonical = name
        
        for suffix in suffixes:
            canonical = re.sub(rf'\s+{re.escape(suffix)}$', '', canonical, flags=re.IGNORECASE)
        
        return canonical.strip().title()
    
    def _canonicalize_location_name(self, name: str, context: List[str]) -> str:
        """Canonicalize location name using context."""
        # Simple canonicalization (would use geographical databases in production)
        return name.strip().title()
    
    def _needs_disambiguation(self, entity: SemanticEntity, canonical_form: str) -> bool:
        """Check if entity needs disambiguation."""
        # Simple heuristic: ambiguous if very common name or multiple possible meanings
        common_names = {
            'PERSON': ['John Smith', 'Mary Johnson', 'Michael Brown'],
            'LOCATION': ['Springfield', 'Georgetown', 'Franklin'],
            'ORGANIZATION': ['ABC', 'XYZ Corp', 'Global Systems']
        }
        
        return canonical_form in common_names.get(entity.entity_type.value, [])
    
    async def _disambiguate_entity(
        self,
        entity: SemanticEntity,
        context_words: List[str],
        document_entities: List[SemanticEntity]
    ) -> Dict[str, Any]:
        """Disambiguate entity using advanced context analysis."""
        
        # Get context patterns for entity type
        patterns = self.context_patterns.get(entity.entity_type.value, [])
        
        context_text = ' '.join(context_words)
        disambiguation_score = 0.0
        
        # Check context patterns
        for pattern in patterns:
            if re.search(pattern.format(re.escape(entity.text)), context_text, re.IGNORECASE):
                disambiguation_score += 0.2
        
        # Check co-occurring entities for additional context
        if document_entities:
            entity_types = Counter([e.entity_type for e in document_entities])
            # Boost score if entity type is common in document
            if entity_types[entity.entity_type] > 1:
                disambiguation_score += 0.3
        
        # Simple disambiguation result (would use knowledge bases in production)
        return {
            'canonical_form': f"{entity.text}_disambiguated",
            'score': min(disambiguation_score, 1.0),
            'method': 'context_pattern_matching'
        }
    
    async def _get_kb_id(self, canonical_form: str, entity_type: EntityType) -> Optional[str]:
        """Get knowledge base ID for canonical form."""
        # Would query actual knowledge bases in production
        # Return placeholder ID
        return f"kb:{entity_type.value.lower()}:{canonical_form.replace(' ', '_').lower()}"

class SemanticRoleLabelingEngine:
    """
    Semantic Role Labeling (SRL) engine with argument structure analysis.
    
    Identifies semantic roles in sentences and builds predicate-argument
    structures for deeper semantic understanding.
    """
    
    def __init__(
        self,
        use_neural_srl: bool = True,
        confidence_threshold: float = 0.6
    ):
        self.use_neural_srl = use_neural_srl
        self.confidence_threshold = confidence_threshold
        
        # Predicate-argument patterns
        self.verb_patterns = self._load_verb_patterns()
        self.role_mappings = self._load_role_mappings()
        
        # Cache for performance
        self.srl_cache = {}
        
        logger.info("SemanticRoleLabelingEngine initialized")
    
    def _load_verb_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load verb patterns for SRL."""
        return {
            'give': [
                {'roles': [SemanticRole.AGENT, SemanticRole.PATIENT, SemanticRole.BENEFICIARY], 
                 'pattern': r'(\w+)\s+gave\s+(\w+)\s+to\s+(\w+)'}
            ],
            'buy': [
                {'roles': [SemanticRole.AGENT, SemanticRole.PATIENT, SemanticRole.LOCATION], 
                 'pattern': r'(\w+)\s+bought\s+(\w+)\s+at\s+(\w+)'}
            ],
            'travel': [
                {'roles': [SemanticRole.AGENT, SemanticRole.LOCATION, SemanticRole.LOCATION], 
                 'pattern': r'(\w+)\s+traveled\s+from\s+(\w+)\s+to\s+(\w+)'}
            ]
        }
    
    def _load_role_mappings(self) -> Dict[str, SemanticRole]:
        """Load dependency relation to semantic role mappings."""
        return {
            'nsubj': SemanticRole.AGENT,
            'dobj': SemanticRole.PATIENT,
            'iobj': SemanticRole.BENEFICIARY,
            'prep_to': SemanticRole.BENEFICIARY,
            'prep_at': SemanticRole.LOCATION,
            'prep_in': SemanticRole.LOCATION,
            'prep_with': SemanticRole.INSTRUMENT,
            'prep_for': SemanticRole.PURPOSE,
            'prep_by': SemanticRole.AGENT,
            'advmod': SemanticRole.MANNER,
            'tmod': SemanticRole.TIME
        }
    
    async def analyze_semantic_roles(
        self,
        sentence: str,
        entities: List[SemanticEntity] = None
    ) -> List[SemanticTriple]:
        """Analyze semantic roles in a sentence."""
        
        # Check cache
        cache_key = hashlib.md5(sentence.encode()).hexdigest()[:16]
        if cache_key in self.srl_cache:
            return self.srl_cache[cache_key]
        
        triples = []
        
        try:
            # Simple pattern-based SRL (would use neural models in production)
            triples = await self._extract_triples_pattern_based(sentence, entities)
            
            # If neural SRL is enabled, enhance with neural predictions
            if self.use_neural_srl:
                neural_triples = await self._extract_triples_neural(sentence, entities)
                triples.extend(neural_triples)
            
            # Merge and validate triples
            triples = self._merge_and_validate_triples(triples)
            
            # Cache results
            self.srl_cache[cache_key] = triples
            
            return triples
            
        except Exception as e:
            logger.error(f"SRL analysis failed for sentence: {e}")
            return []
    
    async def _extract_triples_pattern_based(
        self,
        sentence: str,
        entities: List[SemanticEntity]
    ) -> List[SemanticTriple]:
        """Extract semantic triples using pattern matching."""
        triples = []
        
        # Simple pattern-based extraction
        for verb, patterns in self.verb_patterns.items():
            if verb in sentence.lower():
                for pattern_info in patterns:
                    pattern = pattern_info['pattern']
                    roles = pattern_info['roles']
                    
                    match = re.search(pattern, sentence, re.IGNORECASE)
                    if match:
                        groups = match.groups()
                        
                        if len(groups) >= 2:  # At least subject and predicate
                            subject_text = groups[0]
                            object_text = groups[1] if len(groups) > 1 else None
                            
                            # Create entities if not provided
                            subject_entity = self._create_entity_from_text(subject_text, sentence)
                            
                            triple = SemanticTriple(
                                subject=subject_entity,
                                predicate=verb,
                                object_=object_text if object_text else "",
                                confidence=0.8,  # Pattern-based confidence
                                source_sentence=sentence
                            )
                            
                            # Add role information
                            for i, role in enumerate(roles[:len(groups)]):
                                triple.predicate_roles[role] = groups[i]
                            
                            triples.append(triple)
        
        return triples
    
    async def _extract_triples_neural(
        self,
        sentence: str,
        entities: List[SemanticEntity]
    ) -> List[SemanticTriple]:
        """Extract semantic triples using neural SRL (placeholder)."""
        # This would use a neural SRL model in production
        # For now, return empty list
        return []
    
    def _create_entity_from_text(self, text: str, context: str) -> SemanticEntity:
        """Create a simple entity from text."""
        # Simple entity type detection
        entity_type = EntityType.CONCEPT  # Default
        
        if text[0].isupper():  # Capitalized might be person/org/location
            entity_type = EntityType.PERSON
        
        return SemanticEntity(
            text=text,
            entity_type=entity_type,
            start_pos=context.find(text),
            end_pos=context.find(text) + len(text),
            confidence=0.7
        )
    
    def _merge_and_validate_triples(self, triples: List[SemanticTriple]) -> List[SemanticTriple]:
        """Merge and validate semantic triples."""
        # Simple validation: remove duplicates and low-confidence triples
        validated_triples = []
        seen_triples = set()
        
        for triple in triples:
            if triple.confidence < self.confidence_threshold:
                continue
            
            triple_key = f"{triple.subject.text}_{triple.predicate}_{triple.object_}"
            if triple_key in seen_triples:
                continue
            
            seen_triples.add(triple_key)
            triple.is_validated = True
            triple.validation_score = triple.confidence
            validated_triples.append(triple)
        
        return validated_triples

class AdvancedSemanticEngine:
    """
    Main advanced semantic analysis engine.
    
    Integrates context-aware entity linking, semantic role labeling,
    and intent recognition for comprehensive semantic understanding.
    """
    
    def __init__(
        self,
        weight_sharing_engine: WeightSharingSemanticEngine,
        enable_entity_linking: bool = True,
        enable_srl: bool = True,
        enable_intent_analysis: bool = True
    ):
        self.weight_sharing_engine = weight_sharing_engine
        
        # Initialize components
        self.entity_linker = ContextAwareEntityLinker() if enable_entity_linking else None
        self.srl_engine = SemanticRoleLabelingEngine() if enable_srl else None
        self.enable_intent_analysis = enable_intent_analysis
        
        # Performance tracking
        self.analysis_metrics = defaultdict(list)
        
        logger.info("AdvancedSemanticEngine initialized")
    
    async def comprehensive_semantic_analysis(
        self,
        text: str,
        document_id: str,
        context_level: ContextLevel = ContextLevel.DOCUMENT,
        include_cross_document: bool = False
    ) -> Dict[str, Any]:
        """
        Perform comprehensive semantic analysis combining all engines.
        
        Args:
            text: Input text to analyze
            document_id: Document identifier
            context_level: Level of analysis to perform
            include_cross_document: Whether to include cross-document analysis
            
        Returns:
            Comprehensive semantic analysis results
        """
        start_time = datetime.utcnow()
        
        try:
            results = {
                'document_id': document_id,
                'text': text[:500] + "..." if len(text) > 500 else text,
                'analysis_level': context_level.value,
                'timestamp': start_time.isoformat()
            }
            
            # 1. Basic semantic analysis using weight sharing engine
            if hasattr(self.weight_sharing_engine, 'analyze_context_hierarchy'):
                weight_sharing_result = await self.weight_sharing_engine.analyze_context_hierarchy(
                    text, document_id, task_ids=['semantic_similarity', 'entity_extraction', 'context_understanding']
                )
                results['weight_sharing_analysis'] = weight_sharing_result
            
            # 2. Advanced entity extraction and linking
            entities = []
            if self.entity_linker:
                # Extract entities (simplified - would use NER model)
                entities = await self._extract_entities(text)
                
                # Link entities to canonical forms
                linked_entities = await self.entity_linker.link_entities(entities, text)
                results['entities'] = [
                    {
                        'text': e.text,
                        'type': e.entity_type.value,
                        'canonical_form': e.canonical_form,
                        'confidence': e.confidence,
                        'kb_id': e.knowledge_base_id,
                        'disambiguation_score': e.disambiguation_score
                    }
                    for e in linked_entities
                ]
            
            # 3. Semantic role labeling
            semantic_triples = []
            if self.srl_engine:
                # Split into sentences for SRL analysis
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                
                all_triples = []
                for sentence in sentences[:5]:  # Limit for performance
                    sentence_triples = await self.srl_engine.analyze_semantic_roles(sentence, entities)
                    all_triples.extend(sentence_triples)
                
                results['semantic_roles'] = [
                    {
                        'subject': t.subject.text,
                        'predicate': t.predicate,
                        'object': t.object_ if isinstance(t.object_, str) else t.object_.text,
                        'confidence': t.confidence,
                        'roles': {role.value: arg for role, arg in t.predicate_roles.items()},
                        'source_sentence': t.source_sentence
                    }
                    for t in all_triples
                ]
            
            # 4. Intent analysis
            if self.enable_intent_analysis:
                intent_result = await self._analyze_intent(text, entities, semantic_triples)
                results['intent_analysis'] = intent_result
            
            # 5. Cross-document analysis (if requested)
            if include_cross_document:
                cross_doc_result = await self._analyze_cross_document_semantics(text, document_id, entities)
                results['cross_document_analysis'] = cross_doc_result
            
            # 6. Calculate overall semantic metrics
            overall_metrics = self._calculate_semantic_metrics(results)
            results['semantic_metrics'] = overall_metrics
            
            # Track performance
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.analysis_metrics['processing_time'].append(processing_time)
            results['processing_time'] = processing_time
            
            logger.info(f"Comprehensive semantic analysis completed for {document_id} in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive semantic analysis failed for {document_id}: {e}")
            raise
    
    async def _extract_entities(self, text: str) -> List[SemanticEntity]:
        """Extract entities from text (simplified implementation)."""
        entities = []
        
        # Simple entity extraction using patterns
        patterns = {
            EntityType.PERSON: r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            EntityType.ORGANIZATION: r'\b[A-Z][a-zA-Z\s]+(?:Inc|Corp|LLC|Ltd|Company)\b',
            EntityType.LOCATION: r'\b[A-Z][a-zA-Z\s]+(?:City|State|Country|Street|Avenue)\b',
            EntityType.MONETARY: r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            EntityType.TEMPORAL: r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entity = SemanticEntity(
                    text=match.group(),
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8  # Pattern-based confidence
                )
                entities.append(entity)
        
        return entities
    
    async def _analyze_intent(
        self,
        text: str,
        entities: List[SemanticEntity],
        triples: List[SemanticTriple]
    ) -> Dict[str, Any]:
        """Analyze intent from text, entities, and semantic roles."""
        
        # Simple intent classification based on patterns
        intent_patterns = {
            'question': [r'\?', r'\bwhat\b', r'\bwho\b', r'\bwhere\b', r'\bwhen\b', r'\bhow\b', r'\bwhy\b'],
            'request': [r'\bplease\b', r'\bcould you\b', r'\bwould you\b', r'\bcan you\b'],
            'information': [r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b'],
            'action': [r'\bdo\b', r'\bmake\b', r'\bcreate\b', r'\bprocess\b', r'\banalyze\b']
        }
        
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    score += 1.0
            intent_scores[intent] = score / len(patterns)
        
        # Get primary intent
        primary_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x]) if intent_scores else 'unknown'
        primary_confidence = intent_scores.get(primary_intent, 0.0)
        
        # Extract actions from triples
        actions = [t.predicate for t in triples] if triples else []
        
        return {
            'primary_intent': primary_intent,
            'confidence': primary_confidence,
            'intent_scores': intent_scores,
            'extracted_actions': actions,
            'entity_count': len(entities),
            'semantic_complexity': len(triples)
        }
    
    async def _analyze_cross_document_semantics(
        self,
        text: str,
        document_id: str,
        entities: List[SemanticEntity]
    ) -> Dict[str, Any]:
        """Analyze semantic relationships across documents."""
        
        # This would integrate with the context hierarchy engine for cross-document analysis
        # For now, return a placeholder structure
        
        return {
            'shared_entities': len([e for e in entities if e.entity_type in [EntityType.PERSON, EntityType.ORGANIZATION]]),
            'potential_connections': len(entities) * 0.3,  # Heuristic
            'cross_document_score': 0.5,  # Placeholder
            'analysis_method': 'placeholder_cross_document_analysis'
        }
    
    def _calculate_semantic_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall semantic understanding metrics."""
        
        # Extract metrics from analysis results
        entity_count = len(results.get('entities', []))
        triple_count = len(results.get('semantic_roles', []))
        intent_confidence = results.get('intent_analysis', {}).get('confidence', 0.0)
        
        # Calculate semantic richness
        semantic_richness = min((entity_count + triple_count) / 10.0, 1.0)
        
        # Calculate understanding completeness
        completeness = (
            0.3 * (1.0 if entity_count > 0 else 0.0) +
            0.3 * (1.0 if triple_count > 0 else 0.0) +
            0.4 * intent_confidence
        )
        
        # Calculate complexity score
        complexity = min((entity_count * 0.1 + triple_count * 0.2), 1.0)
        
        return {
            'semantic_richness': semantic_richness,
            'understanding_completeness': completeness,
            'semantic_complexity': complexity,
            'entity_coverage': min(entity_count / 5.0, 1.0),  # Assume 5 entities is good coverage
            'relationship_coverage': min(triple_count / 3.0, 1.0)  # Assume 3 triples is good coverage
        }
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics."""
        return {
            'components_enabled': {
                'entity_linking': self.entity_linker is not None,
                'semantic_role_labeling': self.srl_engine is not None,
                'intent_analysis': self.enable_intent_analysis
            },
            'performance_metrics': {
                'avg_processing_time': np.mean(self.analysis_metrics['processing_time'][-100:]) if self.analysis_metrics['processing_time'] else 0.0,
                'total_analyses': len(self.analysis_metrics['processing_time'])
            },
            'cache_statistics': {
                'entity_cache_size': len(self.entity_linker.entity_cache) if self.entity_linker else 0,
                'srl_cache_size': len(self.srl_engine.srl_cache) if self.srl_engine else 0
            }
        }

# Factory function
def create_advanced_semantic_engine(
    weight_sharing_engine: WeightSharingSemanticEngine,
    config: Dict[str, Any] = None
) -> AdvancedSemanticEngine:
    """Create and initialize advanced semantic engine."""
    config = config or {}
    
    return AdvancedSemanticEngine(
        weight_sharing_engine=weight_sharing_engine,
        enable_entity_linking=config.get('enable_entity_linking', True),
        enable_srl=config.get('enable_srl', True),
        enable_intent_analysis=config.get('enable_intent_analysis', True)
    )
