"""
Entity extraction service using spaCy for named entity recognition.

This service extracts people, projects, technologies, and custom entities 
from conversation text to enable structured understanding and better memory organization.
"""

import logging
import re
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

try:
    import spacy
    from spacy import displacy
    from spacy.matcher import Matcher, PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities we can extract"""
    PERSON = "person"
    ORGANIZATION = "organization"
    TECHNOLOGY = "technology"
    PROJECT = "project"
    PROGRAMMING_LANGUAGE = "programming_language"
    FRAMEWORK = "framework"
    DATABASE = "database"
    TOOL = "tool"
    CONCEPT = "concept"
    LOCATION = "location"
    DATE = "date"
    URL = "url"
    EMAIL = "email"
    FILE_PATH = "file_path"
    FUNCTION_NAME = "function_name"
    CLASS_NAME = "class_name"
    VARIABLE_NAME = "variable_name"
    ERROR_TYPE = "error_type"


@dataclass
class ExtractedEntity:
    """Represents an extracted entity with metadata"""
    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float
    context: str
    metadata: Dict[str, Any]
    canonical_form: Optional[str] = None
    aliases: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class TechnologyPatterns:
    """Predefined patterns for technology-related entities"""
    
    PROGRAMMING_LANGUAGES = {
        'python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'csharp', 
        'php', 'ruby', 'go', 'rust', 'swift', 'kotlin', 'scala', 'r',
        'matlab', 'sql', 'html', 'css', 'bash', 'shell', 'powershell',
        'dart', 'elixir', 'erlang', 'haskell', 'lua', 'perl', 'julia'
    }
    
    FRAMEWORKS = {
        'react', 'angular', 'vue', 'svelte', 'nextjs', 'nuxt', 'gatsby',
        'express', 'fastapi', 'django', 'flask', 'rails', 'laravel',
        'spring', 'springboot', 'dotnet', '.net', 'asp.net',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas',
        'numpy', 'opencv', 'matplotlib', 'plotly', 'bokeh'
    }
    
    DATABASES = {
        'postgresql', 'mysql', 'sqlite', 'mongodb', 'redis', 'elasticsearch',
        'cassandra', 'dynamodb', 'firebase', 'supabase', 'planetscale',
        'cockroachdb', 'mariadb', 'oracle', 'sqlserver', 'neo4j',
        'influxdb', 'clickhouse', 'snowflake', 'bigquery'
    }
    
    TOOLS = {
        'docker', 'kubernetes', 'jenkins', 'gitlab', 'github', 'bitbucket',
        'terraform', 'ansible', 'chef', 'puppet', 'vagrant', 'helm',
        'prometheus', 'grafana', 'kibana', 'splunk', 'datadog',
        'vscode', 'intellij', 'pycharm', 'atom', 'sublime', 'vim', 'emacs',
        'postman', 'insomnia', 'curl', 'wget', 'git', 'svn', 'mercurial'
    }
    
    CLOUD_PLATFORMS = {
        'aws', 'amazon', 'azure', 'gcp', 'google cloud', 'digitalocean',
        'heroku', 'vercel', 'netlify', 'cloudflare', 'linode', 'vultr'
    }


class IntelligentEntityExtractor:
    """
    Intelligent entity extraction service using spaCy and custom patterns.
    
    Extracts and categorizes entities from conversation text including:
    - People and organizations
    - Technologies (languages, frameworks, databases, tools)
    - Projects and concepts
    - Technical entities (functions, classes, variables)
    - URLs, emails, file paths
    - Dates and locations
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self.matcher = None
        self.phrase_matcher = None
        self.tech_patterns = TechnologyPatterns()
        self.custom_entities = set()
        
        # Pattern collections
        self.tech_terms = (
            self.tech_patterns.PROGRAMMING_LANGUAGES |
            self.tech_patterns.FRAMEWORKS |
            self.tech_patterns.DATABASES |
            self.tech_patterns.TOOLS |
            self.tech_patterns.CLOUD_PLATFORMS
        )
        
        # Initialize if spaCy is available
        if SPACY_AVAILABLE:
            try:
                self._initialize_model()
                logger.info("Entity extraction service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize spaCy model: {e}")
                self.nlp = None
        else:
            logger.warning("spaCy not available, entity extraction will use basic patterns only")
    
    def _initialize_model(self):
        """Initialize spaCy model and custom matchers"""
        try:
            self.nlp = spacy.load(self.model_name)
            self.matcher = Matcher(self.nlp.vocab)
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            
            # Add custom patterns
            self._add_custom_patterns()
            self._add_phrase_patterns()
            
        except OSError as e:
            logger.error(f"spaCy model '{self.model_name}' not found: {e}")
            raise
    
    def _add_custom_patterns(self):
        """Add custom regex-based patterns"""
        if not self.matcher:
            return
        
        # URL pattern
        url_pattern = [{"TEXT": {"REGEX": r"https?://[^\s]+"}}]
        self.matcher.add("URL", [url_pattern])
        
        # Email pattern
        email_pattern = [{"TEXT": {"REGEX": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"}}]
        self.matcher.add("EMAIL", [email_pattern])
        
        # File path patterns
        file_path_patterns = [
            [{"TEXT": {"REGEX": r"/[a-zA-Z0-9._/-]+"}}],  # Unix paths
            [{"TEXT": {"REGEX": r"[A-Z]:\\[a-zA-Z0-9._\\-]+"}}],  # Windows paths
            [{"TEXT": {"REGEX": r"\./[a-zA-Z0-9._/-]+"}}],  # Relative paths
        ]
        for pattern in file_path_patterns:
            self.matcher.add("FILE_PATH", [pattern])
        
        # Function name pattern (common programming conventions)
        function_patterns = [
            [{"TEXT": {"REGEX": r"[a-zA-Z_][a-zA-Z0-9_]*\(\)"}}],  # function()
        ]
        for pattern in function_patterns:
            self.matcher.add("FUNCTION", pattern)
        
        # Class name pattern (PascalCase)
        class_pattern = [[{"TEXT": {"REGEX": r"[A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)*"}}]]
        self.matcher.add("CLASS", class_pattern)
        
        # Variable/constant patterns
        variable_patterns = [
            [{"TEXT": {"REGEX": r"[A-Z_][A-Z0-9_]+"}}],  # CONSTANTS
            [{"TEXT": {"REGEX": r"[a-z_][a-zA-Z0-9_]*"}}],  # variables
        ]
        for pattern in variable_patterns:
            self.matcher.add("VARIABLE", pattern)
        
        # Common error types
        error_pattern = [[{"TEXT": {"REGEX": r"[A-Z][a-zA-Z]*(?:Error|Exception|Warning)"}}]]
        self.matcher.add("ERROR_TYPE", error_pattern)
    
    def _add_phrase_patterns(self):
        """Add phrase-based patterns for technology terms"""
        if not self.phrase_matcher:
            return
        
        # Convert tech terms to spaCy docs for phrase matching
        tech_docs = [self.nlp(term) for term in self.tech_terms]
        self.phrase_matcher.add("TECHNOLOGY", tech_docs)
        
        # Add common project indicators
        project_indicators = ["project", "application", "app", "system", "platform", "service"]
        project_docs = [self.nlp(term) for term in project_indicators]
        self.phrase_matcher.add("PROJECT_INDICATOR", project_docs)
    
    def add_custom_entities(self, entities: List[str]):
        """Add custom entities for extraction"""
        self.custom_entities.update(entities)
        
        if self.phrase_matcher and self.nlp:
            custom_docs = [self.nlp(entity.lower()) for entity in entities]
            self.phrase_matcher.add("CUSTOM", custom_docs)
    
    async def extract_entities(self, text: str, context: Dict[str, Any] = None) -> List[ExtractedEntity]:
        """
        Extract entities from text using spaCy and custom patterns.
        
        Args:
            text: Text to extract entities from
            context: Additional context about the text
            
        Returns:
            List of ExtractedEntity objects
        """
        if not text or not text.strip():
            return []
        
        context = context or {}
        entities = []
        
        # Use spaCy if available, otherwise fallback to basic patterns
        if self.nlp:
            entities.extend(await self._extract_with_spacy(text, context))
        else:
            entities.extend(await self._extract_with_patterns(text, context))
        
        # Add custom entity extraction
        entities.extend(await self._extract_custom_entities(text, context))
        
        # Remove duplicates and merge overlapping entities
        entities = self._deduplicate_entities(entities)
        
        # Sort by confidence and position
        entities.sort(key=lambda e: (e.confidence, e.start_pos), reverse=True)
        
        return entities
    
    async def _extract_with_spacy(self, text: str, context: Dict[str, Any]) -> List[ExtractedEntity]:
        """Extract entities using spaCy NLP model"""
        entities = []
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                entity_type = self._map_spacy_label_to_entity_type(ent.label_)
                if entity_type:
                    entities.append(ExtractedEntity(
                        text=ent.text,
                        entity_type=entity_type,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        confidence=0.8,  # Base confidence for spaCy entities
                        context=self._get_entity_context(text, ent.start_char, ent.end_char),
                        metadata={
                            'spacy_label': ent.label_,
                            'spacy_confidence': getattr(ent, 'confidence', None)
                        }
                    ))
            
            # Extract custom pattern matches
            if self.matcher:
                matches = self.matcher(doc)
                for match_id, start, end in matches:
                    label = self.nlp.vocab.strings[match_id]
                    span = doc[start:end]
                    
                    entity_type = self._map_pattern_label_to_entity_type(label)
                    if entity_type:
                        entities.append(ExtractedEntity(
                            text=span.text,
                            entity_type=entity_type,
                            start_pos=span.start_char,
                            end_pos=span.end_char,
                            confidence=0.9,  # High confidence for pattern matches
                            context=self._get_entity_context(text, span.start_char, span.end_char),
                            metadata={'pattern_type': label}
                        ))
            
            # Extract phrase matches (technologies)
            if self.phrase_matcher:
                phrase_matches = self.phrase_matcher(doc)
                for match_id, start, end in phrase_matches:
                    label = self.nlp.vocab.strings[match_id]
                    span = doc[start:end]
                    
                    if label == "TECHNOLOGY":
                        entity_type = self._classify_technology(span.text.lower())
                        entities.append(ExtractedEntity(
                            text=span.text,
                            entity_type=entity_type,
                            start_pos=span.start_char,
                            end_pos=span.end_char,
                            confidence=0.85,
                            context=self._get_entity_context(text, span.start_char, span.end_char),
                            metadata={'tech_category': entity_type.value}
                        ))
            
        except Exception as e:
            logger.error(f"spaCy entity extraction failed: {e}")
        
        return entities
    
    async def _extract_with_patterns(self, text: str, context: Dict[str, Any]) -> List[ExtractedEntity]:
        """Extract entities using basic regex patterns (fallback)"""
        entities = []
        
        # URL pattern
        url_pattern = re.compile(r'https?://[^\s]+')
        for match in url_pattern.finditer(text):
            entities.append(ExtractedEntity(
                text=match.group(),
                entity_type=EntityType.URL,
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.95,
                context=self._get_entity_context(text, match.start(), match.end()),
                metadata={'pattern': 'regex_url'}
            ))
        
        # Email pattern
        email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        for match in email_pattern.finditer(text):
            entities.append(ExtractedEntity(
                text=match.group(),
                entity_type=EntityType.EMAIL,
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.95,
                context=self._get_entity_context(text, match.start(), match.end()),
                metadata={'pattern': 'regex_email'}
            ))
        
        # Technology terms (case-insensitive)
        text_lower = text.lower()
        for tech_term in self.tech_terms:
            pattern = re.compile(r'\b' + re.escape(tech_term) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                entity_type = self._classify_technology(tech_term)
                entities.append(ExtractedEntity(
                    text=match.group(),
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8,
                    context=self._get_entity_context(text, match.start(), match.end()),
                    metadata={'tech_category': entity_type.value, 'pattern': 'regex_tech'}
                ))
        
        return entities
    
    async def _extract_custom_entities(self, text: str, context: Dict[str, Any]) -> List[ExtractedEntity]:
        """Extract custom entities defined by user"""
        entities = []
        
        for custom_entity in self.custom_entities:
            pattern = re.compile(r'\b' + re.escape(custom_entity) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    entity_type=EntityType.CONCEPT,  # Default to concept for custom entities
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9,
                    context=self._get_entity_context(text, match.start(), match.end()),
                    metadata={'custom_entity': True}
                ))
        
        return entities
    
    def _map_spacy_label_to_entity_type(self, spacy_label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to our EntityType enum"""
        mapping = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,  # Geopolitical entity
            'LOC': EntityType.LOCATION,
            'DATE': EntityType.DATE,
            'TIME': EntityType.DATE,
            'PRODUCT': EntityType.TECHNOLOGY,
            'EVENT': EntityType.CONCEPT,
            'WORK_OF_ART': EntityType.CONCEPT,
            'LANGUAGE': EntityType.PROGRAMMING_LANGUAGE
        }
        return mapping.get(spacy_label)
    
    def _map_pattern_label_to_entity_type(self, pattern_label: str) -> Optional[EntityType]:
        """Map custom pattern labels to EntityType enum"""
        mapping = {
            'URL': EntityType.URL,
            'EMAIL': EntityType.EMAIL,
            'FILE_PATH': EntityType.FILE_PATH,
            'FUNCTION': EntityType.FUNCTION_NAME,
            'CLASS': EntityType.CLASS_NAME,
            'VARIABLE': EntityType.VARIABLE_NAME,
            'ERROR_TYPE': EntityType.ERROR_TYPE
        }
        return mapping.get(pattern_label)
    
    def _classify_technology(self, tech_term: str) -> EntityType:
        """Classify a technology term into specific type"""
        tech_term_lower = tech_term.lower()
        
        if tech_term_lower in self.tech_patterns.PROGRAMMING_LANGUAGES:
            return EntityType.PROGRAMMING_LANGUAGE
        elif tech_term_lower in self.tech_patterns.FRAMEWORKS:
            return EntityType.FRAMEWORK
        elif tech_term_lower in self.tech_patterns.DATABASES:
            return EntityType.DATABASE
        elif tech_term_lower in self.tech_patterns.TOOLS:
            return EntityType.TOOL
        else:
            return EntityType.TECHNOLOGY
    
    def _get_entity_context(self, text: str, start: int, end: int, context_size: int = 50) -> str:
        """Get surrounding context for an entity"""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        return text[context_start:context_end]
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate and overlapping entities"""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda e: e.start_pos)
        
        deduplicated = []
        for entity in entities:
            # Check for overlaps with existing entities
            overlapping = False
            for existing in deduplicated:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    # Overlapping entities - keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        break
                    else:
                        overlapping = True
                        break
            
            if not overlapping:
                deduplicated.append(entity)
        
        return deduplicated
    
    def get_entity_statistics(self, entities: List[ExtractedEntity]) -> Dict[str, Any]:
        """Get statistics about extracted entities"""
        if not entities:
            return {'total': 0}
        
        by_type = {}
        confidence_sum = 0
        
        for entity in entities:
            entity_type = entity.entity_type.value
            by_type[entity_type] = by_type.get(entity_type, 0) + 1
            confidence_sum += entity.confidence
        
        return {
            'total': len(entities),
            'by_type': by_type,
            'average_confidence': confidence_sum / len(entities) if entities else 0,
            'unique_entities': len(set(entity.text.lower() for entity in entities)),
            'entity_types_found': len(by_type)
        }
    
    async def extract_entities_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract entities from text chunks and add to metadata.
        
        Args:
            chunks: List of text chunks with content
            
        Returns:
            Updated chunks with entities added to metadata
        """
        updated_chunks = []
        
        for chunk in chunks:
            chunk_copy = chunk.copy()
            
            # Extract entities from chunk content
            entities = await self.extract_entities(
                chunk['content'], 
                context=chunk.get('metadata', {})
            )
            
            # Convert entities to serializable format
            entity_data = []
            for entity in entities:
                entity_data.append({
                    'text': entity.text,
                    'type': entity.entity_type.value,
                    'start_pos': entity.start_pos,
                    'end_pos': entity.end_pos,
                    'confidence': entity.confidence,
                    'context': entity.context,
                    'metadata': entity.metadata
                })
            
            # Update chunk with entities
            chunk_copy['entities'] = [entity.text for entity in entities]  # Simple list for compatibility
            chunk_copy['entity_details'] = entity_data  # Detailed entity information
            
            # Add entity statistics to metadata
            if 'metadata' not in chunk_copy:
                chunk_copy['metadata'] = {}
            
            stats = self.get_entity_statistics(entities)
            chunk_copy['metadata']['entity_stats'] = stats
            
            updated_chunks.append(chunk_copy)
        
        return updated_chunks


# Global service instance
entity_extractor = IntelligentEntityExtractor()


async def extract_entities_from_text(text: str, context: Dict[str, Any] = None) -> List[ExtractedEntity]:
    """
    Convenience function to extract entities from text.
    
    Args:
        text: Text to extract entities from
        context: Additional context
        
    Returns:
        List of ExtractedEntity objects
    """
    return await entity_extractor.extract_entities(text, context)


async def enrich_chunks_with_entities(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich text chunks with extracted entities.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Chunks enriched with entity information
    """
    return await entity_extractor.extract_entities_from_chunks(chunks)