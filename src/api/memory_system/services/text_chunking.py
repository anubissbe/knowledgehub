"""
Intelligent text chunking service for conversation processing.

This service splits conversation text into meaningful chunks while preserving context,
handling code blocks, and maintaining semantic boundaries.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of text chunks"""
    CONVERSATION = "conversation"
    CODE_BLOCK = "code_block"
    COMMAND = "command"
    ERROR_MESSAGE = "error_message"
    QUESTION = "question"
    ANSWER = "answer"
    INSTRUCTION = "instruction"
    FACT = "fact"
    DECISION = "decision"


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    content: str
    chunk_type: ChunkType
    start_index: int
    end_index: int
    metadata: Dict[str, Any]
    importance: float = 0.5
    contains_code: bool = False
    language: Optional[str] = None
    entities: List[str] = None
    semantic_boundary: bool = False
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []


class IntelligentTextChunker:
    """
    Intelligent text chunking service that preserves semantic boundaries
    and handles various types of conversation content.
    """
    
    def __init__(self, 
                 max_chunk_size: int = 512,
                 min_chunk_size: int = 50,
                 overlap_size: int = 50):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        
        # Patterns for different content types
        self.code_block_pattern = re.compile(
            r'```(\w+)?\n(.*?)```', 
            re.DOTALL | re.MULTILINE
        )
        
        self.inline_code_pattern = re.compile(r'`([^`]+)`')
        
        self.command_pattern = re.compile(
            r'(?:^|\n)\s*(?:\$|>|#)\s*([^\n]+)', 
            re.MULTILINE
        )
        
        self.error_pattern = re.compile(
            r'(?:error|exception|failed|traceback|warning)[::\s].*?(?:\n|$)',
            re.IGNORECASE | re.MULTILINE
        )
        
        self.question_pattern = re.compile(
            r'(?:^|\n)\s*(?:what|how|why|when|where|who|can|should|would|could|is|are|do|does|did)\s+.*?\?',
            re.IGNORECASE | re.MULTILINE
        )
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
        
        # Semantic boundary indicators
        self.topic_transitions = [
            r'\b(?:now|next|then|however|meanwhile|in contrast|on the other hand)\b',
            r'\b(?:step \d+|phase \d+|part \d+)\b',
            r'\b(?:first|second|third|finally|lastly)\b',
            r'\b(?:let\'s|now let\'s|moving on)\b'
        ]
        
        self.transition_pattern = re.compile(
            '|'.join(self.topic_transitions),
            re.IGNORECASE
        )
    
    def chunk_conversation(self, text: str, context: Dict[str, Any] = None) -> List[TextChunk]:
        """
        Main method to chunk conversation text into meaningful pieces.
        
        Args:
            text: The conversation text to chunk
            context: Additional context about the conversation
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        context = context or {}
        chunks = []
        
        logger.info(f"Starting text chunking for {len(text)} characters")
        
        # Step 1: Extract and preserve special content blocks
        preserved_blocks = self._extract_preserved_blocks(text)
        working_text = self._mask_preserved_blocks(text, preserved_blocks)
        
        # Step 2: Split into logical sections
        sections = self._split_into_sections(working_text)
        
        # Step 3: Process each section
        for section in sections:
            section_chunks = self._process_section(section, preserved_blocks, context)
            chunks.extend(section_chunks)
        
        # Step 4: Merge small chunks and add overlap
        final_chunks = self._optimize_chunks(chunks)
        
        # Step 5: Add metadata and analyze content
        self._enrich_chunks(final_chunks, context)
        
        logger.info(f"Created {len(final_chunks)} chunks from conversation")
        return final_chunks
    
    def _extract_preserved_blocks(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extract code blocks and other content that should be preserved as-is"""
        preserved = {}
        counter = 0
        
        # Extract code blocks
        for match in self.code_block_pattern.finditer(text):
            placeholder = f"__PRESERVED_BLOCK_{counter}__"
            preserved[placeholder] = {
                'type': 'code_block',
                'content': match.group(0),
                'language': match.group(1) or 'text',
                'start': match.start(),
                'end': match.end()
            }
            counter += 1
        
        # Extract command lines
        for match in self.command_pattern.finditer(text):
            placeholder = f"__PRESERVED_BLOCK_{counter}__"
            preserved[placeholder] = {
                'type': 'command',
                'content': match.group(0),
                'start': match.start(),
                'end': match.end()
            }
            counter += 1
        
        return preserved
    
    def _mask_preserved_blocks(self, text: str, preserved_blocks: Dict[str, Dict[str, Any]]) -> str:
        """Replace preserved blocks with placeholders for processing"""
        working_text = text
        
        # Sort by start position (reverse order to avoid index shifting)
        sorted_blocks = sorted(
            preserved_blocks.items(),
            key=lambda x: x[1]['start'],
            reverse=True
        )
        
        for placeholder, block_info in sorted_blocks:
            start = block_info['start']
            end = block_info['end']
            working_text = working_text[:start] + placeholder + working_text[end:]
        
        return working_text
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections based on paragraph breaks and topic transitions"""
        # First split by paragraph breaks
        paragraphs = self.paragraph_breaks.split(text)
        
        sections = []
        current_section = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check for topic transitions
            if self.transition_pattern.search(paragraph) and current_section:
                # Save current section and start new one
                sections.append(current_section.strip())
                current_section = paragraph
            else:
                # Add to current section
                if current_section:
                    current_section += "\n\n" + paragraph
                else:
                    current_section = paragraph
        
        # Add final section
        if current_section.strip():
            sections.append(current_section.strip())
        
        return sections
    
    def _process_section(self, section: str, preserved_blocks: Dict[str, Dict[str, Any]], 
                        context: Dict[str, Any]) -> List[TextChunk]:
        """Process a single section into chunks"""
        chunks = []
        
        # If section is small enough, treat as single chunk
        if len(section) <= self.max_chunk_size:
            chunk = self._create_chunk(section, preserved_blocks, context)
            chunks.append(chunk)
            return chunks
        
        # Split by sentences for larger sections
        sentences = self._split_sentences(section)
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed max size
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                # Create chunk from current content
                chunk = self._create_chunk(
                    current_chunk.strip(), 
                    preserved_blocks, 
                    context,
                    start_index=current_start
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence
                current_start = current_start + len(current_chunk) - len(overlap_text)
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(), 
                preserved_blocks, 
                context,
                start_index=current_start
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving meaning"""
        # Simple sentence splitting - can be enhanced with NLP libraries
        sentences = self.sentence_endings.split(text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) >= self.min_chunk_size:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _create_chunk(self, content: str, preserved_blocks: Dict[str, Dict[str, Any]], 
                     context: Dict[str, Any], start_index: int = 0) -> TextChunk:
        """Create a TextChunk object with metadata"""
        # Restore preserved blocks
        restored_content = self._restore_preserved_blocks(content, preserved_blocks)
        
        # Determine chunk type
        chunk_type = self._determine_chunk_type(restored_content)
        
        # Check for code content
        contains_code = bool(
            self.code_block_pattern.search(restored_content) or 
            self.inline_code_pattern.search(restored_content) or
            any(block['type'] == 'code_block' for block in preserved_blocks.values() 
                if block['content'] in restored_content)
        )
        
        # Detect language if code is present
        language = None
        if contains_code:
            language = self._detect_language(restored_content)
        
        # Calculate importance score
        importance = self._calculate_importance(restored_content, context)
        
        # Create metadata
        metadata = {
            'original_length': len(content),
            'processed_length': len(restored_content),
            'has_preserved_blocks': bool(preserved_blocks),
            'context': context.get('session_context', {}),
            'word_count': len(restored_content.split()),
            'line_count': len(restored_content.split('\n'))
        }
        
        return TextChunk(
            content=restored_content,
            chunk_type=chunk_type,
            start_index=start_index,
            end_index=start_index + len(restored_content),
            metadata=metadata,
            importance=importance,
            contains_code=contains_code,
            language=language,
            entities=[],  # Will be populated by entity extraction service
            semantic_boundary=True  # Assume semantic boundary for now
        )
    
    def _restore_preserved_blocks(self, content: str, preserved_blocks: Dict[str, Dict[str, Any]]) -> str:
        """Restore preserved blocks from placeholders"""
        restored = content
        
        for placeholder, block_info in preserved_blocks.items():
            if placeholder in restored:
                restored = restored.replace(placeholder, block_info['content'])
        
        return restored
    
    def _determine_chunk_type(self, content: str) -> ChunkType:
        """Determine the type of chunk based on content analysis"""
        content_lower = content.lower()
        
        # Check for code blocks
        if self.code_block_pattern.search(content):
            return ChunkType.CODE_BLOCK
        
        # Check for commands
        if self.command_pattern.search(content):
            return ChunkType.COMMAND
        
        # Check for error messages
        if self.error_pattern.search(content):
            return ChunkType.ERROR_MESSAGE
        
        # Check for questions
        if self.question_pattern.search(content):
            return ChunkType.QUESTION
        
        # Check for decision indicators
        decision_keywords = ['decide', 'chose', 'selected', 'will use', 'going with', 'prefer']
        if any(keyword in content_lower for keyword in decision_keywords):
            return ChunkType.DECISION
        
        # Check for instruction indicators
        instruction_keywords = ['should', 'must', 'need to', 'please', 'make sure', 'remember']
        if any(keyword in content_lower for keyword in instruction_keywords):
            return ChunkType.INSTRUCTION
        
        # Check for fact indicators (more specific patterns)
        fact_patterns = [
            r'^\s*the\s+\w+\s+(is|are|was|were|has|have|contains?|includes?)',
            r'^\s*\w+\s+(is|are|was|were)\s+',
            r'\s+(contains?|includes?|has|have)\s+\w+'
        ]
        if any(re.search(pattern, content_lower) for pattern in fact_patterns):
            return ChunkType.FACT
        
        # Default to conversation
        return ChunkType.CONVERSATION
    
    def _detect_language(self, content: str) -> Optional[str]:
        """Detect programming language from code content"""
        # Extract language from code blocks
        code_match = self.code_block_pattern.search(content)
        if code_match and code_match.group(1):
            return code_match.group(1)
        
        # Simple heuristics for language detection
        if 'import ' in content and ('def ' in content or 'class ' in content):
            return 'python'
        elif 'function ' in content or 'const ' in content or 'let ' in content:
            return 'javascript'
        elif '#include' in content or 'int main' in content:
            return 'c'
        elif 'public class' in content or 'import java' in content:
            return 'java'
        elif '<?php' in content:
            return 'php'
        elif 'SELECT ' in content.upper() or 'INSERT ' in content.upper():
            return 'sql'
        elif 'curl ' in content or 'wget ' in content:
            return 'bash'
        
        return None
    
    def _calculate_importance(self, content: str, context: Dict[str, Any]) -> float:
        """Calculate importance score for the chunk"""
        importance = 0.5  # Base importance
        
        # Length factor (longer chunks might be more important)
        length_factor = min(len(content) / 1000, 0.2)
        importance += length_factor
        
        # Code presence increases importance
        if self.code_block_pattern.search(content) or self.inline_code_pattern.search(content):
            importance += 0.2
        
        # Error messages are important
        if self.error_pattern.search(content):
            importance += 0.15
        
        # Questions are important
        if self.question_pattern.search(content):
            importance += 0.1
        
        # Decision content is important
        decision_keywords = ['decide', 'chose', 'selected', 'will use', 'going with']
        if any(keyword in content.lower() for keyword in decision_keywords):
            importance += 0.15
        
        # Context-based importance
        if context.get('is_critical', False):
            importance += 0.2
        
        # Ensure score is between 0 and 1
        return min(max(importance, 0.0), 1.0)
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text for chunk boundaries"""
        if len(text) <= self.overlap_size:
            return text
        
        # Try to find a good sentence boundary for overlap
        sentences = self.sentence_endings.split(text)
        if len(sentences) > 1:
            # Take the last complete sentence that fits in overlap size
            overlap = ""
            for sentence in reversed(sentences[-2:]):  # Only check last 2 sentences
                sentence = sentence.strip()
                if sentence and len(overlap + sentence) <= self.overlap_size:
                    if overlap:
                        overlap = sentence + ". " + overlap
                    else:
                        overlap = sentence
                else:
                    break
            if overlap:
                return overlap.strip()
        
        # Fallback to character-based overlap
        return text[-self.overlap_size:].strip()
    
    def _optimize_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Optimize chunks by merging small ones and ensuring proper boundaries"""
        if not chunks:
            return chunks
        
        optimized = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            
            # If chunk is too small, try to merge with next chunk
            if (len(current.content) < self.min_chunk_size and 
                i + 1 < len(chunks) and 
                len(current.content) + len(chunks[i + 1].content) <= self.max_chunk_size):
                
                next_chunk = chunks[i + 1]
                
                # Merge chunks
                merged_content = current.content + "\n\n" + next_chunk.content
                merged_metadata = {**current.metadata, **next_chunk.metadata}
                merged_metadata['merged'] = True
                merged_metadata['original_chunks'] = 2
                
                merged_chunk = TextChunk(
                    content=merged_content,
                    chunk_type=current.chunk_type,  # Keep first chunk's type
                    start_index=current.start_index,
                    end_index=next_chunk.end_index,
                    metadata=merged_metadata,
                    importance=max(current.importance, next_chunk.importance),
                    contains_code=current.contains_code or next_chunk.contains_code,
                    language=current.language or next_chunk.language,
                    entities=current.entities + next_chunk.entities,
                    semantic_boundary=next_chunk.semantic_boundary
                )
                
                optimized.append(merged_chunk)
                i += 2  # Skip both chunks
            else:
                optimized.append(current)
                i += 1
        
        return optimized
    
    def _enrich_chunks(self, chunks: List[TextChunk], context: Dict[str, Any]) -> None:
        """Enrich chunks with additional metadata and analysis"""
        for i, chunk in enumerate(chunks):
            # Add position information
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)
            chunk.metadata['is_first'] = i == 0
            chunk.metadata['is_last'] = i == len(chunks) - 1
            
            # Add context references
            if context:
                chunk.metadata['session_id'] = context.get('session_id')
                chunk.metadata['user_id'] = context.get('user_id')
                chunk.metadata['timestamp'] = context.get('timestamp')
            
            # Calculate readability metrics
            chunk.metadata['readability'] = self._calculate_readability(chunk.content)
            
            # Identify key phrases (simple implementation)
            chunk.metadata['key_phrases'] = self._extract_key_phrases(chunk.content)
    
    def _calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate basic readability metrics"""
        words = text.split()
        sentences = len(self.sentence_endings.split(text))
        
        if not words or sentences == 0:
            return {'words_per_sentence': 0, 'avg_word_length': 0}
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        words_per_sentence = len(words) / sentences
        
        return {
            'words_per_sentence': round(words_per_sentence, 2),
            'avg_word_length': round(avg_word_length, 2)
        }
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text (simple implementation)"""
        # This is a simple implementation - can be enhanced with NLP libraries
        key_phrases = []
        
        # Look for quoted text
        quoted_pattern = re.compile(r'"([^"]+)"')
        quotes = quoted_pattern.findall(text)
        key_phrases.extend(quotes)
        
        # Look for technical terms (capitalized words)
        tech_pattern = re.compile(r'\b[A-Z][a-z]*[A-Z][a-zA-Z]*\b')
        tech_terms = tech_pattern.findall(text)
        key_phrases.extend(tech_terms)
        
        # Look for function/method names
        function_pattern = re.compile(r'\b\w+\(\)')
        functions = function_pattern.findall(text)
        key_phrases.extend(functions)
        
        return list(set(key_phrases))[:10]  # Return unique phrases, max 10


# Service instance
text_chunker = IntelligentTextChunker()


def chunk_conversation_text(text: str, context: Dict[str, Any] = None) -> List[TextChunk]:
    """
    Convenience function to chunk conversation text.
    
    Args:
        text: The conversation text to chunk
        context: Additional context about the conversation
        
    Returns:
        List of TextChunk objects
    """
    return text_chunker.chunk_conversation(text, context)


async def process_conversation_chunks(text: str, session_id: str, user_id: str) -> List[Dict[str, Any]]:
    """
    Process conversation text into chunks and return serializable format.
    
    Args:
        text: The conversation text to process
        session_id: ID of the session
        user_id: ID of the user
        
    Returns:
        List of chunk dictionaries ready for storage
    """
    context = {
        'session_id': session_id,
        'user_id': user_id,
        'timestamp': None  # Can be added if needed
    }
    
    chunks = chunk_conversation_text(text, context)
    
    # Convert to serializable format
    serializable_chunks = []
    for chunk in chunks:
        chunk_dict = {
            'content': chunk.content,
            'chunk_type': chunk.chunk_type.value,
            'start_index': chunk.start_index,
            'end_index': chunk.end_index,
            'metadata': chunk.metadata,
            'importance': chunk.importance,
            'contains_code': chunk.contains_code,
            'language': chunk.language,
            'entities': chunk.entities,
            'semantic_boundary': chunk.semantic_boundary
        }
        serializable_chunks.append(chunk_dict)
    
    return serializable_chunks