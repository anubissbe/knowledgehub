"""Simple text chunking service without tiktoken dependency"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for chunking"""
    max_chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    content: str
    start_index: int
    end_index: int
    chunk_index: int
    metadata: Dict[str, Any]


class SmartChunker:
    """Smart text chunking with semantic awareness"""
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """Chunk text into smaller pieces"""
        if not text or not text.strip():
            return []
        
        # Clean text
        text = self._clean_text(text)
        
        # Split into chunks
        if self.config.preserve_paragraphs:
            chunks = self._chunk_by_paragraphs(text, metadata)
        elif self.config.preserve_sentences:
            chunks = self._chunk_by_sentences(text, metadata)
        else:
            chunks = self._chunk_by_size(text, metadata)
        
        # Filter out small chunks
        chunks = [
            chunk for chunk in chunks
            if len(chunk.content.strip()) >= self.config.min_chunk_size
        ]
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        return text.strip()
    
    def _chunk_by_paragraphs(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> List[TextChunk]:
        """Chunk text by paragraphs"""
        # Split by double newlines or multiple spaces
        paragraphs = re.split(r'\n\n+|\r\n\r\n+', text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds max size, create a chunk
            if current_chunk and len(current_chunk) + len(para) + 1 > self.config.max_chunk_size:
                chunks.append(TextChunk(
                    content=current_chunk,
                    start_index=current_start,
                    end_index=current_start + len(current_chunk),
                    chunk_index=chunk_index,
                    metadata=metadata or {}
                ))
                
                # Start new chunk with overlap
                if self.config.chunk_overlap > 0:
                    # Take last sentences for overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + para if overlap_text else para
                else:
                    current_chunk = para
                
                current_start = text.index(para)
                chunk_index += 1
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        
        # Add final chunk
        if current_chunk:
            chunks.append(TextChunk(
                content=current_chunk,
                start_index=current_start,
                end_index=current_start + len(current_chunk),
                chunk_index=chunk_index,
                metadata=metadata or {}
            ))
        
        return chunks
    
    def _chunk_by_sentences(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> List[TextChunk]:
        """Chunk text by sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence exceeds max size, create a chunk
            if current_chunk and len(current_chunk) + len(sentence) + 1 > self.config.max_chunk_size:
                chunks.append(TextChunk(
                    content=current_chunk,
                    start_index=current_start,
                    end_index=current_start + len(current_chunk),
                    chunk_index=chunk_index,
                    metadata=metadata or {}
                ))
                
                # Start new chunk with overlap
                if self.config.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    current_chunk = sentence
                
                current_start = text.index(sentence)
                chunk_index += 1
            else:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(TextChunk(
                content=current_chunk,
                start_index=current_start,
                end_index=current_start + len(current_chunk),
                chunk_index=chunk_index,
                metadata=metadata or {}
            ))
        
        return chunks
    
    def _chunk_by_size(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> List[TextChunk]:
        """Simple size-based chunking"""
        chunks = []
        chunk_index = 0
        
        for i in range(0, len(text), self.config.max_chunk_size - self.config.chunk_overlap):
            chunk_content = text[i:i + self.config.max_chunk_size]
            
            chunks.append(TextChunk(
                content=chunk_content,
                start_index=i,
                end_index=i + len(chunk_content),
                chunk_index=chunk_index,
                metadata=metadata or {}
            ))
            
            chunk_index += 1
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get text for overlap from end of chunk"""
        if len(text) <= self.config.chunk_overlap:
            return text
        
        # Try to find last sentence within overlap size
        overlap_start = len(text) - self.config.chunk_overlap
        overlap_text = text[overlap_start:]
        
        # Try to start at sentence boundary
        sentence_start = overlap_text.find('. ')
        if sentence_start > 0:
            overlap_text = overlap_text[sentence_start + 2:]
        
        return overlap_text.strip()
    
    def estimate_chunks(self, text: str) -> int:
        """Estimate number of chunks for text"""
        if not text:
            return 0
        
        text_length = len(text)
        effective_chunk_size = self.config.max_chunk_size - self.config.chunk_overlap
        
        return max(1, (text_length + effective_chunk_size - 1) // effective_chunk_size)
    
    async def process_chunk(self, chunk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a chunk and return processed chunks"""
        content = chunk_data.get("content", "")
        if not content:
            return []
        
        # Extract metadata
        metadata = {
            "url": chunk_data.get("url", ""),
            "type": chunk_data.get("type", "TEXT"),
            "source_name": chunk_data.get("source_name", ""),
            **chunk_data.get("metadata", {})
        }
        
        # Chunk the content
        text_chunks = self.chunk_text(content, metadata)
        
        # Convert to format expected by the processor
        processed_chunks = []
        for chunk in text_chunks:
            processed_chunks.append({
                "content": chunk.content,
                "url": metadata.get("url", ""),
                "type": metadata.get("type", "TEXT"),
                "metadata": {
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata
                }
            })
        
        return processed_chunks