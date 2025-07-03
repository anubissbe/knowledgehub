"""Smart content chunking for optimal RAG performance"""

import re
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import tiktoken
from bs4 import BeautifulSoup

from ..shared.logging import setup_logging

logger = setup_logging("chunker")


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies"""
    
    @abstractmethod
    async def chunk(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chunk content according to strategy"""
        pass


class TextChunkingStrategy(ChunkingStrategy):
    """Strategy for chunking plain text content"""
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Token counter
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None
    
    async def chunk(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chunk text content intelligently"""
        chunks = []
        
        # Split into paragraphs first
        paragraphs = self._split_paragraphs(content)
        
        # Group paragraphs into chunks
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = self._get_token_count(para)
            
            if current_size + para_size > self.chunk_size and current_chunk:
                # Create chunk from accumulated paragraphs
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, metadata))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > 1:
                    # Keep last paragraph for overlap
                    current_chunk = [current_chunk[-1]]
                    current_size = self._get_token_count(current_chunk[0])
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(para)
            current_size += para_size
        
        # Add remaining content
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(chunk_text, metadata))
        
        return chunks
    
    def _split_paragraphs(self, content: str) -> List[str]:
        """Split content into paragraphs"""
        # Split by double newlines
        paragraphs = content.split("\n\n")
        
        # Clean and filter
        cleaned = []
        for para in paragraphs:
            para = para.strip()
            if para:
                cleaned.append(para)
        
        return cleaned
    
    def _get_token_count(self, text: str) -> int:
        """Get token count for text"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback to word count approximation
            return int(len(text.split()) * 1.3)
    
    def _create_chunk(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create chunk dictionary"""
        return {
            "content": text,
            "type": "text",
            "metadata": {
                **metadata,
                "chunk_strategy": "text",
                "token_count": self._get_token_count(text)
            }
        }


class CodeChunkingStrategy(ChunkingStrategy):
    """Strategy for chunking code content"""
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
    
    async def chunk(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chunk code by logical units (functions, classes)"""
        language = metadata.get("language", "unknown")
        
        # For now, use simple line-based chunking
        # Could be enhanced with language-specific parsing
        chunks = []
        lines = content.split("\n")
        
        current_chunk = []
        current_size = 0
        
        for line in lines:
            current_chunk.append(line)
            current_size += len(line)
            
            # Check for logical boundaries
            if self._is_logical_boundary(line, language) and current_size > self.chunk_size / 2:
                chunk_content = "\n".join(current_chunk)
                chunks.append({
                    "content": chunk_content,
                    "type": "code",
                    "metadata": {
                        **metadata,
                        "chunk_strategy": "code",
                        "language": language
                    }
                })
                current_chunk = []
                current_size = 0
        
        # Add remaining content
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            chunks.append({
                "content": chunk_content,
                "type": "code",
                "metadata": {
                    **metadata,
                    "chunk_strategy": "code",
                    "language": language
                }
            })
        
        return chunks
    
    def _is_logical_boundary(self, line: str, language: str) -> bool:
        """Check if line represents a logical boundary"""
        line = line.strip()
        
        # Common patterns across languages
        if not line:  # Empty line
            return True
        
        # Language-specific patterns
        if language == "python":
            return line.startswith(("def ", "class ", "async def "))
        elif language in ["javascript", "typescript"]:
            return line.startswith(("function ", "class ", "export "))
        elif language == "java":
            return re.match(r"^\s*(public|private|protected).*\{", line) is not None
        
        return False


class MarkdownChunkingStrategy(ChunkingStrategy):
    """Strategy for chunking Markdown content"""
    
    def __init__(self, chunk_size: int = 800):
        self.chunk_size = chunk_size
    
    async def chunk(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chunk Markdown by sections"""
        chunks = []
        
        # Split by headers
        sections = self._split_by_headers(content)
        
        for section in sections:
            if len(section["content"]) > self.chunk_size:
                # Further split large sections
                sub_chunks = await self._split_large_section(section)
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    "content": section["content"],
                    "type": "text",
                    "metadata": {
                        **metadata,
                        "chunk_strategy": "markdown",
                        "section_title": section.get("title", ""),
                        "header_level": section.get("level", 0)
                    }
                })
        
        return chunks
    
    def _split_by_headers(self, content: str) -> List[Dict[str, Any]]:
        """Split Markdown content by headers"""
        sections = []
        lines = content.split("\n")
        
        current_section = {
            "title": "Introduction",
            "level": 0,
            "content": ""
        }
        
        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            
            if header_match:
                # Save current section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                current_section = {
                    "title": title,
                    "level": level,
                    "content": f"{line}\n"
                }
            else:
                current_section["content"] += f"{line}\n"
        
        # Add last section
        if current_section["content"].strip():
            sections.append(current_section)
        
        return sections
    
    async def _split_large_section(
        self,
        section: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Split a large section into smaller chunks"""
        # Use text chunking strategy for large sections
        text_chunker = TextChunkingStrategy(
            chunk_size=self.chunk_size,
            chunk_overlap=100
        )
        
        sub_chunks = await text_chunker.chunk(
            section["content"],
            {
                "section_title": section["title"],
                "header_level": section["level"]
            }
        )
        
        return sub_chunks


class SmartChunker:
    """Smart chunker that selects appropriate strategy"""
    
    def __init__(self):
        self.strategies = {
            "text": TextChunkingStrategy(),
            "code": CodeChunkingStrategy(),
            "markdown": MarkdownChunkingStrategy(),
            "html": TextChunkingStrategy()  # HTML is converted to text
        }
    
    async def process_chunk(
        self,
        chunk: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process a chunk with appropriate strategy"""
        content = chunk.get("content", "")
        chunk_type = chunk.get("type", "text")
        metadata = chunk.get("metadata", {})
        
        # Select strategy
        strategy = self._select_strategy(chunk_type, content, metadata)
        
        # Apply strategy
        processed_chunks = await strategy.chunk(content, metadata)
        
        # Add original URL to all chunks
        url = chunk.get("url", "")
        for processed_chunk in processed_chunks:
            processed_chunk["url"] = url
        
        return processed_chunks
    
    def _select_strategy(
        self,
        chunk_type: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> ChunkingStrategy:
        """Select appropriate chunking strategy"""
        # Check for code
        if chunk_type == "code" or metadata.get("language"):
            return self.strategies["code"]
        
        # Check for markdown
        if self._is_markdown(content) or metadata.get("format") == "markdown":
            return self.strategies["markdown"]
        
        # Default to text
        return self.strategies["text"]
    
    def _is_markdown(self, content: str) -> bool:
        """Check if content appears to be Markdown"""
        markdown_patterns = [
            r"^#{1,6}\s+",  # Headers
            r"\[.+\]\(.+\)",  # Links
            r"^\*\s+",  # Lists
            r"^\d+\.\s+",  # Numbered lists
            r"```",  # Code blocks
            r"\*\*.+\*\*",  # Bold
            r"\*.+\*"  # Italic
        ]
        
        for pattern in markdown_patterns:
            if re.search(pattern, content, re.MULTILINE):
                return True
        
        return False