"""Content parsers for different content types"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import re
from bs4 import BeautifulSoup
import json
from datetime import datetime
import hashlib

from ..shared.logging import setup_logging

logger = setup_logging("parsers")


class ContentParser(ABC):
    """Abstract base class for content parsers"""
    
    @abstractmethod
    async def parse(
        self,
        content: str,
        url: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse content and return chunks"""
        pass
    
    def _generate_chunk_id(self, content: str, index: int) -> str:
        """Generate unique chunk ID"""
        hash_input = f"{content[:100]}{index}".encode()
        return hashlib.md5(hash_input).hexdigest()


class HTMLParser(ContentParser):
    """Parser for HTML content"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    async def parse(
        self,
        content: str,
        url: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse HTML content into chunks"""
        soup = BeautifulSoup(content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract title
        title = soup.find("title")
        page_title = title.text.strip() if title else metadata.get("title", "")
        
        chunks = []
        
        # Extract main content sections
        content_sections = self._extract_content_sections(soup)
        
        for section in content_sections:
            # Create chunks from section
            section_chunks = self._create_chunks_from_section(
                section,
                url,
                page_title
            )
            chunks.extend(section_chunks)
        
        # If no structured content found, fall back to text extraction
        if not chunks:
            text = soup.get_text(separator="\n", strip=True)
            chunks = self._create_text_chunks(text, url, page_title)
        
        return chunks
    
    def _extract_content_sections(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract structured content sections from HTML"""
        sections = []
        
        # Look for article content
        article = soup.find("article") or soup.find("main")
        if article:
            sections.extend(self._extract_from_element(article))
            return sections
        
        # Look for content divs
        content_divs = soup.find_all("div", class_=re.compile(r"content|post|article", re.I))
        for div in content_divs[:3]:  # Limit to avoid too many chunks
            sections.extend(self._extract_from_element(div))
        
        # Extract sections by headings
        for heading in soup.find_all(["h1", "h2", "h3"]):
            section = {
                "type": "HEADING",
                "heading": heading.text.strip(),
                "content": [],
                "level": int(heading.name[1])
            }
            
            # Get content until next heading
            current = heading.next_sibling
            while current and current.name not in ["h1", "h2", "h3"]:
                if current.name == "p":
                    section["content"].append(current.text.strip())
                elif hasattr(current, "text"):
                    text = current.text.strip()
                    if text:
                        section["content"].append(text)
                current = current.next_sibling
            
            if section["content"]:
                sections.append(section)
        
        return sections
    
    def _extract_from_element(self, element) -> List[Dict[str, Any]]:
        """Extract structured content from an element"""
        sections = []
        
        # Extract paragraphs
        paragraphs = element.find_all("p")
        if paragraphs:
            content = [p.text.strip() for p in paragraphs if p.text.strip()]
            if content:
                sections.append({
                    "type": "text",
                    "content": content
                })
        
        # Extract lists
        for list_elem in element.find_all(["ul", "ol"]):
            items = [li.text.strip() for li in list_elem.find_all("li")]
            if items:
                sections.append({
                    "type": "list",
                    "content": items
                })
        
        # Extract code blocks
        for code in element.find_all(["code", "pre"]):
            code_text = code.text.strip()
            if code_text:
                sections.append({
                    "type": "CODE",
                    "content": [code_text],
                    "language": code.get("class", [""])[0] if code.get("class") else ""
                })
        
        return sections
    
    def _create_chunks_from_section(
        self,
        section: Dict[str, Any],
        url: str,
        page_title: str
    ) -> List[Dict[str, Any]]:
        """Create chunks from a content section"""
        chunks = []
        
        # Join content
        if section["type"] == "HEADING":
            content = f"# {section['heading']}\n\n" + "\n\n".join(section["content"])
        else:
            content = "\n\n".join(section["content"])
        
        # Split into chunks if needed
        if len(content) > self.chunk_size:
            text_chunks = self._split_text(content)
            for i, chunk_text in enumerate(text_chunks):
                chunks.append({
                    "content": chunk_text,
                    "type": section["type"],
                    "url": url,
                    "metadata": {
                        "title": page_title,
                        "section_type": section["type"],
                        "chunk_index": i,
                        "total_chunks": len(text_chunks)
                    }
                })
        else:
            chunks.append({
                "content": content,
                "type": section["type"],
                "url": url,
                "metadata": {
                    "title": page_title,
                    "section_type": section["type"]
                }
            })
        
        return chunks
    
    def _create_text_chunks(
        self,
        text: str,
        url: str,
        page_title: str
    ) -> List[Dict[str, Any]]:
        """Create chunks from plain text"""
        chunks = []
        text_chunks = self._split_text(text)
        
        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                "content": chunk_text,
                "type": "text",
                "url": url,
                "metadata": {
                    "title": page_title,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks)
                }
            })
        
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Add overlap between chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    # Add end of previous chunk
                    prev_chunk = chunks[i-1]
                    overlap_text = prev_chunk[-self.chunk_overlap:]
                    chunk = overlap_text + "\n\n" + chunk
                
                overlapped_chunks.append(chunk)
            
            chunks = overlapped_chunks
        
        return chunks


class MarkdownParser(ContentParser):
    """Parser for Markdown content"""
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
    
    async def parse(
        self,
        content: str,
        url: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse Markdown content into chunks"""
        chunks = []
        
        # Split by headers
        sections = re.split(r'^(#{1,6}\s+.*?)$', content, flags=re.MULTILINE)
        
        current_section = ""
        current_header = ""
        
        for i, section in enumerate(sections):
            if re.match(r'^#{1,6}\s+', section):
                # This is a header
                if current_section:
                    chunks.append({
                        "content": current_header + "\n\n" + current_section.strip(),
                        "type": "text",
                        "url": url,
                        "metadata": {
                            "header": current_header.strip(),
                            "format": "markdown"
                        }
                    })
                current_header = section
                current_section = ""
            else:
                current_section += section
        
        # Add last section
        if current_section:
            chunks.append({
                "content": current_header + "\n\n" + current_section.strip(),
                "type": "text",
                "url": url,
                "metadata": {
                    "header": current_header.strip() if current_header else "Introduction",
                    "format": "markdown"
                }
            })
        
        return chunks


class CodeParser(ContentParser):
    """Parser for source code files"""
    
    async def parse(
        self,
        content: str,
        url: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse code content"""
        # Detect language from URL or metadata
        language = self._detect_language(url, metadata)
        
        # For now, treat entire file as one chunk
        # Could be improved to split by functions/classes
        return [{
            "content": content,
            "type": "CODE",
            "url": url,
            "metadata": {
                "language": language,
                "format": "source_code"
            }
        }]
    
    def _detect_language(self, url: str, metadata: Dict[str, Any]) -> str:
        """Detect programming language from URL"""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php"
        }
        
        for ext, lang in extension_map.items():
            if url.endswith(ext):
                return lang
        
        return metadata.get("language", "unknown")


class ContentParserFactory:
    """Factory for creating appropriate content parsers"""
    
    def __init__(self):
        self.parsers = {
            "text/html": HTMLParser(),
            "text/markdown": MarkdownParser(),
            "text/plain": MarkdownParser(),  # Use markdown parser for plain text
            "application/x-python": CodeParser(),
            "text/x-python": CodeParser(),
            "application/javascript": CodeParser(),
            "text/javascript": CodeParser()
        }
    
    def get_parser(self, content_type: str) -> ContentParser:
        """Get appropriate parser for content type"""
        # Normalize content type
        content_type = content_type.lower().split(";")[0].strip()
        
        # Check for exact match
        if content_type in self.parsers:
            return self.parsers[content_type]
        
        # Check for partial matches
        if "html" in content_type:
            return self.parsers["text/html"]
        elif "markdown" in content_type or "md" in content_type:
            return self.parsers["text/markdown"]
        elif any(code_type in content_type for code_type in ["python", "javascript", "java", "code"]):
            return CodeParser()
        
        # Default to HTML parser
        return self.parsers["text/html"]