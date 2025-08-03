#!/usr/bin/env python3
"""Fix remaining Memory imports to use MemorySystemMemory"""

import os
import re

def fix_memory_imports():
    """Fix all remaining Memory imports"""
    
    files_to_fix = [
        "api/memory_system/core/persistent_context.py",
        "api/memory_system/core/session_lifecycle.py",
        "api/memory_system/core/context_compression.py",
        "api/memory_system/services/memory_search_service.py",
        "api/memory_system/services/persistent_context_service.py",
        "api/memory_system/seed_data.py",
        "api/memory_system/api/routers/vector_search.py",
        "api/memory_system/api/routers/context.py",
        "api/memory_system/api/routers/context_compression.py"
    ]
    
    replacements = [
        # Import replacements
        (r"from \.\.models import Memory,", "from ..models import MemorySystemMemory,"),
        (r"from \.\.models import MemorySession, Memory,", "from ..models import MemorySession, MemorySystemMemory,"),
        (r"from \.\.\.models import Memory,", "from ...models import MemorySystemMemory,"),
        (r"from \.\.models\.memory import MemorySession, Memory,", "from ..models.memory import MemorySession, MemorySystemMemory,"),
        
        # Type annotations
        (r"memories: List\[Memory\]", "memories: List[MemorySystemMemory]"),
        (r"memory: Memory([,\)])", r"memory: MemorySystemMemory\1"),
        (r"-> Memory:", "-> MemorySystemMemory:"),
        (r"-> List\[Memory\]", "-> List[MemorySystemMemory]"),
        (r"-> Optional\[Memory\]", "-> Optional[MemorySystemMemory]"),
        (r": Dict\[str, Memory\]", ": Dict[str, MemorySystemMemory]"),
        
        # Query references
        (r"db\.query\(Memory\)", "db.query(MemorySystemMemory)"),
        (r"query = db\.query\(Memory\)", "query = db.query(MemorySystemMemory)"),
        (r"Memory\.session_id", "MemorySystemMemory.session_id"),
        (r"Memory\.memory_type", "MemorySystemMemory.memory_type"),
        (r"Memory\.importance", "MemorySystemMemory.importance"),
        (r"Memory\.created_at", "MemorySystemMemory.created_at"),
        (r"Memory\.content", "MemorySystemMemory.content"),
        (r"Memory\.summary", "MemorySystemMemory.summary"),
        (r"Memory\.entities", "MemorySystemMemory.entities"),
        (r"Memory\.confidence", "MemorySystemMemory.confidence"),
        (r"Memory\.embedding", "MemorySystemMemory.embedding"),
        (r"Memory\.access_count", "MemorySystemMemory.access_count"),
        
        # Class instantiation
        (r"memory = Memory\(", "memory = MemorySystemMemory("),
        (r"new_memory = Memory\(", "new_memory = MemorySystemMemory("),
    ]
    
    fixed_count = 0
    
    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            print(f"Skipping {file_path} - not found")
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all replacements
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Fixed {file_path}")
            fixed_count += 1
        else:
            print(f"No changes needed in {file_path}")
    
    return fixed_count

if __name__ == "__main__":
    print("Fixing remaining Memory imports...")
    fixed = fix_memory_imports()
    print(f"\nFixed {fixed} files")