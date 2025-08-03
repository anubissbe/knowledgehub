#!/usr/bin/env python3
"""
Fix SQLAlchemy duplicate class registration issues.
"""

import os
import re
from pathlib import Path

def fix_sqlalchemy_duplicates():
    """Fix duplicate SQLAlchemy model registrations."""
    
    api_dir = Path('/opt/projects/knowledgehub/api')
    
    # Rename the memory system Memory class to MemorySystemMemory to avoid conflicts
    memory_system_model_file = api_dir / 'memory_system' / 'models' / 'memory.py'
    
    if memory_system_model_file.exists():
        with open(memory_system_model_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Rename the class
        content = re.sub(r'class Memory\(Base\):', 'class MemorySystemMemory(Base):', content)
        
        # Update relationship references
        content = re.sub(r'relationship\("Memory"', 'relationship("MemorySystemMemory"', content)
        
        with open(memory_system_model_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Renamed Memory to MemorySystemMemory in {memory_system_model_file}")
    
    # Now update all imports in memory_system files
    fixes = 0
    
    for root, dirs, files in os.walk(api_dir / 'memory_system'):
        if '__pycache__' in root:
            continue
            
        for file in files:
            if not file.endswith('.py'):
                continue
                
            file_path = Path(root) / file
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Update imports
                content = re.sub(
                    r'from \.models\.memory import Memory\b',
                    'from .models.memory import MemorySystemMemory',
                    content
                )
                content = re.sub(
                    r'from \.\.models\.memory import Memory\b',
                    'from ..models.memory import MemorySystemMemory',
                    content
                )
                content = re.sub(
                    r'from api\.memory_system\.models\.memory import Memory\b',
                    'from api.memory_system.models.memory import MemorySystemMemory',
                    content
                )
                
                # Update class references
                content = re.sub(r'\bMemory\(', 'MemorySystemMemory(', content)
                content = re.sub(r'isinstance\([^,]+,\s*Memory\)', lambda m: m.group(0).replace('Memory)', 'MemorySystemMemory)'), content)
                content = re.sub(r'List\[Memory\]', 'List[MemorySystemMemory]', content)
                content = re.sub(r'Optional\[Memory\]', 'Optional[MemorySystemMemory]', content)
                
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixes += 1
                    print(f"Updated imports in: {file_path}")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Fix relationship references in main models
    main_memory_file = api_dir / 'models' / 'memory.py'
    
    if main_memory_file.exists():
        with open(main_memory_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Use just class name in relationships within same module
        content = re.sub(
            r'relationship\("api\.models\.memory\.Memory"',
            'relationship("Memory"',
            content
        )
        
        with open(main_memory_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Fixed relationships in {main_memory_file}")
    
    print(f"\nTotal files fixed: {fixes + 2}")

if __name__ == "__main__":
    print("Fixing SQLAlchemy duplicate class registrations...")
    fix_sqlalchemy_duplicates()