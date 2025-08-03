#!/usr/bin/env python3
"""
Fix Memory class import conflicts by ensuring proper namespacing.
"""

import os
import re
from pathlib import Path

def fix_memory_imports():
    """Fix Memory class import conflicts."""
    
    api_dir = Path('/opt/projects/knowledgehub/api')
    
    # Files that should use the main Memory model from api.models.memory
    use_main_memory = [
        'services/memory_service.py',
        'services/session_service.py',
        'services/error_learning_service.py',
        'services/decision_service.py',
        'services/real_ai_intelligence.py',
        'routers/memories.py',
        'routers/session_management.py',
        'routers/error_tracking.py',
    ]
    
    # Files that should use the memory system Memory model
    use_memory_system = [
        'memory_system/services/',
        'memory_system/api/',
        'memory_system/core/',
    ]
    
    fixes_made = 0
    
    for root, dirs, files in os.walk(api_dir):
        # Skip __pycache__
        if '__pycache__' in root:
            continue
            
        for file in files:
            if not file.endswith('.py'):
                continue
                
            file_path = Path(root) / file
            relative_path = file_path.relative_to(api_dir)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Determine which Memory model this file should use
                should_use_main = any(str(relative_path).startswith(path) for path in use_main_memory)
                should_use_memory_system = any(str(relative_path).startswith(path) for path in use_memory_system)
                
                if should_use_main:
                    # Ensure imports from api.models.memory
                    if 'from models.memory import Memory' in content:
                        content = content.replace('from models.memory import Memory', 'from ..models.memory import Memory')
                    elif 'from .models.memory import Memory' in content:
                        # Already correct
                        pass
                    elif 'from memory_system.models.memory import Memory' in content:
                        # Wrong import - fix it
                        content = content.replace('from memory_system.models.memory import Memory', 'from ..models.memory import Memory')
                        
                elif should_use_memory_system:
                    # Ensure imports from memory_system.models.memory
                    if 'from models.memory import Memory' in content and 'memory_system' in str(relative_path):
                        content = content.replace('from models.memory import Memory', 'from ..models.memory import Memory')
                    elif 'from api.models.memory import Memory' in content:
                        # Wrong import for memory system files
                        content = content.replace('from api.models.memory import Memory', 'from ..models.memory import Memory')
                
                # Fix relationship references
                if 'relationship("Memory"' in content:
                    # Add explicit class reference to avoid ambiguity
                    if 'memory_system' in str(relative_path):
                        content = re.sub(
                            r'relationship\("Memory"',
                            'relationship("api.memory_system.models.memory.Memory"',
                            content
                        )
                    else:
                        content = re.sub(
                            r'relationship\("Memory"',
                            'relationship("api.models.memory.Memory"',
                            content
                        )
                
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixes_made += 1
                    print(f"Fixed imports in: {relative_path}")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"\nTotal files fixed: {fixes_made}")

if __name__ == "__main__":
    print("Fixing Memory class import conflicts...")
    fix_memory_imports()