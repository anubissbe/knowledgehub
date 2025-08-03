#!/usr/bin/env python3
"""
Script to fix all typos throughout the codebase.
"""

import os
import re
from pathlib import Path

def fix_all_typos():
    """Fix all common typos in import statements."""
    
    # Comprehensive typo patterns to fix
    typo_fixes = [
        # HTTPException typos
        (r'HTTPExceptio\b', r'HTTPException'),
        
        # datetime typos
        (r',\s*datetimen', r''),  # Remove incorrect datetimen import
        
        # Response typos
        (r'Respo,\s*Optionalnse', r'Response'),
        (r'Respo\b', r'Response'),
        
        # Typing typos
        (r',\s*Optionaln\b', r', Optional'),
        (r',\s*Optionalnse\b', r''),
        (r',\s*Optionalny\b', r', Optional'),
        (r'Optionalny\b', r'Optional'),
        (r'Optionalnse\b', r'Optional'),
        (r'Optionaln\b', r'Optional'),
        
        # Generator typos
        (r'Ge,\s*Optionalnerator', r'Generator, Optional'),
        (r'Asy,\s*OptionalncGenerator', r'AsyncGenerator, Optional'),
        (r'Generatorr\b', r'Generator'),
        (r'AsyncGeneratorr\b', r'AsyncGenerator'),
        
        # Other typing imports
        (r'A,\s*Optionalny', r'Any, Optional'),
        (r'Listt\b', r'List'),
        (r'Dictt\b', r'Dict'),
        (r'Anyy\b', r'Any'),
        (r'Unionn\b', r'Union'),
        (r'Tuplee\b', r'Tuple'),
        (r'Sett\b', r'Set'),
        
        # Fix incomplete typing imports
        (r'from typing import ([^,\n]*),\s*$', r'from typing import \1'),
    ]
    
    api_dir = Path('/opt/projects/knowledgehub/api')
    
    files_fixed = 0
    total_fixes = 0
    
    for root, dirs, files in os.walk(api_dir):
        if '__pycache__' in root:
            continue
        
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    fixes_in_file = 0
                    
                    # Apply all typo fixes
                    for pattern, replacement in typo_fixes:
                        new_content, count = re.subn(pattern, replacement, content)
                        if count > 0:
                            content = new_content
                            fixes_in_file += count
                    
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        files_fixed += 1
                        total_fixes += fixes_in_file
                        print(f"Fixed {fixes_in_file} typo(s) in: {file_path.relative_to(api_dir)}")
                
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    print(f"\nSummary:")
    print(f"- Files fixed: {files_fixed}")
    print(f"- Total typos fixed: {total_fixes}")

if __name__ == "__main__":
    print("Fixing all typos throughout the codebase...")
    fix_all_typos()