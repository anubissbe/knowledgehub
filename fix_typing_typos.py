#!/usr/bin/env python3
"""
Script to fix typing typos throughout the codebase.
"""

import os
import re
from pathlib import Path

def fix_typing_typos():
    """Fix common typing typos in import statements."""
    
    # Common typo patterns to fix
    typo_fixes = [
        # Pattern: (typo pattern, correct replacement)
        (r'from typing import ([^,\n]*), Ge, Optionalnerator', r'from typing import \1, Generator, Optional'),
        (r'from typing import Ge, Optionalnerator', r'from typing import Generator, Optional'),
        (r'from typing import ([^,\n]*), A, Optionalny', r'from typing import \1, Any, Optional'),
        (r'from typing import A, Optionalny', r'from typing import Any, Optional'),
        (r'from typing import ([^,\n]*), Optionalny', r'from typing import \1, Optional'),
        (r'from typing import Optionalny', r'from typing import Optional'),
        (r'from typing import ([^,\n]*), Listt', r'from typing import \1, List'),
        (r'from typing import Listt', r'from typing import List'),
        (r'from typing import ([^,\n]*), Dictt', r'from typing import \1, Dict'),
        (r'from typing import Dictt', r'from typing import Dict'),
        (r'from typing import ([^,\n]*), Anyy', r'from typing import \1, Any'),
        (r'from typing import Anyy', r'from typing import Any'),
        (r'from typing import ([^,\n]*), Unionn', r'from typing import \1, Union'),
        (r'from typing import Unionn', r'from typing import Union'),
        (r'from typing import ([^,\n]*), Tuplee', r'from typing import \1, Tuple'),
        (r'from typing import Tuplee', r'from typing import Tuple'),
        (r'from typing import ([^,\n]*), Sett', r'from typing import \1, Set'),
        (r'from typing import Sett', r'from typing import Set'),
        # Also fix some common adjacent typos
        (r'Optionalnerator', r'Optional, Generator'),
        (r'Generatorr', r'Generator'),
        (r'Optionall', r'Optional'),
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
    print("Fixing typing typos throughout the codebase...")
    fix_typing_typos()