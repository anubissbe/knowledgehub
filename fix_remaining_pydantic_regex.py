#!/usr/bin/env python3
"""Fix remaining regex to pattern conversions for Pydantic v2"""

import re
import os
import glob

def fix_regex_to_pattern():
    """Fix all remaining regex to pattern conversions"""
    
    # Find all Python files
    python_files = []
    for pattern in ['api/**/*.py', 'api/*.py']:
        python_files.extend(glob.glob(pattern, recursive=True))
    
    fixed_count = 0
    
    for file_path in python_files:
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # Replace regex= with pattern= in Field() calls
            # This regex matches Field(...regex=...) and replaces regex with pattern
            content = re.sub(
                r'Field\s*\([^)]*\bregex\s*=',
                lambda m: m.group(0).replace('regex=', 'pattern='),
                content
            )
            
            if content != original_content:
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f"Fixed {file_path}")
                fixed_count += 1
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return fixed_count

if __name__ == "__main__":
    print("Fixing remaining regex to pattern conversions...")
    fixed = fix_regex_to_pattern()
    print(f"\nFixed {fixed} files")