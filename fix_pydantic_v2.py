#!/usr/bin/env python3
"""
Fix Pydantic v2 migration issues
"""

import os
import re

def fix_pydantic_validators(filepath):
    """Fix @validator to @field_validator"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Replace @validator with @field_validator
    content = re.sub(r'from pydantic import (.*?)validator', 
                     r'from pydantic import \1field_validator', content)
    content = re.sub(r'@validator\(', r'@field_validator(', content)
    
    # Fix class Config to ConfigDict
    if 'class Config:' in content:
        # Add ConfigDict import if not present
        if 'from pydantic import' in content and 'ConfigDict' not in content:
            content = re.sub(r'(from pydantic import .*?)\n', 
                           r'\1, ConfigDict\n', content, count=1)
        
        # Replace class Config with model_config
        content = re.sub(r'(\s+)class Config:\s*\n(\s+)(.+?)(?=\n\s*\n|\n\s*def|\n\s*@|\Z)', 
                       r'\1model_config = ConfigDict(\3)', content, flags=re.DOTALL)
        
        # Fix orm_mode to from_attributes
        content = content.replace('orm_mode = True', 'from_attributes = True')
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def fix_all_schemas():
    """Fix all schema files"""
    schemas_dir = '/opt/projects/knowledgehub/api/schemas'
    fixed_files = []
    
    for root, dirs, files in os.walk(schemas_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if fix_pydantic_validators(filepath):
                    fixed_files.append(filepath)
                    print(f"Fixed: {filepath}")
    
    # Also fix models directory
    models_dir = '/opt/projects/knowledgehub/api/models'
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if fix_pydantic_validators(filepath):
                    fixed_files.append(filepath)
                    print(f"Fixed: {filepath}")
    
    return fixed_files

if __name__ == "__main__":
    print("Fixing Pydantic v2 migration issues...")
    fixed = fix_all_schemas()
    print(f"\nFixed {len(fixed)} files")