#!/usr/bin/env python3
"""Fix all Config imports in the codebase"""

import os
import re

def fix_config_imports():
    """Replace 'from ..config import settings' with 'from ..config import settings'"""
    
    # Walk through all Python files in api directory
    for root, dirs, files in os.walk('/opt/projects/knowledgehub/api'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    # Check if file has the problematic import
                    if 'from ..config import settings' in content or 'from api.config import settings as Config' in content:
                        # Replace the imports
                        original_content = content
                        content = content.replace('from ..config import settings', 'from ..config import settings')
                        content = content.replace('from api.config import settings as Config', 'from api.config import settings as Config')
                        
                        if content != original_content:
                            with open(filepath, 'w') as f:
                                f.write(content)
                            print(f"Fixed: {filepath}")
                
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    fix_config_imports()
    print("Config import fixes complete!")