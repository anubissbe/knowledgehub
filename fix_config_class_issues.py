#!/usr/bin/env python3
"""
Script to fix specific Config class usage issues.

This script fixes:
1. Files that import settings but use Config in type hints
2. Files that try to instantiate Config() when they should use settings
3. Files that need proper Config class imports
"""

import os
import re
from pathlib import Path

def fix_specific_config_issues():
    """Fix specific Config usage issues in known problematic files."""
    
    # Files that import settings as Config and need fixing
    files_to_fix = [
        '/opt/projects/knowledgehub/api/services/opentelemetry_tracing.py',
        '/opt/projects/knowledgehub/api/services/real_copilot_enhancement.py',
        '/opt/projects/knowledgehub/api/services/alert_service.py',
    ]
    
    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # These files import settings but try to use Config
        if 'from ..config import settings' in content:
            # Fix type hints - change Optional[Config] to Optional[Any] or remove the type hint
            content = re.sub(
                r'def __init__\(self, config: Optional\[Config\] = None\):',
                'def __init__(self, config=None):',
                content
            )
            
            # Fix the instantiation - change config or Config to config or settings
            content = re.sub(
                r'self\.config = config or Config(?:\(\))?',
                'self.config = config or settings',
                content
            )
            
            # Remove any remaining Config references that should be settings
            content = re.sub(r'\bConfig\b', 'settings', content)
            
            # Make sure we don't have settings.settings
            content = re.sub(r'settings\.settings', 'settings', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
    
    # Now fix files that properly import Config but have instantiation issues
    files_with_shared_config = []
    
    api_dir = Path('/opt/projects/knowledgehub/api')
    for root, dirs, files in os.walk(api_dir):
        if '__pycache__' in root:
            continue
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if 'from shared.config import Config' in content:
                        files_with_shared_config.append(file_path)
                except Exception:
                    pass
    
    # These files have proper Config import and usage is correct
    print(f"\nFiles with proper Config import: {len(files_with_shared_config)}")
    
    # Fix files that need Config imported but don't have it
    for root, dirs, files in os.walk(api_dir):
        if '__pycache__' in root:
            continue
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check if file uses Config but doesn't import it properly
                    if 'Optional[Config]' in content and 'from shared.config import Config' not in content and 'from ..config import settings' not in content:
                        # This file needs Config import
                        original_content = content
                        
                        # Add the import after other imports
                        import_section = re.search(r'((?:from|import)[^\n]+\n)+', content)
                        if import_section:
                            # Check if we can use relative import to shared.config
                            if '/api/' in file_path:
                                content = content.replace(
                                    import_section.group(0),
                                    import_section.group(0) + "from shared.config import Config\n"
                                )
                            
                            if content != original_content:
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(content)
                                print(f"Added Config import to: {file_path}")
                
                except Exception as e:
                    pass

def check_remaining_issues():
    """Check for any remaining Config-related issues."""
    
    api_dir = Path('/opt/projects/knowledgehub/api')
    issues = []
    
    for root, dirs, files in os.walk(api_dir):
        if '__pycache__' in root:
            continue
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for problematic patterns
                    if 'Optional[Config]' in content:
                        if 'from shared.config import Config' not in content and 'from ..config import settings' not in content:
                            issues.append((file_path, "Uses Optional[Config] but doesn't import Config"))
                    
                    if re.search(r'Config\(\)', content):
                        if 'from shared.config import Config' not in content:
                            issues.append((file_path, "Instantiates Config() but doesn't import Config"))
                
                except Exception:
                    pass
    
    if issues:
        print("\nRemaining issues found:")
        for file_path, issue in issues:
            print(f"  - {file_path}: {issue}")
    else:
        print("\nNo remaining Config-related issues found!")

if __name__ == "__main__":
    print("Fixing specific Config class issues...")
    fix_specific_config_issues()
    print("\nChecking for remaining issues...")
    check_remaining_issues()