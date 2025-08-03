#!/usr/bin/env python3
"""
Script to fix type annotation issues in the KnowledgeHub API.

This script:
1. Fixes Optional[Config] type hints to properly accept Config instances
2. Ensures proper imports for type annotations
3. Fixes common type annotation errors
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Set

def fix_config_type_annotations(file_path: Path) -> bool:
    """Fix Config-related type annotations in a file."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes_made = False
    
    # Pattern 1: Fix Optional[Config] in function signatures where Config() is instantiated
    # Change from: def __init__(self, config: Optional[Config] = None):
    #                  self.config = config or Config()
    # To use proper instance default or keep Optional but fix usage
    
    # First, let's fix the imports to ensure we have proper typing imports
    if 'from typing import' in content and 'Optional' not in content:
        content = re.sub(
            r'from typing import ([^\\n]+)',
            lambda m: f"from typing import {m.group(1)}, Optional" if "Optional" not in m.group(1) else m.group(0),
            content
        )
        changes_made = True
    
    # Pattern to find methods that use Optional[Config] with Config() instantiation
    pattern = r'def\s+\w+\s*\([^)]*config:\s*Optional\[Config\]\s*=\s*None[^)]*\):'
    
    # Keep the Optional[Config] pattern but ensure proper usage
    # The pattern is actually correct - Optional[Config] = None is the right way
    # The issue is in how Config is imported in some files
    
    # Fix imports that incorrectly alias settings as Config
    if 'from ..config import settings as Config' in content:
        content = content.replace('from ..config import settings as Config', 'from ..config import settings')
        # Now we need to fix usages of Config() to use settings or a proper Config import
        content = re.sub(r'config or Config\(\)', 'config or settings', content)
        content = re.sub(r'self\.config = Config\(\)', 'self.config = settings', content)
        changes_made = True
    elif 'from shared.config import Config' in content:
        # This import is correct, but we need to ensure Config is instantiated properly
        # The pattern config or Config() is correct here
        pass
    
    # Fix other common type annotation issues
    
    # 1. Fix missing imports for datetime
    if 'datetime' in content and 'from datetime import datetime' not in content and 'import datetime' not in content:
        if 'from datetime import' in content:
            content = re.sub(
                r'from datetime import ([^\\n]+)',
                lambda m: f"from datetime import {m.group(1)}, datetime" if "datetime" not in m.group(1) else m.group(0),
                content
            )
        else:
            # Add the import after other imports
            import_section = re.search(r'((?:from|import)[^\\n]+\\n)+', content)
            if import_section:
                content = content.replace(
                    import_section.group(0),
                    import_section.group(0) + "from datetime import datetime\n"
                )
        changes_made = True
    
    # 2. Fix List, Dict, Optional imports
    typing_imports_needed = []
    if re.search(r'\bList\[', content) and 'List' not in content.split('from typing import')[1] if 'from typing import' in content else True:
        typing_imports_needed.append('List')
    if re.search(r'\bDict\[', content) and 'Dict' not in content.split('from typing import')[1] if 'from typing import' in content else True:
        typing_imports_needed.append('Dict')
    if re.search(r'\bOptional\[', content) and 'Optional' not in content.split('from typing import')[1] if 'from typing import' in content else True:
        typing_imports_needed.append('Optional')
    if re.search(r'\bAny\b', content) and 'Any' not in content.split('from typing import')[1] if 'from typing import' in content else True:
        typing_imports_needed.append('Any')
    if re.search(r'\bUnion\[', content) and 'Union' not in content.split('from typing import')[1] if 'from typing import' in content else True:
        typing_imports_needed.append('Union')
    if re.search(r'\bTuple\[', content) and 'Tuple' not in content.split('from typing import')[1] if 'from typing import' in content else True:
        typing_imports_needed.append('Tuple')
    
    if typing_imports_needed:
        if 'from typing import' in content:
            # Extract current imports
            match = re.search(r'from typing import ([^\\n]+)', content)
            if match:
                current_imports = [imp.strip() for imp in match.group(1).split(',')]
                # Add missing imports
                for imp in typing_imports_needed:
                    if imp not in current_imports:
                        current_imports.append(imp)
                # Sort and rejoin
                current_imports.sort()
                new_import_line = f"from typing import {', '.join(current_imports)}"
                content = re.sub(r'from typing import [^\\n]+', new_import_line, content)
                changes_made = True
        else:
            # Add new typing import
            typing_imports_needed.sort()
            new_import = f"from typing import {', '.join(typing_imports_needed)}\n"
            # Add after other imports
            import_section = re.search(r'((?:from|import)[^\\n]+\\n)+', content)
            if import_section:
                content = content.replace(
                    import_section.group(0),
                    import_section.group(0) + new_import
                )
                changes_made = True
    
    # 3. Fix forward references in type annotations (add quotes)
    # This is for cases like -> User where User is defined later
    forward_ref_pattern = r'(\s*->\s*)([A-Z][a-zA-Z0-9_]+)(?:\s*[,\):])'
    
    def check_forward_ref(match):
        type_name = match.group(2)
        # Check if this type is imported or defined before use
        if f'class {type_name}' in content:
            # Find if class is defined after the current position
            class_pos = content.find(f'class {type_name}')
            match_pos = match.start()
            if class_pos > match_pos:
                # It's a forward reference, add quotes
                return f'{match.group(1)}"{type_name}"{match.group(0)[len(match.group(1)) + len(type_name):]}'
        return match.group(0)
    
    # Don't apply forward reference fixes to built-in types
    builtin_types = {'bool', 'int', 'float', 'str', 'bytes', 'None', 'Any', 'Dict', 'List', 'Optional', 'Union', 'Tuple', 'Set'}
    
    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False

def fix_circular_imports(file_path: Path) -> bool:
    """Fix circular import issues."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes_made = False
    
    # Common circular import patterns to fix
    
    # 1. Move imports inside TYPE_CHECKING block for type hints only
    if 'TYPE_CHECKING' not in content and re.search(r'from \.[a-z_]+ import [A-Z][a-zA-Z]+', content):
        # Check if these imports are only used in type annotations
        type_only_imports = []
        
        # Find all relative imports of classes (likely models)
        import_matches = re.findall(r'from (\.[a-z_]+) import ([A-Z][a-zA-Z, ]+)', content)
        
        for module, classes in import_matches:
            class_list = [c.strip() for c in classes.split(',')]
            for class_name in class_list:
                # Check if class is only used in type annotations (: ClassName or -> ClassName)
                usage_pattern = rf'\b{class_name}\b'
                type_pattern = rf'[:,]\s*(?:Optional\[)?{class_name}|>\s*(?:Optional\[)?{class_name}'
                
                all_usages = len(re.findall(usage_pattern, content))
                type_usages = len(re.findall(type_pattern, content))
                
                # Subtract the import line itself
                all_usages -= 1
                
                if all_usages > 0 and all_usages == type_usages:
                    type_only_imports.append((module, class_name))
        
        if type_only_imports:
            # Add TYPE_CHECKING import
            if 'from typing import' in content:
                content = re.sub(
                    r'from typing import ([^\\n]+)',
                    lambda m: f"from typing import {m.group(1)}, TYPE_CHECKING" if "TYPE_CHECKING" not in m.group(1) else m.group(0),
                    content
                )
            else:
                # Add new typing import
                import_section = re.search(r'((?:from|import)[^\\n]+\\n)+', content)
                if import_section:
                    content = content.replace(
                        import_section.group(0),
                        import_section.group(0) + "from typing import TYPE_CHECKING\n"
                    )
            
            # Move type-only imports to TYPE_CHECKING block
            type_checking_block = "\n\nif TYPE_CHECKING:\n"
            for module, class_name in type_only_imports:
                type_checking_block += f"    from {module} import {class_name}\n"
                # Remove from regular imports
                content = re.sub(rf'from {re.escape(module)} import [^\\n]*{class_name}[^\\n]*\\n', '', content)
            
            # Add TYPE_CHECKING block after imports
            import_section = re.search(r'((?:from|import)[^\\n]+\\n)+', content)
            if import_section:
                content = content.replace(
                    import_section.group(0),
                    import_section.group(0) + type_checking_block
                )
            
            changes_made = True
    
    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False

def main():
    """Main function to fix type annotations across the codebase."""
    
    api_dir = Path('/opt/projects/knowledgehub/api')
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(api_dir):
        # Skip __pycache__ directories
        if '__pycache__' in root:
            continue
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    print(f"Found {len(python_files)} Python files to check")
    
    files_fixed = 0
    issues_found = []
    
    for file_path in python_files:
        try:
            # Fix Config type annotations
            if fix_config_type_annotations(file_path):
                files_fixed += 1
                print(f"Fixed type annotations in: {file_path.relative_to(api_dir)}")
            
            # Fix circular imports
            if fix_circular_imports(file_path):
                files_fixed += 1
                print(f"Fixed circular imports in: {file_path.relative_to(api_dir)}")
            
        except Exception as e:
            issues_found.append((file_path, str(e)))
            print(f"Error processing {file_path}: {e}")
    
    print(f"\nSummary:")
    print(f"- Files processed: {len(python_files)}")
    print(f"- Files fixed: {files_fixed}")
    print(f"- Issues found: {len(issues_found)}")
    
    if issues_found:
        print("\nFiles with issues:")
        for file_path, error in issues_found:
            print(f"  - {file_path.relative_to(api_dir)}: {error}")

if __name__ == "__main__":
    main()