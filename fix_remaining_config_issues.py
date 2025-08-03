#!/usr/bin/env python3
"""
Script to fix the remaining Config-related issues.
"""

import os
import re

def fix_remaining_files():
    """Fix the remaining files that have Config() instantiation without proper import."""
    
    files_to_fix = [
        '/opt/projects/knowledgehub/api/cors_config.py',
        '/opt/projects/knowledgehub/api/memory_system/distributed_sharding.py',
        '/opt/projects/knowledgehub/api/security/headers.py',
        '/opt/projects/knowledgehub/api/services/circuit_breaker.py',
        '/opt/projects/knowledgehub/api/services/database_recovery.py',
        '/opt/projects/knowledgehub/api/routes/cors_security.py',
    ]
    
    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix the typo in cors_config.py first
            if 'cors_config.py' in file_path:
                content = content.replace('from typing import List, Dict, A, Optionalny', 'from typing import List, Dict, Any, Optional')
            
            # Check what kind of Config usage this file has
            if 'Config()' in content:
                # These files create their own Config classes, not the shared one
                # Let's check if they define their own Config class
                if 'class CORSSecurityConfig' in content or 'class CSRFConfig' in content or 'class CircuitConfig' in content or 'class RetryConfig' in content or 'class ShardingConfig' in content:
                    # These files use their own Config classes, no need to import shared Config
                    pass
                else:
                    # Need to add shared Config import
                    # Add the import after other imports
                    import_section = re.search(r'((?:from|import)[^\n]+\n)+', content)
                    if import_section:
                        # Determine the right import path based on file location
                        if '/api/' in file_path:
                            depth = file_path.count('/') - file_path[:file_path.rfind('/api/')].count('/') - 2
                            if depth == 1:
                                import_stmt = "from shared.config import Config\n"
                            elif depth == 2:
                                import_stmt = "from ..shared.config import Config\n"
                            else:
                                import_stmt = "from shared.config import Config\n"
                            
                            content = content.replace(
                                import_section.group(0),
                                import_section.group(0) + import_stmt
                            )
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Fixed: {file_path}")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    print("Fixing remaining Config issues...")
    fix_remaining_files()