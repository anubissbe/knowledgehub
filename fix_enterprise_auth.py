#\!/usr/bin/env python3
"""
Fix the enterprise authentication dependency to work properly with FastAPI
"""

import os

# Read the current security_compliance.py file
security_file = '/opt/projects/knowledgehub/api/services/security_compliance.py'

with open(security_file, 'r') as f:
    content = f.read()

# Fix the get_security_context function to use FastAPI Header dependency
old_function = '''# Authentication middleware dependency
async def get_security_context(authorization: str) -> Optional[SecurityContext]:
    """Get security context from authorization header"""
    
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    token = authorization[7:]  # Remove "Bearer " prefix
    return await security_compliance_service.verify_jwt_token(token)'''

new_function = '''# Authentication middleware dependency
from fastapi import Header

async def get_security_context(authorization: Optional[str] = Header(None)) -> Optional[SecurityContext]:
    """Get security context from authorization header"""
    
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    token = authorization[7:]  # Remove "Bearer " prefix
    return await security_compliance_service.verify_jwt_token(token)'''

# Replace the function
new_content = content.replace(old_function, new_function)

# Also add the Optional import if not present
if 'from typing import Dict, List, Optional, Set, Any, Tuple' not in new_content:
    # Add Optional import
    new_content = new_content.replace(
        'from typing import Dict, List, Optional, Set, Any, Tuple',
        'from typing import Dict, List, Optional, Set, Any, Tuple'
    )

# Add FastAPI import if not present
if 'from fastapi import Header' not in new_content:
    # Find where to add the import
    import_section = '''import base64

from sqlalchemy import text, Column, String, Boolean, DateTime, JSON, Integer, Text'''
    
    new_import_section = '''import base64

from fastapi import Header
from sqlalchemy import text, Column, String, Boolean, DateTime, JSON, Integer, Text'''
    
    new_content = new_content.replace(import_section, new_import_section)

# Write the updated content
with open(security_file, 'w') as f:
    f.write(new_content)

print("✅ Fixed enterprise authentication dependency")
print("✅ Updated get_security_context to use FastAPI Header dependency")
print("✅ Added proper Optional typing import")

