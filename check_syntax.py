#!/usr/bin/env python3
"""Quick syntax check for all Python files"""

import ast
import sys
from pathlib import Path


def check_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ast.parse(f.read(), filename=str(file_path))
        return True, None
    except SyntaxError as e:
        return False, f"{e.filename}:{e.lineno}:{e.offset}: {e.msg}"
    except Exception as e:
        return False, f"{file_path}: {str(e)}"


def main():
    """Check syntax of all Python files in src/"""
    src_path = Path("src")
    if not src_path.exists():
        print("src/ directory not found")
        return 1
    
    errors = []
    checked = 0
    
    for py_file in src_path.rglob("*.py"):
        checked += 1
        valid, error = check_syntax(py_file)
        if not valid:
            errors.append(error)
    
    if errors:
        print(f"Found {len(errors)} syntax errors in {checked} files:")
        for error in errors:
            print(f"  {error}")
        return 1
    else:
        print(f"✓ All {checked} Python files have valid syntax")
        return 0


if __name__ == "__main__":
    sys.exit(main())