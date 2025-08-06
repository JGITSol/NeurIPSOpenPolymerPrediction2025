#!/usr/bin/env python3
"""
Script to fix JSON formatting issues in the T4x2 notebook
"""

import json
import re

def fix_notebook_json():
    """Fix JSON formatting issues in the notebook."""
    
    # Read the notebook file
    with open('neurips-t4x2-complete-solution-fixed.ipynb', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Common JSON fixes
    fixes = [
        # Fix missing quotes at end of strings
        (r'([^"\\])"(\s*\n\s*["}])', r'\1"\2'),
        # Fix missing commas before closing quotes
        (r'([^",])\n(\s*")', r'\1,\n\2'),
        # Fix missing quotes at start of strings
        (r'(\n\s*)([^"{\s][^"]*")', r'\1"\2'),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
    # Try to parse and fix specific issues
    try:
        json.loads(content)
        print("✅ JSON is now valid!")
    except json.JSONDecodeError as e:
        print(f"❌ JSON error at line {e.lineno}, column {e.colno}: {e.msg}")
        print(f"   Context: {content[max(0, e.pos-50):e.pos+50]}")
        
        # Try to fix common issues around the error position
        lines = content.split('\n')
        error_line_idx = e.lineno - 1
        
        if error_line_idx < len(lines):
            error_line = lines[error_line_idx]
            print(f"   Error line: {error_line}")
            
            # Common fixes
            if 'Expecting \',\' delimiter' in e.msg:
                # Add missing comma
                if not error_line.rstrip().endswith(',') and not error_line.rstrip().endswith('{'):
                    lines[error_line_idx] = error_line.rstrip() + ','
                    content = '\n'.join(lines)
                    print("   Fixed: Added missing comma")
            
            elif 'Unterminated string' in e.msg or 'Expecting' in e.msg:
                # Fix quote issues
                if error_line.count('"') % 2 == 1:  # Odd number of quotes
                    lines[error_line_idx] = error_line + '"'
                    content = '\n'.join(lines)
                    print("   Fixed: Added missing quote")
    
    # Write the fixed content
    with open('neurips-t4x2-complete-solution-fixed.ipynb', 'w', encoding='utf-8') as f:
        f.write(content)
    
    return content

if __name__ == "__main__":
    print("🔧 Fixing T4x2 notebook JSON formatting...")
    fix_notebook_json()
    print("✅ Notebook fixing completed!")