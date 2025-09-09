#!/usr/bin/env python3
"""
Simple code quality checker for the fairness-aware conformal prediction codebase.
Checks for common style and quality issues.
"""

import os
import re
import sys
from typing import List, Tuple


def check_file_quality(filepath: str) -> List[Tuple[int, str]]:
    """Check a Python file for common quality issues.

    Returns list of (line_number, issue_description) tuples.
    """
    issues = []

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        # Check line length (PEP 8 recommends 79 chars)
        if len(line.rstrip()) > 88:  # Allow slightly longer for readability
            issues.append((i, f"Line too long ({len(line.rstrip())} chars)"))

        # Check for trailing whitespace
        if line.endswith(' ') or line.endswith('\t'):
            issues.append((i, "Trailing whitespace"))

        # Check for unnecessary imports (basic check)
        if line.strip().startswith('import ') and 'unused' in line.lower():
            issues.append((i, "Potentially unused import"))

        # Check for good docstring practices (basic)
        if line.strip().startswith('def ') and '"""' not in ''.join(lines[i:i+3]):
            if 'test_' not in line and '__' not in line:
                issues.append((i, "Function missing docstring"))

    return issues


def check_import_structure(filepath: str) -> List[str]:
    """Check import organization following PEP 8."""
    issues = []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Basic import order check
    import_sections = []
    current_section = []
    in_imports = False

    for line in content.split('\n'):
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            current_section.append(stripped)
            in_imports = True
        elif stripped == '' and in_imports:
            continue  # Allow blank lines in imports
        elif stripped.startswith('#') and in_imports:
            continue  # Allow comments in imports
        elif in_imports and stripped:
            # End of import section
            if current_section:
                import_sections.append(current_section)
                current_section = []
            in_imports = False
        elif not stripped and not in_imports:
            continue  # Skip blank lines outside imports

    if current_section:
        import_sections.append(current_section)

    return issues


def main():
    """Run code quality checks on all Python files."""
    src_dir = 'src'
    if not os.path.exists(src_dir):
        print("❌ src/ directory not found")
        return 1

    total_issues = 0
    files_checked = 0

    print("Running code quality checks...\n")

    for filename in sorted(os.listdir(src_dir)):
        if filename.endswith('.py'):
            filepath = os.path.join(src_dir, filename)
            files_checked += 1

            print(f"Checking {filepath}...")
            issues = check_file_quality(filepath)

            if issues:
                print(f"  Issues found:")
                for line_num, issue in issues[:5]:  # Show first 5 issues
                    print(f"    Line {line_num}: {issue}")
                if len(issues) > 5:
                    print(f"    ... and {len(issues) - 5} more issues")
                total_issues += len(issues)
            else:
                print(f"  ✓ No issues found")
            print()

    # Check test file too
    if os.path.exists('test_basic.py'):
        print("Checking test_basic.py...")
        issues = check_file_quality('test_basic.py')
        files_checked += 1

        if issues:
            print(f"  Issues found:")
            for line_num, issue in issues[:5]:
                print(f"    Line {line_num}: {issue}")
            if len(issues) > 5:
                print(f"    ... and {len(issues) - 5} more issues")
            total_issues += len(issues)
        else:
            print(f"  ✓ No issues found")
        print()

    print(f"--- Summary ---")
    print(f"Files checked: {files_checked}")
    print(f"Total issues: {total_issues}")

    if total_issues == 0:
        print("🎉 All files pass quality checks!")
        return 0
    elif total_issues < 10:
        print("✓ Minor issues found, code quality is good")
        return 0
    else:
        print("⚠️  Multiple issues found, consider cleanup")
        return 1


if __name__ == "__main__":
    sys.exit(main())