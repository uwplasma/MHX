from __future__ import annotations

from pathlib import Path
import re
import sys


PATTERNS = [
    re.compile(r"^\s*import\s+mhd_", re.MULTILINE),
    re.compile(r"^\s*from\s+mhd_", re.MULTILINE),
    re.compile(r"^\s*import\s+run_MHD", re.MULTILINE),
    re.compile(r"^\s*from\s+run_MHD", re.MULTILINE),
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = []
    for folder in ["mhx", "tests", "examples"]:
        paths.extend((root / folder).rglob("*.py"))

    violations = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for pattern in PATTERNS:
            if pattern.search(text):
                violations.append(str(path))
                break

    if violations:
        print("Legacy script imports detected in:")
        for path in sorted(set(violations)):
            print(f"  - {path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
