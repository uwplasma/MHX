"""Write a Shields.io endpoint JSON file from coverage.py JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def coverage_color(percent: float) -> str:
    """Return a conservative Shields color for a coverage percentage."""
    if percent >= 95.0:
        return "brightgreen"
    if percent >= 90.0:
        return "green"
    if percent >= 80.0:
        return "yellowgreen"
    if percent >= 70.0:
        return "yellow"
    if percent >= 60.0:
        return "orange"
    return "red"


def badge_payload(percent: float) -> dict[str, str | int]:
    """Return a Shields endpoint payload."""
    return {
        "schemaVersion": 1,
        "label": "coverage",
        "message": f"{percent:.1f}%",
        "color": coverage_color(percent),
    }


def write_coverage_badge(coverage_json: str | Path, output: str | Path) -> Path:
    """Convert coverage.py JSON into a Shields endpoint JSON file."""
    coverage_path = Path(coverage_json)
    output_path = Path(output)
    data = json.loads(coverage_path.read_text(encoding="utf-8"))
    percent = float(data["totals"]["percent_covered"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(badge_payload(percent), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage_json", help="coverage.py JSON file")
    parser.add_argument("output", help="Shields endpoint JSON output path")
    args = parser.parse_args()
    write_coverage_badge(args.coverage_json, args.output)


if __name__ == "__main__":
    main()
