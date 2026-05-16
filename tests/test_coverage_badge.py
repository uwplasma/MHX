from __future__ import annotations

import json

from tools.write_coverage_badge import badge_payload, coverage_color, write_coverage_badge


def test_coverage_badge_payload_colors() -> None:
    assert coverage_color(95.0) == "brightgreen"
    assert coverage_color(90.0) == "green"
    assert coverage_color(80.0) == "yellowgreen"
    assert coverage_color(70.0) == "yellow"
    assert coverage_color(60.0) == "orange"
    assert coverage_color(59.9) == "red"
    assert badge_payload(95.37)["message"] == "95.4%"


def test_write_coverage_badge(tmp_path) -> None:
    coverage_json = tmp_path / "coverage.json"
    coverage_json.write_text(
        json.dumps({"totals": {"percent_covered": 93.25}}),
        encoding="utf-8",
    )
    output = write_coverage_badge(coverage_json, tmp_path / "badges" / "coverage.json")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == {
        "schemaVersion": 1,
        "label": "coverage",
        "message": "93.2%",
        "color": "green",
    }
