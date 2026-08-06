"""Check the first-read MHX documents for direct technical prose."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "docs" / "index.md",
    ROOT / "docs" / "getting_started" / "install.md",
    ROOT / "docs" / "getting_started" / "first_run.md",
    ROOT / "docs" / "getting_started" / "first_movie.md",
    ROOT / "docs" / "getting_started" / "troubleshooting.md",
    ROOT / "docs" / "how_to" / "run_on_gpus.md",
    ROOT / "docs" / "physics" / "reduced_mhd.md",
    ROOT / "docs" / "physics" / "mhd3d.md",
    ROOT / "docs" / "physics" / "spectral_method.md",
    ROOT / "docs" / "physics" / "time_integration.md",
    ROOT / "docs" / "physics" / "differentiability.md",
    ROOT / "docs" / "physics" / "solvax_boundary.md",
    ROOT / "docs" / "validation" / "index.md",
    ROOT / "docs" / "gallery.md",
    ROOT / "docs" / "reference" / "api" / "index.md",
    ROOT / "docs" / "reference" / "config_schema.md",
    ROOT / "docs" / "reference" / "performance.md",
    ROOT / "docs" / "develop" / "architecture.md",
    ROOT / "docs" / "develop" / "style.md",
    ROOT / "examples" / "README.md",
    ROOT / "examples" / "gallery" / "README.md",
)

BANNED_TERMS = (
    "beacon",
    "blazingly",
    "cutting-edge",
    "delve",
    "effortless",
    "elevate",
    "embark",
    "empower",
    "ever-evolving",
    "facilitate",
    "foster",
    "game changer",
    "harness",
    "in conclusion",
    "in order to",
    "intricate",
    "it is worth noting",
    "landscape",
    "leverage",
    "meticulous",
    "moreover",
    "multifaceted",
    "paradigm shift",
    "paramount",
    "pivotal",
    "powerful",
    "realm",
    "robust",
    "seamless",
    "state-of-the-art",
    "streamline",
    "supercharge",
    "tapestry",
    "transformative",
    "unlock",
    "utilize",
    "vital role",
)


def prose_without_code(text: str) -> str:
    """Remove code, math, inline code, links, and tables before checks."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$\n]+\$", "", text)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"^\[!\[.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    lines = [
        line
        for line in text.splitlines()
        if not line.lstrip().startswith("|")
        and not line.lstrip().startswith("#")
        and not line.lstrip().startswith(":")
    ]
    return "\n".join(lines)


def check_document(path: Path) -> list[str]:
    """Return actionable style errors for one Markdown file."""
    prose = prose_without_code(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    lowered = prose.lower()
    for term in BANNED_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            errors.append(f"replace filler term {term!r}")
    if ";" in prose:
        errors.append("replace semicolons with full stops or separate steps")
    if "—" in prose:
        errors.append("replace em dashes with commas or full stops")

    # Find written contractions through their apostrophe forms.
    apostrophe_forms = (
        "aren't",
        "can't",
        "couldn't",
        "didn't",
        "doesn't",
        "don't",
        "hasn't",
        "haven't",
        "isn't",
        "it's",
        "shouldn't",
        "that's",
        "they're",
        "wasn't",
        "we're",
        "won't",
        "wouldn't",
    )
    for contraction in apostrophe_forms:
        if re.search(rf"\b{re.escape(contraction)}\b", lowered):
            errors.append(f"expand contraction {contraction!r}")

    sentences = re.split(r"(?<=[.!?])\s+|\n\s*\n", prose.strip())
    for sentence in sentences:
        sentence = re.sub(r"\s+", " ", sentence)
        words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", sentence)
        if len(words) > 25:
            preview = " ".join(words[:8])
            errors.append(f"shorten {len(words)}-word sentence starting {preview!r}")
    return errors


def main() -> int:
    """Print all prose errors and return a shell status."""
    failures = []
    for document in DOCUMENTS:
        for error in check_document(document):
            failures.append(f"{document.relative_to(ROOT)}: {error}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Prose check passed for {len(DOCUMENTS)} documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
