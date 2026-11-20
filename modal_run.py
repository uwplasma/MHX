"""Run any example script on a Modal GPU.

Authenticate with Modal once (``modal setup``), then run from the repository root:

    modal run modal_run.py --script examples/gallery/11_turbulence_3d.py

Every example keeps its input parameters as module-level constants, so a run is
resized by overriding them. Pass ``--overrides`` a space-separated list of
``NAME=VALUE`` pairs, where each value is a Python literal:

    modal run --detach modal_run.py \
        --script examples/sweet_parker/collapse_check.py \
        --overrides "NX=3072 NY=384 ETA=2e-3 T_END=300.0"

The script runs in the ``mhx-outputs`` volume, so its relative ``outputs/...``
folders persist there. ``--detach`` keeps a long run alive after you close the
terminal. When the run finishes, the launcher prints one ``modal volume get``
command per new file.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

app = modal.App("mhx-run-example")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]",
        "jaxtyping",
        "matplotlib",
        "numpy",
        "scipy",
        "rich",
        "solvax",
        "imageio",
        "typer",
        "optax",
    )
    .add_local_dir("src", remote_path="/root/mhx/src", copy=True)
    .add_local_dir("examples", remote_path="/root/mhx/examples", copy=True)
    .env(
        {
            "PYTHONPATH": "/root/mhx/src",
            "JAX_ENABLE_X64": "1",
            "JAX_PLATFORM_NAME": "cuda",
            "MPLBACKEND": "Agg",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
)

volume = modal.Volume.from_name("mhx-outputs", create_if_missing=True)


def parse_overrides(text: str) -> dict[str, object]:
    """Turn ``"NX=64 ETA=2e-3 NAME=abc"`` into ``{"NX": 64, "ETA": 0.002, "NAME": "abc"}``."""
    overrides: dict[str, object] = {}
    for item in text.split():
        name, separator, raw = item.partition("=")
        if not separator or not name.isidentifier():
            raise ValueError(f"override {item!r} must look like NAME=VALUE")
        try:
            overrides[name] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            overrides[name] = raw
    return overrides


def patched_source(source: str, overrides: dict[str, object]) -> str:
    """Replace module-level ``NAME = value`` assignments in a script's source."""
    tree = ast.parse(source)
    found: set[str] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in overrides
        ):
            node.value = ast.parse(repr(overrides[node.targets[0].id]), mode="eval").body
            found.add(node.targets[0].id)
    missing = sorted(set(overrides) - found)
    if missing:
        raise KeyError(f"the script has no module-level assignment for: {', '.join(missing)}")
    return ast.unparse(ast.fix_missing_locations(tree))


@app.function(image=image, gpu="H100", volumes={"/outputs": volume}, timeout=6 * 3600)
def run_script(script: str, overrides: dict[str, object]) -> list[str]:
    """Run ``script`` with its constants overridden and return the files it wrote."""
    root = Path("/root/mhx")
    source = patched_source((root / script).read_text(encoding="utf-8"), overrides)
    workdir = Path("/outputs")
    runner = workdir / f"run_{Path(script).stem}.py"
    runner.write_text(source, encoding="utf-8")

    before = {path for path in workdir.rglob("*") if path.is_file()}
    command = [sys.executable, "-u", str(runner)]  # -u streams progress to the Modal logs
    print(f"[modal] running {script} with overrides {overrides}", flush=True)
    start = time.perf_counter()
    result = subprocess.run(command, cwd=workdir, env=os.environ.copy())
    volume.commit()
    print(f"[modal] finished in {time.perf_counter() - start:.1f} s", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed with exit code {result.returncode}")
    after = {path for path in workdir.rglob("*") if path.is_file()}
    return sorted(str(path.relative_to(workdir)) for path in after - before if path != runner)


@app.local_entrypoint()
def run(script: str, overrides: str = ""):
    """Launch ``script`` on Modal and print the commands that download its outputs."""
    files = run_script.remote(script, parse_overrides(overrides))
    print(f"\nDone. {len(files)} new files in the mhx-outputs volume.")
    for name in files:
        target = Path(name)
        print(f"  mkdir -p {target.parent} && modal volume get mhx-outputs {name} {name}")
