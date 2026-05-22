from __future__ import annotations

import ast
import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
PUBLICATION_EXAMPLE_GLOB = "publication_*.py"
OUTDIR_ROOT_ENV = "MHX_EXAMPLE_OUTDIR_ROOT"
SMOKE_TIMEOUT_SECONDS = 120
FAST_REPRESENTATIVE_PATTERNS = (
    "publication_*resistive*decay*.py",
    "publication_*exact*decay*.py",
    "publication_*fast*.py",
    "publication_*smoke*.py",
    "publication_*linear*tearing*eigenvalue*.py",
    "publication_*fkr*window*.py",
)
EXPECTED_OUTPUT_SUFFIXES_BY_SCRIPT = {
    "publication_resistive_decay.py": (
        "manifest.json",
        "figures/decay_amplitude.png",
        "figures/decay_energy.png",
        "figures/decay_relative_error.png",
    ),
    "publication_exact_decay.py": (
        "manifest.json",
        "figures/decay_amplitude.png",
        "figures/decay_energy.png",
        "figures/decay_relative_error.png",
    ),
}
OUTPUT_PARAMETER_HINTS = (
    "OUTDIR",
    "OUTPUT",
    "ARTIFACT",
    "FIGURE",
    "PLOT",
    "RUN_DIR",
    "ROOT",
)
USER_PARAMETER_HINTS = (
    *OUTPUT_PARAMETER_HINTS,
    "CASE",
    "DT",
    "ETA",
    "GRID",
    "MODE",
    "NX",
    "NY",
    "NU",
    "SAVE",
    "SEED",
    "SHAPE",
    "STEPS",
    "T_END",
)


def _publication_scripts() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob(PUBLICATION_EXAMPLE_GLOB))


PUBLICATION_SCRIPTS = _publication_scripts()


def _script_id(script_path: Path) -> str:
    return script_path.relative_to(ROOT).as_posix()


def _parse_script(script_path: Path) -> ast.Module:
    return ast.parse(script_path.read_text(encoding="utf-8"), filename=_script_id(script_path))


def _is_docstring_statement(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _is_import_statement(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.Import | ast.ImportFrom)


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Tuple | ast.List):
        return {name for nested_target in target.elts for name in _target_names(nested_target)}
    return set()


def _assignment_names(statement: ast.stmt) -> set[str]:
    if isinstance(statement, ast.Assign):
        return {name for target in statement.targets for name in _target_names(target)}
    if isinstance(statement, ast.AnnAssign):
        return _target_names(statement.target)
    return set()


def _assignment_value(statement: ast.stmt) -> ast.expr | None:
    if isinstance(statement, ast.Assign | ast.AnnAssign):
        return statement.value
    return None


def _is_public_parameter_name(name: str) -> bool:
    return name.isupper() and not name.startswith("_")


def _leading_parameter_names(tree: ast.Module) -> list[str]:
    parameter_names: list[str] = []
    for statement in tree.body:
        if _is_docstring_statement(statement) or _is_import_statement(statement):
            continue

        statement_parameter_names = sorted(
            name for name in _assignment_names(statement) if _is_public_parameter_name(name)
        )
        if not statement_parameter_names:
            break
        parameter_names.extend(statement_parameter_names)

    return parameter_names


def _expr_references_name(expression: ast.expr | None, names: set[str]) -> bool:
    if expression is None:
        return False
    return any(isinstance(node, ast.Name) and node.id in names for node in ast.walk(expression))


def _expr_references_outdir_env(expression: ast.expr | None) -> bool:
    if expression is None:
        return False
    return any(
        isinstance(node, ast.Constant) and node.value == OUTDIR_ROOT_ENV
        for node in ast.walk(expression)
    )


def _env_backed_top_level_names(tree: ast.Module) -> set[str]:
    env_backed_names: set[str] = set()
    for statement in tree.body:
        assigned_names = {
            name for name in _assignment_names(statement) if _is_public_parameter_name(name)
        }
        if not assigned_names:
            continue

        value = _assignment_value(statement)
        if _expr_references_outdir_env(value) or _expr_references_name(value, env_backed_names):
            env_backed_names.update(assigned_names)

    return env_backed_names


def _is_output_parameter_name(name: str) -> bool:
    return any(hint in name for hint in OUTPUT_PARAMETER_HINTS)


def _is_dunder_main_name(expression: ast.expr) -> bool:
    return isinstance(expression, ast.Name) and expression.id == "__name__"


def _is_dunder_main_literal(expression: ast.expr) -> bool:
    return isinstance(expression, ast.Constant) and expression.value == "__main__"


def _is_dunder_main_guard(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.If) or not isinstance(statement.test, ast.Compare):
        return False
    comparison = statement.test
    operands = [comparison.left, *comparison.comparators]
    return any(
        (_is_dunder_main_name(left_operand) and _is_dunder_main_literal(right_operand))
        or (_is_dunder_main_literal(left_operand) and _is_dunder_main_name(right_operand))
        for left_operand, right_operand in zip(operands, operands[1:], strict=False)
    )


def _declares_ci_smoke(script_path: Path) -> bool:
    tree = _parse_script(script_path)
    for statement in tree.body:
        if "CI_SMOKE" not in _assignment_names(statement):
            continue
        value = _assignment_value(statement)
        return isinstance(value, ast.Constant) and value.value is True
    return False


def _select_fast_representative_script(scripts: list[Path]) -> Path:
    smoke_scripts = [script_path for script_path in scripts if _declares_ci_smoke(script_path)]
    if smoke_scripts:
        return smoke_scripts[0]

    for pattern in FAST_REPRESENTATIVE_PATTERNS:
        matches = [
            script_path for script_path in scripts if fnmatch.fnmatch(script_path.name, pattern)
        ]
        if matches:
            return matches[0]

    pytest.fail(
        "Publication examples exist, but none is marked as the FAST CI "
        "representative. Add top-level CI_SMOKE = True to the cheapest script or "
        f"rename it to match one of {FAST_REPRESENTATIVE_PATTERNS}."
    )


def _relative_suffix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _has_path_with_suffix(paths: list[Path], root: Path, suffix: str) -> bool:
    return any(_relative_suffix(path, root).endswith(suffix) for path in paths)


@pytest.mark.parametrize("script_path", PUBLICATION_SCRIPTS, ids=_script_id)
def test_publication_examples_are_standalone_scripts(script_path: Path) -> None:
    tree = _parse_script(script_path)
    main_function_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "main"
    ]
    dunder_main_guard_lines = [
        node.lineno for node in ast.walk(tree) if _is_dunder_main_guard(node)
    ]

    assert main_function_lines == [], (
        f"{_script_id(script_path)} should execute as a standalone top-level "
        f"example, not via def main(); lines={main_function_lines}"
    )
    assert dunder_main_guard_lines == [], (
        f"{_script_id(script_path)} should not hide execution behind an "
        f"if __name__ == '__main__' guard; lines={dunder_main_guard_lines}"
    )


@pytest.mark.parametrize("script_path", PUBLICATION_SCRIPTS, ids=_script_id)
def test_publication_examples_use_top_level_user_parameters(script_path: Path) -> None:
    tree = _parse_script(script_path)
    leading_parameters = _leading_parameter_names(tree)
    argparse_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and any(alias.name == "argparse" for alias in node.names)
    ]

    assert leading_parameters, (
        f"{_script_id(script_path)} should declare user-editable ALL_CAPS "
        "parameters immediately after imports."
    )
    assert any(
        any(hint in parameter_name for hint in USER_PARAMETER_HINTS)
        for parameter_name in leading_parameters
    ), (
        f"{_script_id(script_path)} top-level parameters should include user-facing "
        f"run/output controls; found {leading_parameters}."
    )
    assert argparse_lines == [], (
        f"{_script_id(script_path)} should use top-level user parameters instead "
        f"of argparse; import lines={argparse_lines}"
    )


@pytest.mark.parametrize("script_path", PUBLICATION_SCRIPTS, ids=_script_id)
def test_publication_examples_honor_outdir_root_env(script_path: Path) -> None:
    tree = _parse_script(script_path)
    env_backed_names = _env_backed_top_level_names(tree)
    output_env_names = sorted(name for name in env_backed_names if _is_output_parameter_name(name))

    assert output_env_names, (
        f"{_script_id(script_path)} should route output parameters through "
        f"{OUTDIR_ROOT_ENV}; env-backed names={sorted(env_backed_names)}."
    )


def test_fast_publication_example_executes_under_tmp_outdir(tmp_path: Path) -> None:
    if not PUBLICATION_SCRIPTS:
        pytest.skip(f"no examples/{PUBLICATION_EXAMPLE_GLOB} scripts in this checkout")

    script_path = _select_fast_representative_script(PUBLICATION_SCRIPTS)
    env = {
        **os.environ,
        OUTDIR_ROOT_ENV: str(tmp_path),
        "JAX_ENABLE_X64": "1",
        "JAX_PLATFORM_NAME": "cpu",
        "MPLBACKEND": "Agg",
    }
    result = subprocess.run(
        [sys.executable, _script_id(script_path)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=SMOKE_TIMEOUT_SECONDS,
        check=False,
    )

    assert result.returncode == 0, (
        f"{_script_id(script_path)} failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    manifest_paths = sorted(tmp_path.rglob("manifest.json"))
    plot_paths = sorted(
        path for path in tmp_path.rglob("*") if path.suffix.lower() in {".png", ".gif"}
    )
    produced_paths = sorted(path for path in tmp_path.rglob("*") if path.is_file())

    assert manifest_paths, f"{_script_id(script_path)} did not write a manifest under tmp_path"
    assert plot_paths, f"{_script_id(script_path)} did not write any plot files under tmp_path"

    for suffix in EXPECTED_OUTPUT_SUFFIXES_BY_SCRIPT.get(script_path.name, ()):
        assert _has_path_with_suffix(produced_paths, tmp_path, suffix), (
            f"{_script_id(script_path)} did not write expected output suffix {suffix}; "
            f"produced={[_relative_suffix(path, tmp_path) for path in produced_paths]}"
        )

    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative_output in manifest.get("outputs", {}).values():
            if isinstance(relative_output, str):
                assert (manifest_path.parent / relative_output).exists(), (
                    f"{_relative_suffix(manifest_path, tmp_path)} references missing "
                    f"output {relative_output}"
                )
