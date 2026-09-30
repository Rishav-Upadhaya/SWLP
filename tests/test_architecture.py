"""Architecture guard tests — the rules in AGENTS.md, enforced.

Two classes of defect this repo has actually shipped, both invisible to every
other test:

1. A config field that is parsed but never read. ``SWLP_KV_DISK_DIR`` reached
   ``AppConfig`` and stopped there, so the entire Phase 12 disk-spill tier was
   unreachable in production while its unit tests passed.
2. An import that crosses a layer boundary. The import table in AGENTS.md is
   what keeps ``core/`` simulatable without a model and ``runner/`` free of
   benchmark imports; nothing checked it.

Both are structural, so both are checked structurally — by parsing the tree,
not by running anything.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "swlp"


def _py_files(*parts: str) -> list[Path]:
    base = SRC.joinpath(*parts) if parts else SRC
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


# ── 1. every config field must have a consumer ───────────────────────────────

def _runtime_config_fields() -> list[str]:
    tree = ast.parse((SRC / "config.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RuntimeConfig":
            return [
                s.target.id
                for s in node.body
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
            ]
    raise AssertionError("RuntimeConfig not found in config.py")


def test_every_runtime_config_field_is_read_somewhere() -> None:
    """A field nobody reads is a promise the CLI and docs cannot keep.

    ``config.py`` itself does not count: parsing a value into the dataclass is
    not the same as acting on it.
    """
    consumers = [p for p in _py_files() if p.name != "config.py"]
    haystack = "\n".join(p.read_text() for p in consumers)

    unread = [f for f in _runtime_config_fields() if f not in haystack]
    assert not unread, (
        "RuntimeConfig fields parsed but never read outside config.py: "
        f"{unread}. Either wire each one to its consumer or delete it — a "
        "config knob that changes nothing is worse than no knob."
    )


def test_every_config_field_is_overridable_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every field's SWLP_* env var must reach AppConfig — a dropped overlay
    silently ignores the user."""
    from dataclasses import fields

    from swlp.config import AppConfig, env_name, load_config

    samples = {"bool": None, "int": "7", "float": "0.25", "str": "x", "str | None": "x"}
    defaults = AppConfig()
    expected: dict[tuple[str, str], object] = {}
    for section in fields(AppConfig):
        for f in fields(getattr(defaults, section.name)):
            default = getattr(getattr(defaults, section.name), f.name)
            raw = samples.get(f.type, str(tmp_path / f.name))
            if raw is None:  # bool: flip the default
                raw = "false" if default else "true"
            monkeypatch.setenv(env_name(f.name), raw)
            expected[(section.name, f.name)] = default

    loaded = load_config(tmp_path / "missing.toml")
    unchanged = [
        k for k, default in expected.items()
        if getattr(getattr(loaded, k[0]), k[1]) == default
    ]
    assert not unchanged, f"env var ignored for: {unchanged}"


# ── 2. the import layering in AGENTS.md ──────────────────────────────────────

# package -> packages it may NOT import from
FORBIDDEN: dict[str, set[str]] = {
    "config": {"core", "hardware", "model", "runner", "benchmark"},
    "metrics": {"core", "hardware", "model", "runner", "benchmark"},
    "logging": {"core", "hardware", "model", "runner", "benchmark"},
    "core": {"runner", "benchmark"},
    "hardware": {"core", "runner", "benchmark"},
    "model": {"core", "runner", "benchmark"},
    "runner": {"benchmark"},
}


def _own_package(path: Path) -> str:
    rel = path.relative_to(SRC)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def _imported_swlp_packages(path: Path) -> set[str]:
    """Every in-package module this file imports, at any nesting depth.

    Catches deferred imports inside functions too — the streaming hot path uses
    them heavily, and a layering break hidden in one is still a layering break.
    """
    found: set[str] = set()
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module:          # from ..core.x import y
                found.add(node.module.split(".")[0])
            elif node.level and not node.module:     # from .. import core
                found.update(a.name.split(".")[0] for a in node.names)
            elif node.module and node.module.startswith("swlp."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("swlp."):
                    found.add(a.name.split(".")[1])
    return found


@pytest.mark.parametrize("path", _py_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_import_layering(path: Path) -> None:
    """No module may import from a layer above it (see AGENTS.md import table)."""
    pkg = _own_package(path)
    forbidden = FORBIDDEN.get(pkg)
    if forbidden is None:          # cli*, chat, serve, codec: may import anything
        return
    violations = sorted(_imported_swlp_packages(path) & forbidden - {pkg})
    assert not violations, (
        f"{path.relative_to(SRC)} is in '{pkg}/' and may not import from "
        f"{violations} — see the import table in AGENTS.md."
    )


def test_foundation_modules_import_only_stdlib() -> None:
    """config/metrics/logging are imported by everything; any in-package
    dependency here becomes a cycle."""
    for name in ("config.py", "metrics.py", "logging.py"):
        assert not _imported_swlp_packages(SRC / name), (
            f"{name} must depend on stdlib only — it is the import root."
        )


# ── 3. requirements.txt must not contradict pyproject.toml ───────────────────

def _pyproject_constraints() -> dict[str, str]:
    """{package: specifier} across [project].dependencies and every extra."""
    import tomllib

    data = tomllib.loads((SRC.parents[1] / "pyproject.toml").read_text())
    project = data["project"]
    specs: list[str] = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        specs.extend(extra)

    out: dict[str, str] = {}
    for spec in specs:
        for op in (">=", "==", "~=", ">", "<"):
            if op in spec:
                name, _, ver = spec.partition(op)
                out[name.strip().lower().replace("_", "-")] = op + ver.strip()
                break
    return out


def test_requirements_pins_satisfy_pyproject() -> None:
    """The lockfile records the measured environment; pyproject declares what
    is supported. A pin below pyproject's floor means the published numbers
    came from a version the package claims not to support."""
    req = SRC.parents[1] / "requirements.txt"
    constraints = _pyproject_constraints()

    for line in req.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        name, _, pinned = line.partition("==")
        key = name.strip().lower().replace("_", "-")
        constraint = constraints.get(key)
        assert constraint is not None, (
            f"requirements.txt pins '{name}' but pyproject.toml never declares it — "
            "one of the two files is stale."
        )
        if constraint.startswith(">="):
            floor = constraint[2:]
            got = tuple(int(x) for x in pinned.split(".") if x.isdigit())
            want = tuple(int(x) for x in floor.split(".") if x.isdigit())
            assert got >= want, (
                f"requirements.txt pins {name}=={pinned}, below pyproject's {constraint}"
            )
