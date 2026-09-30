# Contributing to SWLP

Contributions of all sizes are welcome: bug reports, docs fixes, benchmark
numbers from other Apple Silicon machines, and code. SWLP supports Apple
Silicon (M-series Macs) only.

## Setup

```bash
git clone https://github.com/Rishav-Upadhaya/SWLP.git
cd SWLP
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,apple]"      # add ,codec to work on the .swz codec

pytest                             # full suite, no model download needed
ruff check src tests scripts       # zero errors
swlp --backend mock --prompt "hi"  # CLI smoke test
```

## Workflow

1. Open an issue first for anything non-trivial so the approach is agreed
   before code is written.
2. Branch off `main` (`git checkout -b fix/short-description`).
3. Make the change and add tests: every new function in `src/swlp/` needs a
   test in `tests/`. Use `MockRunner` for runner tests; never download a real
   model in a unit test.
4. Run `pytest` and `ruff check src tests scripts`; both must be clean. CI runs
   the same checks on macOS.
5. Open a pull request, fill in the template, and link the issue it closes.
   Performance changes should include hardware, model, and before/after numbers.

Write imperative commit messages (`Add expert-cache budget clamp`, not
`fixed stuff`) and reference issues where relevant (`Fixes #42`).

## Conventions

The full operating manual is
[`.claude/CLAUDE.md`](https://github.com/Rishav-Upadhaya/SWLP/blob/main/.claude/CLAUDE.md).
The essentials:

- **No quality compromise.** Anything lossy must be opt-in, off by default, and
  clearly labelled.
- **Layout and imports.** Runners go in `runner/` and register in
  `build_runner()`, algorithms in `core/`, disk I/O in `model/`, reporters in
  `reporting/`. `core/`, `hardware/` and `model/` must not import from
  `runner/`, `benchmark/` or `reporting/` (`tests/test_architecture.py`
  enforces this).
- **No hardcoded values.** Use `AppConfig`, TOML configs in `configs/`, and
  `SWLP_*` environment variables. A config field with no consumer is not allowed.
- **Logging, not `print()`,** for runtime output (`configure_logging()`);
  user-facing CLI/chat output is the exception. Hot-path failures go through
  `self.degrade(reason, exc)`.
- Full type annotations, `snake_case` / `PascalCase`, no new file over 300 lines,
  and remove dead code rather than leaving it dormant.

## Reporting bugs

Use the bug-report template and include the exact `swlp ...` command, your
chip, RAM, macOS and Python versions, the output of `swlp doctor`, and the full
error (or `--json` metrics if it ran but misbehaved). Report security issues
privately as described in [SECURITY.md](https://github.com/Rishav-Upadhaya/SWLP/blob/main/SECURITY.md).

## Releasing (maintainers)

1. Bump `__version__` in `src/swlp/__init__.py`.
2. Move the `[Unreleased]` notes in `CHANGELOG.md` under the new version and date.
3. Tag and push: `git tag vX.Y.Z && git push --tags`.
4. `.github/workflows/release.yml` builds the sdist and wheel, runs
   `twine check`, and publishes to PyPI via Trusted Publishing after approval
   on the `pypi` environment.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](https://github.com/Rishav-Upadhaya/SWLP/blob/main/LICENSE).
