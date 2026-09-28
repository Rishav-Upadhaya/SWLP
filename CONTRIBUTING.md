# Contributing to SWLP

Thanks for your interest in **SWLP — Sliding Window Layer Pipeline**! This guide
gets you from a fresh clone to a merged pull request. Contributions of all
sizes are welcome — bug reports, docs fixes, new hardware benchmarks, and code.

The project's most-wanted contribution right now is **benchmark numbers on
hardware we don't have** (especially Linux + NVIDIA). The CUDA streaming path is
built but unmeasured — see [Areas we'd love help with](#areas-wed-love-help-with).

---

## Quick start

```bash
git clone https://github.com/Rishav-Upadhaya/SWLP.git
cd SWLP

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core + pytest + ruff

# Confirm everything works — no model or GPU needed:
pytest                             # all tests must pass
ruff check src/                    # zero errors
swlp --backend mock --prompt "hi" # smoke-test the CLI
```

If those three commands succeed, your environment is ready.

---

## Development workflow

1. **Open an issue first** for anything non-trivial, so we can agree on the
   approach before you write code.
2. **Create a branch** off `main`: `git checkout -b fix/short-description`.
3. **Make your change**, following the conventions below.
4. **Run the checks** (all must be clean):
   ```bash
   pytest                 # all tests pass
   ruff check src/        # zero lint errors
   ruff check --fix src/  # auto-fix what it can
   ```
5. **Add or update tests.** Every new function in `src/swlp/` needs a test in
   `tests/`. Use `MockRunner` for runner tests — never download a real model in
   a unit test. Use `simulate_scenario()` for pipeline math.
6. **Open a pull request** with a clear description of what changed and why.
   Fill in the PR template; link the issue it closes.

CI runs `pytest` and `ruff check src/` on every PR — both must be green to merge.

---

## Project conventions

These keep the codebase consistent and reviewable. The full operating manual is
in [`.claude/CLAUDE.md`](.claude/CLAUDE.md); the highlights:

- **`snake_case`** for variables/functions/files/modules, **`PascalCase`** for
  classes.
- **Full type annotations** on every function (arguments + return). No bare
  `Any`, no module-level globals, no magic numbers — use `AppConfig`, TOML
  configs, dataclasses, and named constants.
- **Never `print()`** for runtime output — use `configure_logging()` from
  `src/swlp/logging.py`. (Direct user-facing CLI/chat output is the only
  exception.)
- **No new file exceeds 300 lines** — split by responsibility.
- **Respect the folder structure** (see the README and `CLAUDE.md`):
  new runners go in `runner/`, algorithm changes in `core/`, reporters in
  `reporting/`, etc. New runners register in `build_runner()`; don't modify
  existing runners.
- **Respect import boundaries** — `core/`, `hardware/`, and `model/` must not
  import from `runner/`, `benchmark/`, or `reporting/`. The full table is in
  `CLAUDE.md`.
- **Don't hardcode** model paths, device strings, or magic numbers — use
  `AppConfig`, TOML configs in `configs/`, and `SWLP_*` env vars.
- **No quality compromise.** SWLP's whole premise is exact FP16 inference.
  Anything lossy (e.g. INT4 KV) must be **opt-in, off by default, and clearly
  labelled**.

---

## Tests

Tests live in `tests/` and mirror the `src/swlp/` layout — each `test_*.py`
maps to one module (e.g. `tests/test_kv_cache.py` → `src/swlp/core/kv_cache.py`).

```bash
pytest                                          # full suite
pytest tests/test_simulator.py                  # one file
pytest -k test_plan_residency                   # one test by name
pytest tests/test_simulator.py::test_overlap    # one exact test
```

The full suite runs in seconds with no GPU or model download — it uses
`MockRunner` and pure-math functions.

---

## Commit messages

Write clear, imperative commit messages (`Add CUDA stream overlap guard`, not
`fixed stuff`). Reference issues where relevant (`Fixes #42`).

---

## Areas we'd love help with

- **NVIDIA / Linux benchmarks.** The CUDA async-PCIe streaming path is wired but
  has no hardware numbers. Run it on an NVIDIA GPU and open a PR with results —
  this is the single highest-value contribution right now.
- **Windows testing.** Untested, but no known blockers.
- **More model architectures** beyond Llama / Mistral / GPT-2 family.
- **Documentation** — clearer guides, fixed typos, better examples.

---

## Reporting bugs

Open an issue using the bug-report template. Include:

- What you ran (the exact `swlp ...` command).
- Your OS, hardware (chip / GPU, RAM), and Python version.
- The full error output (or `--json` metrics if it ran but behaved wrong).

---

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you agree to uphold it.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE) that covers this project.
