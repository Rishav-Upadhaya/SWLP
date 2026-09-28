## What this PR does

## Checklist

- [ ] `pytest` passes (all tests, no GPU or model download needed)
- [ ] `ruff check src/` is clean
- [ ] New functions have tests in `tests/` (use `MockRunner`, never real models)
- [ ] Follows the folder layout: runners in `runner/`, algorithms in `core/`,
      reporters in `reporting/` (see `.claude/CLAUDE.md`)
- [ ] No hardcoded model paths, devices, or magic numbers — config via
      `AppConfig` / TOML / `SWLP_*` env vars

## Benchmarks (if performance-related)

Hardware, model, and before/after numbers.
