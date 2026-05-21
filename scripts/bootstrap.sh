#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
TORCH_INDEX_URL="${SWLP_TORCH_INDEX_URL:-}"

select_python() {
  local candidate
  local version

  if [[ -n "${PYTHON:-}" ]] && command -v "$PYTHON" >/dev/null 2>&1; then
    candidate="$PYTHON"
    version="$($candidate -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
      printf '%s' "$candidate"
      return 0
    fi
  fi

  for candidate in python3.13 python3.12 python3.11 python3 python; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi

    version="$($candidate -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
      printf '%s' "$candidate"
      return 0
    fi
  done

  printf 'No Python 3.11+ interpreter found. Set PYTHON to a supported interpreter and retry.\n' >&2
  exit 1
}

PYTHON_BIN="$(select_python)"

if [[ -d "$VENV_DIR" && -x "$VENV_DIR/bin/python" ]]; then
  if ! "$VENV_DIR/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    rm -rf "$VENV_DIR"
  fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

INSTALL_ARGS=("-e" "$ROOT_DIR[dev]")
if [[ -n "$TORCH_INDEX_URL" ]]; then
  INSTALL_ARGS=("--extra-index-url" "$TORCH_INDEX_URL" "${INSTALL_ARGS[@]}")
fi

python -m pip install "${INSTALL_ARGS[@]}"
