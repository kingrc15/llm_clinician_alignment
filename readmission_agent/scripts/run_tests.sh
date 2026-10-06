#!/usr/bin/env bash
# Run reasoning_tree unit tests using this repo's .venv (avoids conda base
# Python + NumPy 2 / SciPy binary mismatches).
#
# Usage (from anywhere):
#   /path/to/readmission_agent_share/scripts/run_tests.sh
# Or:
#   cd readmission_agent_share && ./scripts/run_tests.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  echo "ERROR: ${REPO_ROOT}/.venv/bin/python not found." >&2
  echo "Create the venv and install:  python -m venv .venv && .venv/bin/pip install -e '.[graph]'" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${REPO_ROOT}/.venv/bin/python" -m unittest discover -s "${REPO_ROOT}/reasoning_tree_tests" -v "$@"
