#!/usr/bin/env bash
set -euo pipefail
PY="${PY:-$(command -v python || true)}"
PY="${PY:-$(command -v python3 || true)}"

echo "[1/6] Python:"; if [ -n "${PY}" ]; then "$PY" -V; else echo "(ingen python hittad)"; fi
echo "[2/6] Which python:"; echo "${PY:-"(none)"}"
echo "[3/6] Pip list (kort):"; if [ -n "${PY}" ]; then "$PY" -m pip list | head -n 25 || true; else echo "(skip)"; fi
echo "[4/6] Git branch/status:"; git rev-parse --abbrev-ref HEAD; git status -s || true
echo "[5/6] .env nycklar (maskat):"; grep -E 'BORS|BORSDATA|API|KEY' .env 2>/dev/null | sed 's/=.*/=*****/' || echo "(ingen .env hittad)"
echo "[6/6] Venv:"; [ -d ".venv" ] && echo "finns .venv" || echo "saknar .venv"
