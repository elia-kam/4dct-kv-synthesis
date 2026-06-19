#!/usr/bin/env bash
set -euo pipefail

export APP_DATA_DIR="${APP_DATA_DIR:-$PWD/data}"
export DYNAGAN_DIR="${DYNAGAN_DIR:-$HOME/Dynagan}"
export DYNAGAN_PYTHON="${DYNAGAN_PYTHON:-python}"
export DEEPDRR_MODE="${DEEPDRR_MODE:-singularity}"
export DEEPDRR_SIF="${DEEPDRR_SIF:-$HOME/Data/containers/deepdrr.sif}"
export DEEPDRR_BIND="${DEEPDRR_BIND:-$HOME:$HOME}"

mkdir -p "$APP_DATA_DIR"
UVICORN_BIN="${UVICORN_BIN:-uvicorn}"
if [ -x "$PWD/.venv/bin/uvicorn" ]; then
  UVICORN_BIN="$PWD/.venv/bin/uvicorn"
fi

ARGS=(app.main:app --host 0.0.0.0 --port "${PORT:-8080}")
if [ "${DEV_RELOAD:-0}" = "1" ]; then
  ARGS+=(--reload)
fi

"$UVICORN_BIN" "${ARGS[@]}"
