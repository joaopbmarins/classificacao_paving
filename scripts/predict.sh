#!/usr/bin/env bash
set -euo pipefail

python -m src.inference.predict "$@"
