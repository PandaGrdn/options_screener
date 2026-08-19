#!/bin/bash
# Daily near-ATM chain snapshot (raw bid/ask). Prefer GitHub Actions; this is a local fallback.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
LOG="$DIR/iv_snapshot.log"
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  "$DIR/.venv/bin/python" "$DIR/snapshot.py"
} >>"$LOG" 2>&1
