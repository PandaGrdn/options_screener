#!/bin/bash
# Daily ATM-IV snapshot. Cron should call this after the US cash close.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
LOG="$DIR/iv_snapshot.log"
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  "$DIR/.venv/bin/python" "$DIR/screener.py" snapshot "$DIR/iv_history.csv"
} >>"$LOG" 2>&1
