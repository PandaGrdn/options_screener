"""Paper trading package. Forecast before evaluate — never reverse that order."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# chain_history/ (monthly YYYY-MM.csv partitions) lives at repo root (Actions);
# single-file chain_history.csv kept as a legacy/test fallback
CHAIN_HISTORY_DIR = ROOT / "chain_history"
CHAIN_HISTORY_CANDIDATES = (
    CHAIN_HISTORY_DIR,
    DATA / "chain_history.csv",
    ROOT / "chain_history.csv",
)

FORECASTS = DATA / "forecasts.csv"
TRADES = DATA / "trades.csv"
MARKS = DATA / "marks.csv"

UNDERLYING_HISTORY_CANDIDATES = (
    DATA / "underlying_history.csv",
    ROOT / "underlying_history.csv",
)
EARNINGS_CALENDAR_CANDIDATES = (
    DATA / "earnings_calendar.csv",
    ROOT / "earnings_calendar.csv",
)

PORTFOLIO = 5000.0
MAX_PER_TRADE_PCT = 0.04
MAX_DEPLOYED_PCT = 0.20
REFERENCE_ONLY = frozenset({"SPY", "QQQ", "IWM"})


def chain_history_path() -> Path:
    for p in CHAIN_HISTORY_CANDIDATES:
        if p.exists():
            return p
    return CHAIN_HISTORY_CANDIDATES[0]


def read_chain_history(path: Path | None = None):
    """All chain rows from a partition directory (chain_history/*.csv, in month
    order) or a single CSV file."""
    import pandas as pd

    path = Path(path or chain_history_path())
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no chain history partitions in {path}")
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return pd.read_csv(path)


def underlying_history_path() -> Path:
    for p in UNDERLYING_HISTORY_CANDIDATES:
        if p.exists():
            return p
    return UNDERLYING_HISTORY_CANDIDATES[0]


def earnings_calendar_path() -> Path:
    for p in EARNINGS_CALENDAR_CANDIDATES:
        if p.exists():
            return p
    return EARNINGS_CALENDAR_CANDIDATES[0]


def ensure_data_dir() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
