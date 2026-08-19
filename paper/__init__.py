"""Paper trading package. Forecast before evaluate — never reverse that order."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# chain_history lives at repo root (Actions) with data/ fallback
CHAIN_HISTORY_CANDIDATES = (
    DATA / "chain_history.csv",
    ROOT / "chain_history.csv",
)

FORECASTS = DATA / "forecasts.csv"
TRADES = DATA / "trades.csv"
MARKS = DATA / "marks.csv"

PORTFOLIO = 5000.0
MAX_PER_TRADE_PCT = 0.04
MAX_DEPLOYED_PCT = 0.20
REFERENCE_ONLY = frozenset({"SPY", "QQQ", "IWM"})


def chain_history_path() -> Path:
    for p in CHAIN_HISTORY_CANDIDATES:
        if p.exists():
            return p
    return CHAIN_HISTORY_CANDIDATES[0]


def ensure_data_dir() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
