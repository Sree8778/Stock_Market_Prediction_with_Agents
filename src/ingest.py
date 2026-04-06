"""
Multi-Market Data Ingestor  +  FRED Macro Engine
  USA  : NYSE/NASDAQ — SPY, QQQ, ITA, NVDA, AAPL
  India: NSE/BSE     — RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ^NSEI
  FRED : CPI (CPIAUCSL), Fed Funds Rate (FEDFUNDS)
  FX   : Live USD/INR via USDINR=X
"""

import io
import time
from pathlib import Path

import diskcache
import pandas as pd
import requests
import yfinance as yf

DATA_DIR    = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

_CACHE_DIR  = str(DATA_DIR / ".macro_cache")
_cache      = diskcache.Cache(_CACHE_DIR)
_TTL_MACRO  = 86_400          # FRED: refresh once per day
_TTL_FX     = 300             # FX: refresh every 5 min

PERIOD      = "5y"
INTERVAL    = "1d"
MIN_SAMPLES = 1_000

# ── Market registries ──────────────────────────────────────────────────────────
USA_TICKERS = {
    "SPY":  "S&P 500 ETF (NYSE)",
    "QQQ":  "Nasdaq-100 ETF (NASDAQ)",
    "ITA":  "iShares Defense ETF (NYSE)",
    "NVDA": "NVIDIA Corp (NASDAQ)",
    "AAPL": "Apple Inc (NASDAQ)",
}
INDIA_TICKERS = {
    "RELIANCE.NS": "Reliance Industries (NSE)",
    "TCS.NS":      "Tata Consultancy Services (NSE)",
    "INFY.NS":     "Infosys Ltd (NSE)",
    "HDFCBANK.NS": "HDFC Bank (NSE)",
    "^NSEI":       "Nifty 50 Index (NSE)",
}
ALL_TICKERS = {**USA_TICKERS, **INDIA_TICKERS}
TICKERS     = list(USA_TICKERS.keys())

# ── FRED series ────────────────────────────────────────────────────────────────
FRED_SERIES = {
    "CPIAUCSL":  "CPI — All Urban Consumers (YoY proxy)",
    "FEDFUNDS":  "Federal Funds Effective Rate",
    "T10YIE":    "10-Year Breakeven Inflation Rate",
    "UNRATE":    "US Unemployment Rate",
}
_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="


def fetch_fred(series_id: str, n_periods: int = 24) -> pd.Series:
    """
    Download a FRED time-series CSV and return a pd.Series (monthly or daily).
    Cached for 24 hours.  Returns empty Series on failure.
    """
    cache_key = f"fred:{series_id}"
    if cache_key in _cache:
        return _cache[cache_key]

    try:
        url  = _FRED_BASE + series_id
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(
            io.StringIO(resp.text),
            index_col=0, parse_dates=True,
        ).squeeze()
        df = df.replace(".", float("nan")).dropna().astype(float)
        df = df.tail(n_periods)
        _cache.set(cache_key, df, expire=_TTL_MACRO)
        return df
    except Exception as exc:
        print(f"[ingest/FRED] {series_id} fetch failed: {exc}")
        return pd.Series(dtype=float)


def get_macro_state() -> dict:
    """
    Compute current macro regime from FRED data.
    Returns a rich dict used by the Investment Oracle.
    """
    cache_key = "macro:state"
    if cache_key in _cache:
        return _cache[cache_key]

    cpi_s      = fetch_fred("CPIAUCSL", n_periods=14)
    fed_s      = fetch_fred("FEDFUNDS", n_periods=6)
    breakeven  = fetch_fred("T10YIE",   n_periods=3)
    unrate     = fetch_fred("UNRATE",   n_periods=3)

    # CPI YoY %
    cpi_yoy = float("nan")
    if len(cpi_s) >= 13:
        cpi_yoy = round((cpi_s.iloc[-1] / cpi_s.iloc[-13] - 1) * 100, 2)

    # Fed Funds Rate — latest value
    fed_rate = round(float(fed_s.iloc[-1]),  2) if not fed_s.empty else float("nan")
    fed_prev = round(float(fed_s.iloc[-2]),  2) if len(fed_s) >= 2 else fed_rate

    # Derived regime flags
    inflation_high  = (not pd.isna(cpi_yoy))  and cpi_yoy  > 3.5
    hawkish         = (not pd.isna(fed_rate)) and fed_rate  > 4.5
    rate_rising     = (not pd.isna(fed_rate)) and (not pd.isna(fed_prev)) and fed_rate > fed_prev
    be_inflation    = round(float(breakeven.iloc[-1]), 2) if not breakeven.empty else float("nan")
    unemployment    = round(float(unrate.iloc[-1]),    2) if not unrate.empty    else float("nan")

    # Regime label
    if hawkish and inflation_high:
        regime = "HAWKISH"
        regime_color = "#ff4757"
        regime_note  = "High inflation + elevated Fed rate — risk-off for growth/tech"
    elif hawkish and not inflation_high:
        regime = "RESTRICTIVE"
        regime_color = "#ffa502"
        regime_note  = "Fed remains restrictive; inflation cooling but rates still high"
    elif not hawkish and inflation_high:
        regime = "BEHIND-THE-CURVE"
        regime_color = "#ff6b35"
        regime_note  = "Inflation elevated but Fed is accommodative — stagflation risk"
    else:
        regime = "DOVISH"
        regime_color = "#00ff9d"
        regime_note  = "Low inflation + low/falling rates — growth-friendly environment"

    # Macro sell-pressure multiplier for tech (QQQ/growth)
    # Range: 0.0 (no pressure) → 1.0 (max pressure)
    tech_pressure = 0.0
    if hawkish:         tech_pressure += 0.4
    if inflation_high:  tech_pressure += 0.3
    if rate_rising:     tech_pressure += 0.2
    if not pd.isna(be_inflation) and be_inflation > 2.5:
        tech_pressure += 0.1
    tech_pressure = round(min(1.0, tech_pressure), 3)

    state = {
        "cpi_yoy":        cpi_yoy,
        "fed_rate":       fed_rate,
        "fed_prev":       fed_prev,
        "be_inflation":   be_inflation,
        "unemployment":   unemployment,
        "inflation_high": inflation_high,
        "hawkish":        hawkish,
        "rate_rising":    rate_rising,
        "regime":         regime,
        "regime_color":   regime_color,
        "regime_note":    regime_note,
        "tech_pressure":  tech_pressure,
        "cpi_series":     cpi_s.tail(13).to_dict(),
        "fed_series":     fed_s.to_dict(),
        "ts":             pd.Timestamp.now().isoformat(),
    }
    _cache.set(cache_key, state, expire=_TTL_MACRO)
    return state


def get_macro_oracle_adjustment(ticker: str, macro: dict) -> dict:
    """
    Return an additive adjustment to the Oracle composite score
    based on FRED macro state and the specific ticker.

    Returns:
        {adjustment: float, note: str, pressure: float}
    """
    TECH_TICKERS   = {"QQQ", "NVDA", "AAPL", "MSFT", "META", "TSLA", "AMZN"}
    DEFENSE_TICKERS = {"ITA", "LMT", "NOC", "RTX", "GD"}
    base   = ticker.upper().replace(".NS", "").replace(".BO", "")
    adj    = 0.0
    notes  = []

    if base in TECH_TICKERS:
        # Hawkish + high inflation → drag on tech valuations
        if macro.get("hawkish") and macro.get("inflation_high"):
            adj   -= 0.35
            notes.append(f"Hawkish FOMC ({macro['fed_rate']}%) + CPI {macro['cpi_yoy']}% "
                         f"→ P/E compression risk for growth")
        elif macro.get("hawkish"):
            adj   -= 0.20
            notes.append(f"Restrictive Fed ({macro['fed_rate']}%) pressures tech multiples")
        elif macro.get("inflation_high"):
            adj   -= 0.15
            notes.append(f"CPI {macro['cpi_yoy']}% elevated — real rate squeeze on tech")

    if base in DEFENSE_TICKERS:
        # Defense benefits from geopolitical tension (implicit in hawkish environment)
        if macro.get("hawkish"):
            adj   += 0.10
            notes.append("Defense spending resilient in high-rate environment")

    if base == "SPY":
        # Broad market — partial tech drag
        if macro.get("hawkish") and macro.get("inflation_high"):
            adj -= 0.15
            notes.append("Broad market faces valuation headwinds from FOMC hawkishness")

    return {
        "adjustment":   round(adj, 3),
        "note":         " | ".join(notes) if notes else "No macro adjustment applied",
        "tech_pressure": macro.get("tech_pressure", 0.0),
        "regime":        macro.get("regime", "UNKNOWN"),
    }


# ── Currency ────────────────────────────────────────────────────────────────────
_fx_cache: dict = {}

def get_usd_inr() -> float:
    now = time.time()
    if _fx_cache.get("ts", 0) > now - _TTL_FX:
        return _fx_cache["rate"]
    try:
        t    = yf.Ticker("USDINR=X")
        hist = t.history(period="1d", interval="1m", auto_adjust=True)
        rate = float(hist["Close"].iloc[-1]) if not hist.empty else 83.5
    except Exception:
        rate = 83.5
    _fx_cache.update({"rate": rate, "ts": now})
    return rate


def convert_to_inr(price_usd: float) -> float:
    return round(price_usd * get_usd_inr(), 2)


def is_indian(ticker: str) -> bool:
    return ticker.endswith(".NS") or ticker.endswith(".BO") or ticker.startswith("^")


# ── Dataset guard ──────────────────────────────────────────────────────────────
def verify_dataset(df: pd.DataFrame, ticker: str = "",
                   min_samples: int = MIN_SAMPLES) -> dict:
    n  = len(df)
    ok = n >= min_samples
    print(f"[ingest] {'OK' if ok else 'WARN'} {ticker}: {n:,} rows "
          f"({'>='+str(min_samples) if ok else '<'+str(min_samples)+' required'})")
    return {"ok": ok, "n": n, "min": min_samples, "ticker": ticker,
            "message": f"{'OK' if ok else 'WARN'} — {ticker}: {n:,} rows"}


# ── Core fetchers ──────────────────────────────────────────────────────────────
def fetch_historical(ticker: str = "SPY", period: str = PERIOD) -> pd.DataFrame:
    print(f"[ingest] Fetching {ticker} historical ({period}) ...")
    try:
        t   = yf.Ticker(ticker)
        df  = t.history(period=period, interval=INTERVAL, auto_adjust=True)
        if df is None or df.empty:
            print(f"[ingest] WARN {ticker}: empty response from yfinance")
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df.index = pd.to_datetime(df.index).tz_localize(None)
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df   = df[cols].dropna()
        verify_dataset(df, ticker)
        safe = ticker.replace("^", "_").replace(".", "_")
        path = DATA_DIR / f"{safe}_historical.csv"
        df.to_csv(path)
        return df
    except Exception as exc:
        print(f"[ingest] fetch_historical({ticker}) failed: {exc.__class__.__name__}: {exc}")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def fetch_realtime(ticker: str = "SPY") -> pd.DataFrame:
    """
    Fetch 1-min bars for the last 7 days.
    Returns an empty DataFrame on any failure (market closed, network error,
    yfinance API change, etc.) so callers can fall back gracefully.
    """
    try:
        t  = yf.Ticker(ticker)
        df = t.history(period="7d", interval="1m", auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df.index = pd.to_datetime(df.index).tz_localize(None)
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[cols].dropna()
    except Exception as exc:
        print(f"[ingest] fetch_realtime({ticker}) failed: {exc.__class__.__name__}: {exc}")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def load_or_fetch(ticker: str = "SPY", refresh: bool = False) -> pd.DataFrame:
    safe = ticker.replace("^", "_").replace(".", "_")
    path = DATA_DIR / f"{safe}_historical.csv"
    if path.exists() and not refresh:
        if time.time() - path.stat().st_mtime < 86_400:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            verify_dataset(df, ticker)
            return df
    return fetch_historical(ticker)


def fetch_all(tickers: list | None = None) -> dict:
    return {t: load_or_fetch(t) for t in (tickers or TICKERS)}


def get_dataset_report(tickers: list | None = None) -> list:
    tickers = tickers or list(ALL_TICKERS.keys())
    report  = []
    for t in tickers:
        safe = t.replace("^", "_").replace(".", "_")
        path = DATA_DIR / f"{safe}_historical.csv"
        if path.exists():
            df  = pd.read_csv(path, index_col=0, parse_dates=True)
            row = verify_dataset(df, t)
        else:
            row = {"ok": False, "n": 0, "min": MIN_SAMPLES, "ticker": t,
                   "message": f"NOT CACHED — {t}"}
        row["label"]  = ALL_TICKERS.get(t, t)
        row["market"] = "India" if is_indian(t) else "USA"
        report.append(row)
    return report


def get_latest_bar(ticker: str = "SPY") -> dict:
    df = fetch_realtime(ticker)
    if df.empty:
        return {}
    last = df.iloc[-1]
    p    = float(last["Close"])
    return {
        "ticker": ticker, "timestamp": str(df.index[-1]),
        "open": round(last["Open"], 2), "high": round(last["High"], 2),
        "low":  round(last["Low"],  2), "close": p,
        "volume": int(last["Volume"]),
        "close_inr": convert_to_inr(p) if not is_indian(ticker) else None,
        "usd_inr": get_usd_inr(),
    }


if __name__ == "__main__":
    print("=== FRED Macro State ===")
    m = get_macro_state()
    print(f"CPI YoY:    {m['cpi_yoy']}%")
    print(f"Fed Rate:   {m['fed_rate']}%")
    print(f"Regime:     {m['regime']} — {m['regime_note']}")
    print(f"Tech Pres:  {m['tech_pressure']}")
    print(f"\nQQQ adj:   {get_macro_oracle_adjustment('QQQ', m)}")
    print(f"ITA adj:    {get_macro_oracle_adjustment('ITA', m)}")
