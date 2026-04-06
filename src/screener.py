"""
Global Screener Engine
  WATCHLIST           — 25-ticker universe (MAG7 + USA blue-chips + NIFTY50 top 10)
  MARKET_CAPS         — approximate market caps for treemap sizing
  build_screener_row  — compute one row of the screener grid
  build_screener_df   — parallel-compute all rows (cached 15 min)
  get_signal_description — Gemini one-liner for each signal
"""

import time
import os
import re
from pathlib import Path

import diskcache
import pandas as pd

_CACHE_DIR = str(Path(__file__).parent.parent / "data" / ".screener_cache")
_cache     = diskcache.Cache(_CACHE_DIR)
_TTL_GRID  = 900    # 15 min grid refresh
_TTL_DESC  = 86400  # 24 h signal descriptions

# ── Watchlist universe ─────────────────────────────────────────────────────────
WATCHLIST: dict[str, dict] = {
    # MAG7
    "AAPL":        {"name": "Apple",          "sector": "Tech",    "region": "USA",   "cap_b": 3300},
    "MSFT":        {"name": "Microsoft",       "sector": "Tech",    "region": "USA",   "cap_b": 3100},
    "NVDA":        {"name": "NVIDIA",          "sector": "Tech",    "region": "USA",   "cap_b": 2900},
    "GOOGL":       {"name": "Alphabet",        "sector": "Tech",    "region": "USA",   "cap_b": 2100},
    "AMZN":        {"name": "Amazon",          "sector": "Cons",    "region": "USA",   "cap_b": 1900},
    "META":        {"name": "Meta",            "sector": "Tech",    "region": "USA",   "cap_b": 1400},
    "TSLA":        {"name": "Tesla",           "sector": "Auto",    "region": "USA",   "cap_b": 780},
    # USA ETFs & Blue-chips
    "SPY":         {"name": "S&P 500 ETF",     "sector": "ETF",     "region": "USA",   "cap_b": 580},
    "QQQ":         {"name": "Nasdaq-100 ETF",  "sector": "ETF",     "region": "USA",   "cap_b": 290},
    "ITA":         {"name": "Defense ETF",     "sector": "Defense", "region": "USA",   "cap_b": 34},
    "GLD":         {"name": "Gold ETF",        "sector": "Commod",  "region": "USA",   "cap_b": 72},
    "JPM":         {"name": "JPMorgan",        "sector": "Finance", "region": "USA",   "cap_b": 680},
    "V":           {"name": "Visa",            "sector": "Finance", "region": "USA",   "cap_b": 530},
    "JNJ":         {"name": "J&J",             "sector": "Health",  "region": "USA",   "cap_b": 370},
    "XOM":         {"name": "ExxonMobil",      "sector": "Energy",  "region": "USA",   "cap_b": 490},
    # NIFTY50 top 10
    "RELIANCE.NS": {"name": "Reliance",        "sector": "Energy",  "region": "India", "cap_b": 220},
    "TCS.NS":      {"name": "TCS",             "sector": "Tech",    "region": "India", "cap_b": 170},
    "HDFCBANK.NS": {"name": "HDFC Bank",       "sector": "Finance", "region": "India", "cap_b": 160},
    "INFY.NS":     {"name": "Infosys",         "sector": "Tech",    "region": "India", "cap_b": 85},
    "ICICIBANK.NS":{"name": "ICICI Bank",      "sector": "Finance", "region": "India", "cap_b": 100},
    "HINDUNILVR.NS":{"name":"HUL",             "sector": "Cons",    "region": "India", "cap_b": 65},
    "BAJFINANCE.NS":{"name":"Bajaj Finance",   "sector": "Finance", "region": "India", "cap_b": 55},
    "SBIN.NS":     {"name": "SBI",             "sector": "Finance", "region": "India", "cap_b": 75},
    "WIPRO.NS":    {"name": "Wipro",           "sector": "Tech",    "region": "India", "cap_b": 35},
    "LT.NS":       {"name": "Larsen & Toubro", "sector": "Infra",   "region": "India", "cap_b": 60},
}

# ── Column tooltips (beginner translation layer) ───────────────────────────────
COLUMN_TOOLTIPS = {
    "Ticker":      "The stock's unique identifier on the exchange.",
    "Name":        "Full company name.",
    "Region":      "USA = US exchanges; India = NSE/BSE.",
    "Sector":      "Industry the company operates in.",
    "Price":       "Latest available market price in the stock's native currency.",
    "24h Chg%":    "How much the price moved vs the previous day's close. Green = up, Red = down.",
    "Oracle":      "AI recommendation: BUY, HOLD, or SELL — combining sentiment + technicals + macro.",
    "Geo Score":   "Gemini AI sentiment score for geopolitical risk. +1 = very bullish, -1 = very bearish.",
    "Veracity%":   "How reliable the latest headline is. >85% means high confidence the news is real.",
    "Macro Risk":  "Measures how much the Federal Reserve's interest rates and inflation are hurting this specific stock. HIGH = significant headwind.",
    "ATR%":        "Average True Range as % of price — measures daily price swing. High ATR = more volatile stock.",
    "Signal":      "One-line AI explanation of why the Oracle chose this recommendation.",
}

# ── Gemini key ────────────────────────────────────────────────────────────────
def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    try:
        toml = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"
        if toml.exists():
            m = re.search(r'GEMINI_API_KEY\s*=\s*"([^"]+)"', toml.read_text())
            if m:
                return m.group(1)
    except Exception:
        pass
    return ""


def get_signal_description(ticker: str, signal: str, composite: float,
                            geo: float, tech: float, macro_note: str) -> str:
    """
    Generate a one-line plain-English signal description via Gemini.
    Cached 24h per (ticker, signal, rounded composite).
    """
    ck  = f"sigdesc:{ticker}:{signal}:{round(composite,1)}"
    hit = _cache.get(ck)
    if hit:
        return hit

    # Mock descriptions (always available)
    sig_map = {
        "BUY":  f"{ticker} shows bullish momentum — positive geo sentiment ({geo:+.2f}) "
                f"and strong technical trend ({tech:+.2f}) support accumulation.",
        "SELL": f"{ticker} faces headwinds — bearish geo risk ({geo:+.2f}) "
                f"and weak technicals ({tech:+.2f}) suggest reducing exposure.",
        "HOLD": f"{ticker} is range-bound — mixed signals ({composite:+.2f} composite) "
                f"indicate waiting for a clearer directional catalyst.",
    }
    desc = sig_map.get(signal, f"Oracle composite {composite:+.2f}.")

    key = _get_api_key()
    if key:
        prompt = (
            f"You are a financial analyst. Write ONE sentence (max 15 words) explaining "
            f"why {ticker} has a {signal} signal. Facts: geo={geo:+.2f}, tech={tech:+.2f}, "
            f"composite={composite:+.2f}, macro='{macro_note[:60]}'. "
            f"Be specific and plain-English. No jargon."
        )
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            resp = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
            desc = resp.text.strip()[:120]
        except Exception:
            pass

    _cache.set(ck, desc, expire=_TTL_DESC)
    return desc


def build_screener_row(ticker: str, headline: str, macro: dict) -> dict | None:
    """
    Compute one complete screener row for a ticker.
    Returns None on failure so the caller can skip it gracefully.
    """
    from ingest  import load_or_fetch, is_indian, get_macro_oracle_adjustment
    from analyze import (get_technical_indicators, score_geopolitical_risk_full,
                         score_geopolitical_risk_mock)
    from filter  import score as veracity_score

    meta = WATCHLIST.get(ticker, {})
    ccy  = "₹" if is_indian(ticker) else "$"

    try:
        df = load_or_fetch(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            return None

        closes = df["Close"].astype(float)
        price  = float(closes.iloc[-1])
        prev_c = float(closes.iloc[-2]) if len(closes) > 1 else price
        chg_pct = (price - prev_c) / prev_c * 100 if prev_c else 0.0

        ti      = get_technical_indicators(df)
        tech    = ti.get("score", 0.0)
        atr_pct = ti.get("atr14_pct") or 0.0

        try:
            geo, geo_unc = score_geopolitical_risk_full(headline)
        except Exception:
            geo, geo_unc = score_geopolitical_risk_mock(headline)

        v_score  = veracity_score(headline)
        ma       = get_macro_oracle_adjustment(ticker, macro)

        from analyze import get_recommendation
        rec = get_recommendation(geo, tech, geo_unc, ti.get("uncertainty", 0.2),
                                 ma["adjustment"], ma["note"], ticker)

        # Macro risk label
        macro_adj = abs(ma["adjustment"])
        macro_risk = ("HIGH"   if macro_adj >= 0.20 else
                      "MEDIUM" if macro_adj >= 0.08 else "LOW")

        # Signal description (Gemini, cached)
        sig_desc = get_signal_description(
            ticker, rec["signal"], rec["composite"],
            geo, tech, ma.get("note", ""),
        )

        return {
            "Ticker":     ticker,
            "Name":       meta.get("name", ticker),
            "Region":     meta.get("region", "?"),
            "Sector":     meta.get("sector", "?"),
            "Price":      f"{ccy}{price:,.2f}",
            "24h Chg%":   round(chg_pct, 2),
            "Oracle":     rec["signal"],
            "Geo Score":  round(geo, 3),
            "Veracity%":  round(v_score * 100, 1),
            "Macro Risk": macro_risk,
            "ATR%":       round(atr_pct, 2),
            "Signal":     sig_desc,
            # Internal fields for filtering / colouring
            "_composite": round(rec["composite"], 3),
            "_price_raw": price,
            "_chg_raw":   round(chg_pct, 2),
            "_cap_b":     meta.get("cap_b", 10),
            "_geo_raw":   round(geo, 3),
        }
    except Exception as exc:
        print(f"[screener] {ticker} failed: {exc.__class__.__name__}: {exc}")
        return None


def build_screener_df(headline: str, macro: dict,
                      tickers: list[str] | None = None,
                      force: bool = False) -> pd.DataFrame:
    """
    Build the full screener grid. Results cached 15 min.
    Rows that fail are silently skipped.
    """
    tickers = tickers or list(WATCHLIST.keys())
    ck = f"screener:{'|'.join(tickers)}:{headline[:40]}"
    if not force and ck in _cache:
        return pd.DataFrame(_cache[ck])

    rows = []
    for tkr in tickers:
        row = build_screener_row(tkr, headline, macro)
        if row:
            rows.append(row)
        time.sleep(0.05)   # tiny throttle — avoid hammering yfinance

    if rows:
        _cache.set(ck, rows, expire=_TTL_GRID)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_treemap(df: pd.DataFrame) -> object:
    """
    Build a Plotly treemap sized by market cap, coloured by Geo Score.
    df must have columns: Name, Sector, Region, _cap_b, _geo_raw, Oracle, Ticker
    """
    import plotly.graph_objects as go

    if df.empty:
        return go.Figure()

    # Build labels/parents for treemap hierarchy: World > Region > Sector > Ticker
    ids, labels, parents, values, colors, customdata = [], [], [], [], [], []

    # Root
    ids.append("World"); labels.append("🌍 World"); parents.append(""); values.append(0)
    colors.append(0); customdata.append("")

    regions = df["Region"].unique().tolist()
    for reg in regions:
        ids.append(reg); labels.append(f"{'🇺🇸' if reg=='USA' else '🇮🇳'} {reg}")
        parents.append("World"); values.append(0); colors.append(0); customdata.append("")

        sub = df[df["Region"] == reg]
        sectors = sub["Sector"].unique().tolist()
        for sec in sectors:
            sid = f"{reg}_{sec}"
            ids.append(sid); labels.append(sec)
            parents.append(reg); values.append(0); colors.append(0); customdata.append("")

            sec_rows = sub[sub["Sector"] == sec]
            for _, r in sec_rows.iterrows():
                leaf_id = r["Ticker"]
                ids.append(leaf_id)
                labels.append(f"{r['Ticker']}<br>{r['Oracle']}")
                parents.append(sid)
                values.append(max(1, r["_cap_b"]))
                colors.append(r["_geo_raw"])
                customdata.append(
                    f"{r['Name']}<br>Signal: {r['Oracle']}<br>"
                    f"Geo: {r['_geo_raw']:+.3f}<br>Price: {r['Price']}<br>"
                    f"24h: {r['_chg_raw']:+.2f}%"
                )

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        customdata=customdata,
        hovertemplate="<b>%{label}</b><br>%{customdata}<extra></extra>",
        marker=dict(
            colors=colors,
            colorscale=[[0, "#ff1744"], [0.35, "#ff4757"],
                        [0.5, "#ffa500"], [0.65, "#00cc96"], [1, "#00e676"]],
            cmin=-1, cmax=1,
            showscale=True,
            colorbar=dict(
                title="Geo Score",
                titlefont=dict(color="#64748b", size=10),
                tickfont=dict(color="#64748b", size=9),
                thickness=12, len=0.7,
            ),
        ),
        textfont=dict(size=11, color="white"),
        pathbar=dict(visible=True),
        branchvalues="total",
        maxdepth=3,
    ))
    fig.update_layout(
        title=dict(
            text="Global Market Mood — Sized by Market Cap · Coloured by AI Geo Sentiment",
            font=dict(size=13, color="#94a3b8"),
        ),
        template="plotly_dark",
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=44, b=0),
    )
    return fig
