"""
Agentic Daily Market Recap Engine  —  Gemini 2.5 Flash
  get_daily_recap(tickers, scores, macro, headlines) -> dict
    {summary: str, sentiment_rows: list, cached: bool, error: str|None}

Generates a ~300-word "Global Market Summary" grounded in:
  - Technical scores for each watchlist ticker
  - FRED macro regime (CPI, Fed Rate)
  - Optional list of analysed headlines

Cached once per calendar day per combination of market data.
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import diskcache

_CACHE_DIR = str(Path(__file__).parent.parent / "data" / ".recap_cache")
_cache     = diskcache.Cache(_CACHE_DIR)
_TTL       = 86_400     # 24 h

# ── Gemini key resolution (mirrors analyze.py pattern) ────────────────────────
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


def _get_client():
    import google.generativeai as genai
    key = _get_api_key()
    if not key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-2.5-flash")


# ── Prompt template ───────────────────────────────────────────────────────────
_RECAP_PROMPT = """\
You are a senior market strategist writing the "Daily Global Market Intelligence Brief."
Write exactly 300 words. Use professional but clear language. No markdown headers.
Use paragraph breaks between sections.

Today: {date}
Macro Regime: {regime} | Fed Rate: {fed_rate}% | CPI YoY: {cpi_yoy}% | Regime Note: {regime_note}

Watchlist Scores (Oracle composite -1 to +1):
{score_table}

Recent Headlines Analysed:
{headline_block}

Write 3 paragraphs:
1. Macro environment overview — what the Fed/CPI regime means for equity risk appetite right now.
2. Sector spotlight — highlight the strongest and weakest tickers from the watchlist, with brief reasoning.
3. Key risk & opportunity — one actionable risk to monitor and one catalyst to watch.

End with a one-line "Oracle Verdict:" summarising the overall market bias (Bullish/Neutral/Bearish)
and a confidence level (High/Medium/Low).\
"""

_MOCK_SUMMARY = """\
Global markets continue to navigate a complex macro backdrop as the Federal Reserve maintains a watchful stance on inflation. With the current {regime} regime indicating {regime_note}, equity risk appetite remains measured, particularly for growth-sensitive sectors that face valuation pressure from elevated real rates.

Among the watchlist, {best_ticker} stands out with the strongest Oracle composite signal, supported by bullish technical momentum and constructive sector dynamics. Conversely, {worst_ticker} shows elevated headwinds, with technical indicators flagging potential mean reversion risk against a backdrop of macro uncertainty. Indian markets continue to demonstrate relative resilience, with domestic consumption-driven names outperforming export-linked peers amid currency stability.

The primary risk to monitor remains a re-acceleration in inflation data that could force a hawkish policy pivot, compressing P/E multiples across the growth spectrum. On the opportunity side, any confirmed dovish signal — whether from FOMC communications or a below-consensus CPI print — would likely catalyse a broad risk-on rotation, with technology and emerging market equities positioned to benefit most.

Oracle Verdict: {verdict} | Confidence: Medium\
"""


def _build_score_table(ticker_scores: dict) -> str:
    lines = []
    for tkr, info in ticker_scores.items():
        score  = info.get("composite", info.get("score", 0.0))
        signal = info.get("signal", "HOLD")
        desc   = info.get("desc", tkr)
        lines.append(f"  {tkr:<18} {score:+.3f}  [{signal}]  {desc}")
    return "\n".join(lines) or "  (no ticker data)"


def _build_headline_block(headlines: list[str]) -> str:
    if not headlines:
        return "  (none — using macro/technical signals only)"
    return "\n".join(f"  - {h[:120]}" for h in headlines[:10])


# ── Public API ─────────────────────────────────────────────────────────────────
def get_daily_recap(
    ticker_scores: dict,        # {ticker: {composite, signal, desc}}
    macro:         dict,        # from get_macro_state()
    headlines:     list[str] | None = None,
    force:         bool = False,
) -> dict:
    """
    Generate (or retrieve cached) daily market summary.

    Args:
        ticker_scores : {ticker: {composite, signal, desc}} — Oracle scores per ticker
        macro         : FRED macro state dict
        headlines     : list of recent news headlines analysed today
        force         : bypass 24h cache

    Returns:
        {summary, sentiment_rows, date, regime, cached, error}
    """
    today = datetime.now().strftime("%Y-%m-%d")
    # Cache key changes with the day and overall market bias
    scores_hash = sum(
        hash(f"{k}{round(v.get('composite', 0.0), 1)}")
        for k, v in ticker_scores.items()
    )
    cache_key = f"recap:{today}:{macro.get('regime','?')}:{scores_hash % 9999}"

    if not force and cache_key in _cache:
        cached = dict(_cache[cache_key])
        cached["cached"] = True
        return cached

    regime      = macro.get("regime", "UNKNOWN")
    fed_rate    = macro.get("fed_rate", "N/A")
    cpi_yoy     = macro.get("cpi_yoy", "N/A")
    regime_note = macro.get("regime_note", "")

    score_table    = _build_score_table(ticker_scores)
    headline_block = _build_headline_block(headlines or [])

    prompt = _RECAP_PROMPT.format(
        date=today,
        regime=regime, fed_rate=fed_rate, cpi_yoy=cpi_yoy, regime_note=regime_note,
        score_table=score_table, headline_block=headline_block,
    )

    summary    = None
    error_msg  = None

    key_available = bool(_get_api_key())
    if key_available:
        for attempt in range(3):
            try:
                client  = _get_client()
                resp    = client.generate_content(prompt)
                summary = resp.text.strip()
                break
            except Exception as exc:
                err = str(exc)
                if "429" in err or "quota" in err.lower():
                    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err)
                    wait = int(m.group(1)) + 2 if m else 30
                    if attempt < 2:
                        time.sleep(wait)
                        continue
                error_msg = err
                break

    if summary is None:
        # Fallback mock
        scores = [v.get("composite", 0.0) for v in ticker_scores.values()]
        tickers = list(ticker_scores.keys())
        best  = tickers[scores.index(max(scores))] if tickers else "SPY"
        worst = tickers[scores.index(min(scores))] if tickers else "QQQ"
        avg   = sum(scores) / len(scores) if scores else 0.0
        verdict = "Bullish" if avg > 0.15 else "Bearish" if avg < -0.15 else "Neutral"
        summary = _MOCK_SUMMARY.format(
            regime=regime, regime_note=regime_note,
            best_ticker=best, worst_ticker=worst, verdict=verdict,
        )

    # Build sentiment summary table for the UI
    sentiment_rows = [
        {
            "Ticker":    tkr,
            "Signal":    v.get("signal", "HOLD"),
            "Composite": round(v.get("composite", 0.0), 3),
            "Desc":      v.get("desc", tkr)[:40],
        }
        for tkr, v in ticker_scores.items()
    ]

    result = {
        "summary":        summary,
        "sentiment_rows": sentiment_rows,
        "date":           today,
        "regime":         regime,
        "cached":         False,
        "error":          error_msg,
        "headlines_used": len(headlines or []),
    }
    if not error_msg:
        _cache.set(cache_key, result, expire=_TTL)
    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ingest  import get_macro_state
    from analyze import get_recommendation

    macro = get_macro_state()
    tscores = {
        "SPY":  {"composite": 0.12, "signal": "HOLD",  "desc": "S&P 500 ETF"},
        "QQQ":  {"composite": 0.05, "signal": "HOLD",  "desc": "Nasdaq-100 ETF"},
        "ITA":  {"composite": 0.44, "signal": "BUY",   "desc": "Defense ETF"},
    }
    result = get_daily_recap(tscores, macro, headlines=["Fed holds rates steady"])
    print(result["summary"])
