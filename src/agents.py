"""
Bull vs. Bear Analyst Debate Engine
  run_debate(ticker, geo, tech, macro, headline) -> {bull, bear, verdict}
  Two Gemini Flash agents with opposing mandates, cached 24h per (ticker, headline).
"""

import re
import time
from pathlib import Path

import diskcache

_CACHE_DIR = str(Path(__file__).parent.parent / "data" / ".agent_cache")
_cache     = diskcache.Cache(_CACHE_DIR)
_TTL       = 86_400   # 24 h


# ── Gemini client (reuses key logic from analyze.py) ──────────────────────────
def _get_client():
    import os, google.generativeai as genai
    from pathlib import Path as P

    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
    if not key:
        try:
            toml = P(__file__).parent.parent / ".streamlit" / "secrets.toml"
            if toml.exists():
                m = re.search(r'GEMINI_API_KEY\s*=\s*"([^"]+)"', toml.read_text())
                if m:
                    key = m.group(1)
        except Exception:
            pass
    if not key:
        raise EnvironmentError("GEMINI_API_KEY not found.")
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-2.5-flash")


# ── Agent prompts ──────────────────────────────────────────────────────────────
_BULL_PROMPT = """\
You are Agent BULL — a top Wall Street equity analyst whose mandate is to find the STRONGEST BUY case.
Your job is to argue persuasively FOR buying {ticker}.

Context:
  Ticker        : {ticker} ({ticker_desc})
  Technical Score: {tech:+.3f}  (range -1 to +1, positive = bullish)
  RSI-14        : {rsi}
  SMA20 vs SMA50: {sma_note}
  Geo Sentiment : {geo:+.3f}  (Gemini-scored headline)
  Headline      : "{headline}"
  Macro Regime  : {regime}  (Fed Rate: {fed_rate}%,  CPI YoY: {cpi_yoy}%)
  Macro Note    : {macro_note}

Instructions:
  1. Find 3 compelling BUY arguments from the technicals, macro, and news.
  2. Acknowledge any bear case briefly, then rebut it.
  3. Conclude with a conviction level: HIGH / MEDIUM / LOW and a 1-sentence target thesis.
  4. Keep response to 180 words max. Use short bullet points. Be direct and confident.
  5. Start with "BULL CASE:" on its own line.\
"""

_BEAR_PROMPT = """\
You are Agent BEAR — a contrarian hedge-fund analyst whose mandate is to find the STRONGEST SELL/SHORT case.
Your job is to argue persuasively AGAINST buying {ticker}.

Context:
  Ticker        : {ticker} ({ticker_desc})
  Technical Score: {tech:+.3f}  (range -1 to +1, negative = bearish)
  RSI-14        : {rsi}
  SMA20 vs SMA50: {sma_note}
  Geo Sentiment : {geo:+.3f}  (Gemini-scored headline)
  Headline      : "{headline}"
  Macro Regime  : {regime}  (Fed Rate: {fed_rate}%,  CPI YoY: {cpi_yoy}%)
  Macro Note    : {macro_note}
  Tech Pressure : {tech_pressure:.0%}  (hawkish macro pressure on growth/tech)

Instructions:
  1. Find 3 compelling SELL/AVOID arguments from geopolitical risk, macro headwinds, and technicals.
  2. Acknowledge any bull case briefly, then tear it down.
  3. Conclude with a conviction level: HIGH / MEDIUM / LOW and a 1-sentence downside thesis.
  4. Keep response to 180 words max. Use short bullet points. Be direct and analytical.
  5. Start with "BEAR CASE:" on its own line.\
"""

_VERDICT_PROMPT = """\
You are the Chief Investment Officer. Two analysts have debated {ticker}.
Synthesize their arguments into a final verdict.

BULL CASE:
{bull}

BEAR CASE:
{bear}

Macro: {regime} | Fed {fed_rate}% | CPI {cpi_yoy}% | Oracle Signal: {signal} ({composite:+.3f})

Output a JSON object ONLY (no markdown):
{{
  "verdict": "BUY" | "HOLD" | "SELL",
  "confidence": <0.0 to 1.0>,
  "winner": "BULL" | "BEAR" | "DRAW",
  "summary": "<2 sentences max>",
  "key_risk": "<single biggest risk>",
  "key_catalyst": "<single biggest upside catalyst>"
}}\
"""

_MOCK_BULL = {
    "QQQ":  "BULL CASE:\n• SMA20 > SMA50 signals momentum strength.\n• AI/cloud mega-trend drives secular earnings growth.\n• Fed pivot odds rising — multiple expansion likely.\n\nConviction: MEDIUM — Tech leadership intact but macro headwinds persist.",
    "SPY":  "BULL CASE:\n• Broad market breadth improving across sectors.\n• Earnings estimates being revised upward for Q3.\n• Defensive rotation into dividend payers provides floor.\n\nConviction: MEDIUM — Economy resilient; soft landing base case.",
    "ITA":  "BULL CASE:\n• Defense spending mandated to grow with NATO commitments.\n• Geopolitical tensions sustain multi-year procurement cycles.\n• ITA at 52-week high signals institutional accumulation.\n\nConviction: HIGH — Structural demand thesis unchallenged.",
    "_default": "BULL CASE:\n• Technical momentum positive with SMA cross signal.\n• Earnings growth trajectory intact despite headwinds.\n• Institutional accumulation visible in volume patterns.\n\nConviction: MEDIUM — Risk/reward favours longs at current levels.",
}
_MOCK_BEAR = {
    "QQQ":  "BEAR CASE:\n• Fed at {fed_rate}% — historically crushes P/E multiples for growth.\n• CPI {cpi_yoy}% still above 2% target; no pivot yet.\n• RSI overbought territory — mean reversion risk elevated.\n\nConviction: MEDIUM — Rate sensitivity is the dominant risk.",
    "SPY":  "BEAR CASE:\n• Inverted yield curve historically precedes recession.\n• Consumer credit stress rising; savings rate at multi-year lows.\n• Valuations stretched at 20x forward earnings vs historical 16x.\n\nConviction: MEDIUM — Late cycle signals warrant defensive positioning.",
    "ITA":  "BEAR CASE:\n• Defense stocks pricing in maximum conflict premium.\n• Any ceasefire news would trigger sharp profit-taking.\n• High relative valuation vs historical P/E range.\n\nConviction: LOW — Structural bull thesis limits downside.",
    "_default": "BEAR CASE:\n• Macro headwinds from elevated rates compress valuations.\n• Geopolitical uncertainty raises discount rate for risk assets.\n• Technical resistance at current levels; breakout unconfirmed.\n\nConviction: MEDIUM — Risk management suggests caution near resistance.",
}


def _call_gemini(prompt: str, retries: int = 2) -> str:
    client = _get_client()
    for attempt in range(retries):
        try:
            resp = client.generate_content(prompt)
            return resp.text.strip()
        except Exception as exc:
            err = str(exc)
            if "429" in err or "quota" in err.lower():
                import re as _re
                m = _re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err)
                wait = int(m.group(1)) + 1 if m else 20
                if attempt < retries - 1:
                    time.sleep(wait)
                    continue
            if attempt >= retries - 1:
                raise
    return ""


def run_debate(
    ticker:    str,
    geo:       float,
    tech:      float,
    macro:     dict,
    tech_ind:  dict,
    headline:  str,
    oracle_rec: dict,
    ticker_desc: str = "",
) -> dict:
    """
    Run Bull vs Bear debate. Returns:
      {bull: str, bear: str, verdict: dict, cached: bool, error: str|None}
    """
    from ingest import ALL_TICKERS
    ticker_desc = ticker_desc or ALL_TICKERS.get(ticker, ticker)
    cache_key   = f"debate:{ticker}:{headline.strip().lower()[:80]}"

    if cache_key in _cache:
        cached = _cache[cache_key]
        cached["cached"] = True
        return cached

    rsi      = tech_ind.get("rsi14", "N/A")
    sma20    = tech_ind.get("sma20", 0)
    sma50    = tech_ind.get("sma50", 0)
    sma_note = (
        f"SMA20 ({sma20}) ABOVE SMA50 ({sma50}) — bullish cross"
        if tech_ind.get("sma_signal", 0) > 0
        else f"SMA20 ({sma20}) BELOW SMA50 ({sma50}) — bearish cross"
    )
    ctx = dict(
        ticker=ticker, ticker_desc=ticker_desc,
        geo=geo, tech=tech,
        rsi=rsi, sma_note=sma_note,
        headline=headline,
        regime=macro.get("regime", "UNKNOWN"),
        fed_rate=macro.get("fed_rate", "N/A"),
        cpi_yoy=macro.get("cpi_yoy", "N/A"),
        macro_note=macro.get("regime_note", ""),
        tech_pressure=macro.get("tech_pressure", 0.0),
        signal=oracle_rec.get("signal", "HOLD"),
        composite=oracle_rec.get("composite", 0.0),
    )

    bull_text = bear_text = verdict_dict = None
    error_msg = None

    try:
        bull_text = _call_gemini(_BULL_PROMPT.format(**ctx))
        time.sleep(1.5)   # polite delay between requests
        bear_text = _call_gemini(_BEAR_PROMPT.format(**ctx))
        time.sleep(1.5)
        verdict_raw = _call_gemini(_VERDICT_PROMPT.format(
            ticker=ticker,
            bull=bull_text[:500], bear=bear_text[:500],
            **{k: ctx[k] for k in ("regime","fed_rate","cpi_yoy","signal","composite")},
        ))
        # Parse verdict JSON
        import json
        clean = re.sub(r"```[a-z]*", "", verdict_raw).strip("` \n")
        verdict_dict = json.loads(clean)
    except Exception as exc:
        error_msg = str(exc)
        # Fallback mocks
        base = ticker.upper()
        if bull_text is None:
            tmpl = _MOCK_BULL.get(base, _MOCK_BULL["_default"])
            bull_text = tmpl.format(**ctx) if "{" in tmpl else tmpl
        if bear_text is None:
            tmpl = _MOCK_BEAR.get(base, _MOCK_BEAR["_default"])
            bear_text = tmpl.format(**ctx) if "{" in tmpl else tmpl
        if verdict_dict is None:
            composite = oracle_rec.get("composite", 0.0)
            verdict_dict = {
                "verdict":   oracle_rec.get("signal", "HOLD"),
                "confidence": oracle_rec.get("confidence", 0.3),
                "winner":    "BULL" if composite > 0.1 else "BEAR" if composite < -0.1 else "DRAW",
                "summary":   f"Oracle composite {composite:+.3f} — {oracle_rec.get('rationale','')[:100]}",
                "key_risk":  macro.get("regime_note", "Macro uncertainty"),
                "key_catalyst": "Technical momentum reversal or Fed pivot signal",
            }

    result = {
        "bull":    bull_text,
        "bear":    bear_text,
        "verdict": verdict_dict,
        "cached":  False,
        "error":   error_msg,
        "context": {
            "ticker": ticker, "geo": geo, "tech": tech,
            "regime": macro.get("regime"), "fed_rate": macro.get("fed_rate"),
        },
    }
    if not error_msg:
        _cache.set(cache_key, result, expire=_TTL)
    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ingest  import get_macro_state, load_or_fetch
    from analyze import score_geopolitical_risk_full, get_technical_indicators, get_recommendation

    ticker  = "QQQ"
    hl      = "Fed signals rates to stay higher for longer amid sticky inflation"
    df      = load_or_fetch(ticker)
    geo, gu = score_geopolitical_risk_full(hl)
    tech_d  = get_technical_indicators(df)
    macro   = get_macro_state()
    rec     = get_recommendation(geo, tech_d["score"])
    debate  = run_debate(ticker, geo, tech_d["score"], macro, tech_d, hl, rec)

    print("=== BULL ==="); print(debate["bull"])
    print("\n=== BEAR ==="); print(debate["bear"])
    print("\n=== VERDICT ==="); print(debate["verdict"])
