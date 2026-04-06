"""
Social Pulse Engine  —  Retail Sentiment Synthesiser
  get_social_sentiment(ticker)  -> dict  {score, label, confidence, sources, cached}
  refresh_social_pulse(tickers) -> None  (background thread)

Since Reddit/X APIs require authentication keys we don't have, this engine uses
a curated seed of representative retail-language snippets per ticker, then passes
them to Gemini 2.5 Flash for aggregate sentiment scoring.  On first run it
bootstraps from keyword heuristics so the gauge always has a value.

Cache: 6-hour TTL (social mood shifts faster than FRED data, slower than prices).
"""

import os
import re
import threading
import time
from pathlib import Path

import diskcache

_CACHE_DIR = str(Path(__file__).parent.parent / "data" / ".social_cache")
_cache     = diskcache.Cache(_CACHE_DIR)
_TTL       = 21_600   # 6 hours

# ── Keyword sentiment lexicon (retail tone) ────────────────────────────────────
_BULL_KW = [
    "moon", "mooning", "to the moon", "🚀", "buy the dip", "hodl",
    "bullish", "breakout", "all-time high", "ath", "strong buy",
    "long", "calls", "yolo", "easy money", "squeeze", "gamma",
    "earnings beat", "growth", "undervalued", "accumulate",
]
_BEAR_KW = [
    "puts", "short", "overvalued", "dump", "crash", "sell",
    "bearish", "recession", "bubble", "bagholders", "rug",
    "disappointing", "miss", "debt", "fear", "panic",
    "overbought", "correction", "red day", "falling knife",
]

# Curated retail-style snippets per ticker (mimic Reddit/X post style)
_SNIPPETS: dict[str, list[str]] = {
    "SPY": [
        "SPY holding support at 200-day SMA, classic buy the dip setup",
        "Broad market breadth improving, bulls in control short term",
        "SPY options flow very bullish this week, large call sweeps",
        "Recession fears creeping back in, watch for SPY breakdown",
        "Fed pause = SPY moon? Historical pattern says yes",
    ],
    "QQQ": [
        "QQQ AI mega-trend is NOT priced in yet, still early innings",
        "Tech valuations stretched, QQQ P/E at 30x is dangerous territory",
        "QQQ gamma squeeze setup forming, short interest elevated",
        "Rate sensitivity killing QQQ, higher for longer is the new normal",
        "QQQ breakout above resistance, all systems go for bulls",
    ],
    "ITA": [
        "ITA defense ETF mooning on geopolitical tensions, NATO spending up",
        "Defense spending is non-cyclical, ITA is the safest play right now",
        "ITA at 52-week highs, is the defense rally getting crowded?",
        "Geopolitical premium in ITA is real and here to stay",
        "ITA earnings cycle is multi-year, still buying dips",
    ],
    "NVDA": [
        "NVDA data center demand is insatiable, buying every dip",
        "NVDA forward P/E at 40x, how much AI hype is already baked in?",
        "NVDA short interest surging, classic setup for another squeeze",
        "Jensen Huang is a genius, NVDA will be first $5T company",
        "NVDA competition intensifying from AMD and custom silicon, be careful",
    ],
    "AAPL": [
        "AAPL services revenue compounding beautifully, hidden gem in plain sight",
        "AAPL iPhone cycle peak worries are overblown, India expansion huge",
        "AAPL buyback machine is unmatched, $90B in repurchases this year",
        "AAPL AI features are table stakes, where is the real innovation?",
        "AAPL China risk is real and underappreciated by bulls",
    ],
    "TSLA": [
        "TSLA FSD is finally working, robotaxi launch will be massive",
        "TSLA market share eroding fast in China and Europe",
        "Elon focused on DOGE, TSLA is on autopilot literally and figuratively",
        "TSLA energy storage business is a sleeping giant",
        "TSLA valuation only makes sense if robotaxi works",
    ],
    "RELIANCE.NS": [
        "Reliance Jio subscriber growth is phenomenal, long term hold",
        "RIL new energy bets are huge capex but necessary for future",
        "Reliance retail is quietly becoming India's Amazon",
        "RIL chairman succession clarity would unlock significant valuation",
        "India infrastructure boom = Reliance wins, no brainer",
    ],
    "TCS.NS": [
        "TCS deal wins accelerating, IT demand revival is real",
        "TCS margins under pressure from wage inflation and pricing",
        "TCS dividend yield attractive for long-term investors",
        "TCS AI transformation spending by clients is a multi-year tailwind",
        "TCS attrition normalised, execution back on track",
    ],
    "INFY.NS": [
        "Infosys guidance cut is concerning, is the worst over?",
        "INFY large deal TCV improving, turnaround in progress",
        "Infosys cheaper than TCS for similar quality, value play",
        "INFY Topaz AI platform gaining traction with enterprise clients",
        "Infosys valuation re-rating possible if guidance improves",
    ],
    "HDFCBANK.NS": [
        "HDFC Bank post-merger integration complete, re-rating incoming",
        "HDFCBANK credit growth strong, NIM pressure is temporary",
        "India's best bank at reasonable valuation, buy and hold",
        "HDFCBANK deposit growth lagging loan growth, watch carefully",
        "HDFCBANK FII buying resuming, institutional confidence returning",
    ],
    "^NSEI": [
        "Nifty breakout above 22k, foreign flows turning positive",
        "Nifty valuations elevated at 22x forward, wait for correction",
        "India macro story intact, Nifty heading to 25k this year",
        "FII selling pressure on Nifty continues, domestic SIPs supporting",
        "Nifty breadth improving, mid and small caps outperforming",
    ],
    "GLD": [
        "Gold breaking out, geopolitical risk premium expanding rapidly",
        "GLD as portfolio hedge makes sense when uncertainty is high",
        "Gold challenging all-time highs, central bank buying accelerating",
        "GLD no yield but ultimate safe haven when chaos hits",
        "Gold demand from India and China central banks is structural",
    ],
}

_DEFAULT_SNIPPETS = [
    "Strong technical setup, momentum building",
    "Mixed signals, market uncertain here",
    "Fundamentals solid despite macro headwinds",
    "Risk/reward unfavorable at current levels",
    "Institutional accumulation visible in volume",
]


# ── Gemini key resolution ──────────────────────────────────────────────────────
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


# ── Keyword-based mock scorer ──────────────────────────────────────────────────
def _keyword_score(snippets: list[str]) -> tuple[float, float]:
    """Returns (score ∈ [-1,1], confidence ∈ [0,1])."""
    total_hits, bull_hits, bear_hits = 0, 0, 0
    for s in snippets:
        sl = s.lower()
        b  = sum(1 for kw in _BULL_KW if kw in sl)
        be = sum(1 for kw in _BEAR_KW if kw in sl)
        bull_hits += b
        bear_hits += be
        total_hits += b + be

    if total_hits == 0:
        return 0.0, 0.35

    score      = (bull_hits - bear_hits) / total_hits
    confidence = min(0.85, 0.4 + total_hits * 0.04)
    return round(score, 3), round(confidence, 3)


_SOCIAL_PROMPT = """\
You are a quantitative retail sentiment analyst. Analyse these social media posts about {ticker}.
Score the AGGREGATE retail sentiment as a single JSON object:
  {{"score": <float -1.0 to 1.0>, "confidence": <float 0.0 to 1.0>, "dominant_theme": "<5 words max>"}}

score scale:
  -1.0  extremely bearish (panic, crashing, sell everything)
  -0.5  moderately bearish (worried, taking profits, cautious)
   0.0  neutral/mixed
  +0.5  moderately bullish (optimistic, buying dips, holding)
  +1.0  extremely bullish (euphoria, YOLO, moon)

confidence: how consistent the sentiment is (0=very mixed, 1=unanimous).
dominant_theme: one short phrase capturing the main retail narrative.

Posts:
{posts}

Respond with ONLY raw JSON. No markdown, no explanation.\
"""


def _score_via_gemini(ticker: str, snippets: list[str]) -> dict:
    import json
    key = _get_api_key()
    if not key:
        raise EnvironmentError("No API key")

    import google.generativeai as genai
    genai.configure(api_key=key)
    client = genai.GenerativeModel("gemini-2.5-flash")

    posts_txt = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets))
    prompt    = _SOCIAL_PROMPT.format(ticker=ticker, posts=posts_txt)

    for attempt in range(3):
        try:
            resp = client.generate_content(prompt)
            raw  = re.sub(r"```[a-z]*", "", resp.text.strip()).strip("` \n")
            data = json.loads(raw)
            score      = float(max(-1.0, min(1.0, data.get("score", 0.0))))
            confidence = float(max(0.0,  min(1.0, data.get("confidence", 0.5))))
            theme      = str(data.get("dominant_theme", "Mixed sentiment"))[:60]
            return {"score": score, "confidence": confidence, "theme": theme, "via": "gemini"}
        except Exception as exc:
            err = str(exc)
            if "429" in err or "quota" in err.lower():
                m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err)
                wait = int(m.group(1)) + 2 if m else 30
                if attempt < 2:
                    time.sleep(wait)
                    continue
            break
    raise RuntimeError("Gemini unavailable")


# ── Public API ─────────────────────────────────────────────────────────────────
def get_social_sentiment(ticker: str, force: bool = False) -> dict:
    """
    Return social sentiment dict for ticker:
      {score, label, confidence, theme, sources, via, cached}
    score ∈ [-1, 1]:  negative = bearish retail mood, positive = bullish
    """
    key = f"social:{ticker.upper()}"
    if not force and key in _cache:
        cached = dict(_cache[key])
        cached["cached"] = True
        return cached

    snippets = _SNIPPETS.get(ticker.upper(), _DEFAULT_SNIPPETS)
    via      = "keyword"
    theme    = "Mixed signals"
    score    = 0.0
    conf     = 0.35

    try:
        result = _score_via_gemini(ticker, snippets)
        score  = result["score"]
        conf   = result["confidence"]
        theme  = result["theme"]
        via    = result["via"]
    except Exception:
        score, conf = _keyword_score(snippets)
        theme = "Retail sentiment (keyword analysis)"

    label = (
        "Strongly Bullish" if score >= 0.60 else
        "Bullish"          if score >= 0.25 else
        "Slightly Bullish" if score >= 0.05 else
        "Neutral"          if score >= -0.05 else
        "Slightly Bearish" if score >= -0.25 else
        "Bearish"          if score >= -0.60 else
        "Strongly Bearish"
    )

    out = {
        "score":      round(score, 3),
        "label":      label,
        "confidence": round(conf, 3),
        "theme":      theme,
        "sources":    len(snippets),
        "via":        via,
        "cached":     False,
    }
    _cache.set(key, out, expire=_TTL)
    return out


def refresh_social_pulse(tickers: list[str]) -> None:
    """
    Background thread: refresh social sentiment for all provided tickers.
    Call once at app start; results land in cache for instant reads.
    """
    def _worker():
        for t in tickers:
            try:
                get_social_sentiment(t, force=False)
                time.sleep(1.2)   # polite delay between Gemini calls
            except Exception:
                pass

    th = threading.Thread(target=_worker, daemon=True, name="social_pulse")
    th.start()
    return th


if __name__ == "__main__":
    for tkr in ["SPY", "QQQ", "RELIANCE.NS"]:
        r = get_social_sentiment(tkr)
        print(f"{tkr:20s}  {r['score']:+.3f}  [{r['label']}]  via={r['via']}")
