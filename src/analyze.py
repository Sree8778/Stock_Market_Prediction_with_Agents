"""
Sentiment + Oracle Engine — Gemini 2.0 Flash Lite
  score_geopolitical_risk(text)     -> float [-1, 1]   (cached 24h)
  get_technical_indicators(df)      -> dict  (SMA, RSI, MACD, Bollinger)
  get_recommendation(geo, tech)     -> dict  with XAI confidence interval
  get_confidence_interval(score, u) -> (lower, upper)

All Gemini calls and cross-market results are stored in diskcache (24h TTL).
"""

import json
import math
import os
import re
import time
from pathlib import Path

import diskcache

# ── Persistent cache (24-hour TTL) ────────────────────────────────────────────
_CACHE_DIR = str(Path(__file__).parent.parent / "data" / ".sentiment_cache")
_cache     = diskcache.Cache(_CACHE_DIR)
_TTL       = 86_400

# ── Gemini client ──────────────────────────────────────────────────────────────
_client = None


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
    global _client
    if _client is None:
        import google.generativeai as genai
        key = _get_api_key()
        if not key:
            raise EnvironmentError("GEMINI_API_KEY not set.")
        genai.configure(api_key=key)
        _client = genai.GenerativeModel("gemini-2.5-flash")
    return _client


_SYSTEM_PROMPT = """\
You are a quantitative geopolitical risk analyst for global equities (USA & India).
Given a news headline, output ONLY a JSON object:
  {"score": <float -1.0 to 1.0>, "uncertainty": <float 0.0 to 0.5>}

score scale:
  -1.0  extreme negative (war, crash, sanctions, disaster)
  -0.5  moderately negative (tension, rate hike, supply shock)
   0.0  neutral
  +0.5  moderately positive (trade deal, strong earnings)
  +1.0  highly positive (peace deal, breakthrough stimulus)

uncertainty: your confidence interval half-width (0=certain, 0.5=very uncertain).

Respond with ONLY raw JSON. No markdown, no explanation.\
"""

# ── Keyword mock ───────────────────────────────────────────────────────────────
_KW = {
    "war": -0.9, "missile": -0.8, "sanction": -0.7, "crash": -0.8,
    "conflict": -0.6, "tension": -0.5, "inflation": -0.4, "rate hike": -0.4,
    "ceasefire": 0.6, "deal": 0.5, "peace": 0.7, "earnings": 0.4,
    "growth": 0.4, "recovery": 0.5, "stimulus": 0.6, "surge": 0.3,
    "drop": -0.3, "fall": -0.3, "rise": 0.3, "beat": 0.4,
    "deficit": -0.3, "record": 0.3, "expansion": 0.4, "recession": -0.6,
    "contract": 0.3, "awarded": 0.4, "upgrade": 0.3, "downgrade": -0.4,
    "rbi": -0.1, "sebi": 0.0, "nifty": 0.1, "sensex": 0.1,
    "rupee": -0.1, "crude": -0.2, "tariff": -0.3, "export": 0.2,
}


def score_geopolitical_risk_mock(text: str) -> tuple[float, float]:
    """Returns (score, uncertainty)."""
    t    = text.lower()
    hits = [v for k, v in _KW.items() if k in t]
    if not hits:
        return 0.0, 0.25
    return round(sum(hits) / len(hits), 3), 0.15


# ── Primary scorer ─────────────────────────────────────────────────────────────
def score_geopolitical_risk(text: str, retries: int = 3) -> float:
    """Public API — returns scalar score only (backward-compat)."""
    score, _ = score_geopolitical_risk_full(text, retries)
    return score


def score_geopolitical_risk_full(text: str,
                                  retries: int = 3) -> tuple[float, float]:
    """
    Returns (score, uncertainty) where:
      score       ∈ [-1, 1]
      uncertainty ∈ [0, 0.5]  — model-reported half-width of 95% CI
    Results are cached for 24h to minimise API calls across markets.
    """
    cache_key = f"geo_full:{text.strip().lower()}"
    if cache_key in _cache:
        return _cache[cache_key]

    try:
        client = _get_client()
    except EnvironmentError:
        result = score_geopolitical_risk_mock(text)
        _cache.set(cache_key, result, expire=_TTL)
        return result

    for attempt in range(retries):
        try:
            resp = client.generate_content(f"{_SYSTEM_PROMPT}\n\nText: {text}")
            raw  = re.sub(r"```[a-z]*", "", resp.text.strip()).strip("` \n")
            data = json.loads(raw)
            score = float(max(-1.0, min(1.0, data.get("score", 0.0))))
            unc   = float(max(0.0,  min(0.5, data.get("uncertainty", 0.1))))
            result = (score, unc)
            _cache.set(cache_key, result, expire=_TTL)
            return result
        except Exception as exc:
            err = str(exc)
            if "429" in err or "quota" in err.lower():
                m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err)
                wait = int(m.group(1)) + 2 if m else 30
                if attempt < retries - 1:
                    print(f"[analyze] Rate limited — waiting {wait}s")
                    time.sleep(wait)
                    continue
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"[analyze] Gemini fallback to mock ({exc.__class__.__name__})")
                result = score_geopolitical_risk_mock(text)
                _cache.set(cache_key, result, expire=_TTL)
                return result
    return (0.0, 0.25)


def get_scorer():
    return score_geopolitical_risk


# ── Technical Indicators ───────────────────────────────────────────────────────
def get_technical_indicators(df) -> dict:
    """
    Compute a full technical indicator suite from daily OHLCV data.
    Returns a dict with individual signals and a composite score in [-1, 1].
    """
    closes = df["Close"].astype(float)
    n      = len(closes)
    result = {
        "score":       0.0,
        "uncertainty": 0.2,
        "sma20":       None, "sma50": None, "sma200": None,
        "rsi14":       None,
        "macd":        None, "macd_signal": None,
        "bb_upper":    None, "bb_lower":    None, "bb_mid": None,
        "sma_signal":  0.0,  "rsi_signal":  0.0,
        "macd_signal_v": 0.0,
        "bb_signal":   0.0,
    }
    if n < 50:
        return result

    # SMA signals
    sma20  = closes.rolling(20).mean()
    sma50  = closes.rolling(50).mean()
    result["sma20"] = round(sma20.iloc[-1], 2)
    result["sma50"] = round(sma50.iloc[-1], 2)
    sma_sig = 1.0 if sma20.iloc[-1] > sma50.iloc[-1] else -1.0
    result["sma_signal"] = sma_sig

    if n >= 200:
        sma200 = closes.rolling(200).mean()
        result["sma200"]   = round(sma200.iloc[-1], 2)
        sma_sig = (sma_sig + (1.0 if closes.iloc[-1] > sma200.iloc[-1] else -1.0)) / 2

    # RSI-14
    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    loss  = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
    rs    = gain / loss if loss != 0 else float("inf")
    rsi   = 100 - 100 / (1 + rs)
    result["rsi14"]      = round(rsi, 1)
    rsi_norm             = max(-1.0, min(1.0, (rsi - 50) / 50))
    result["rsi_signal"] = round(rsi_norm, 3)

    # MACD (12/26/9)
    if n >= 26:
        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        sig9  = macd.ewm(span=9, adjust=False).mean()
        result["macd"]        = round(macd.iloc[-1], 3)
        result["macd_signal"] = round(sig9.iloc[-1], 3)
        macd_v = 1.0 if macd.iloc[-1] > sig9.iloc[-1] else -1.0
        result["macd_signal_v"] = macd_v

    # Bollinger Bands (20, 2σ)
    bb_mid  = closes.rolling(20).mean()
    bb_std  = closes.rolling(20).std()
    bb_up   = bb_mid + 2 * bb_std
    bb_lo   = bb_mid - 2 * bb_std
    last_c  = closes.iloc[-1]
    result["bb_upper"] = round(bb_up.iloc[-1], 2)
    result["bb_lower"] = round(bb_lo.iloc[-1], 2)
    result["bb_mid"]   = round(bb_mid.iloc[-1], 2)
    bb_range = bb_up.iloc[-1] - bb_lo.iloc[-1]
    if bb_range > 0:
        bb_pos = (last_c - bb_lo.iloc[-1]) / bb_range   # 0=lower band, 1=upper
        bb_sig = (bb_pos - 0.5) * 2                      # -1 to +1
        result["bb_signal"] = round(bb_sig, 3)

    # Composite: SMA 35%, RSI 30%, MACD 20%, BB 15%
    composite = (
        0.35 * result["sma_signal"]    +
        0.30 * result["rsi_signal"]    +
        0.20 * result["macd_signal_v"] +
        0.15 * result["bb_signal"]
    )
    result["score"] = round(max(-1.0, min(1.0, composite)), 3)

    # Technical uncertainty from recent volatility
    vol_20 = closes.pct_change().rolling(20).std().iloc[-1]
    result["uncertainty"] = round(min(0.4, vol_20 * 10), 3)

    # ── ATR-14 (Average True Range) ───────────────────────────────────────────
    if all(c in df.columns for c in ["High", "Low", "Close"]):
        highs  = df["High"].astype(float)
        lows   = df["Low"].astype(float)
        prev_c = closes.shift(1)
        tr     = (
            (highs - lows)
            .combine(abs(highs - prev_c), max)
            .combine(abs(lows  - prev_c), max)
        )
        atr14  = tr.rolling(14).mean()
        result["atr14"]         = round(atr14.iloc[-1], 3)
        result["atr14_pct"]     = round(atr14.iloc[-1] / closes.iloc[-1] * 100, 3)
        # ATR vs 30-day history to classify squeeze
        if len(atr14.dropna()) >= 30:
            atr30 = atr14.dropna().tail(30)
            result["atr_30d_low"]   = round(atr30.min(), 3)
            result["atr_30d_high"]  = round(atr30.max(), 3)
            result["atr_pctile"]    = round(
                (atr14.iloc[-1] - atr30.min()) / (atr30.max() - atr30.min() + 1e-9) * 100, 1
            )
        else:
            result["atr_30d_low"]  = result["atr14"]
            result["atr_30d_high"] = result["atr14"]
            result["atr_pctile"]   = 50.0
    else:
        result["atr14"] = result["atr14_pct"] = None
        result["atr_30d_low"] = result["atr_30d_high"] = result["atr_pctile"] = None

    # ── Bollinger Band Width (normalised) ─────────────────────────────────────
    if result["bb_upper"] and result["bb_lower"] and result["bb_mid"]:
        bbw = (result["bb_upper"] - result["bb_lower"]) / result["bb_mid"]
        result["bb_width"] = round(bbw, 4)
        # BB Width vs 30-day history
        bb_mid_s = closes.rolling(20).mean()
        bb_std_s = closes.rolling(20).std()
        bbw_s    = (bb_mid_s + 2 * bb_std_s - (bb_mid_s - 2 * bb_std_s)) / bb_mid_s
        if len(bbw_s.dropna()) >= 30:
            bbw30 = bbw_s.dropna().tail(30)
            result["bbw_30d_low"]  = round(bbw30.min(), 4)
            result["bbw_30d_high"] = round(bbw30.max(), 4)
            result["bbw_pctile"]   = round(
                (bbw - bbw30.min()) / (bbw30.max() - bbw30.min() + 1e-9) * 100, 1
            )
        else:
            result["bbw_30d_low"]  = result["bb_width"]
            result["bbw_30d_high"] = result["bb_width"]
            result["bbw_pctile"]   = 50.0
    else:
        result["bb_width"] = result["bbw_30d_low"] = result["bbw_30d_high"] = None
        result["bbw_pctile"] = None

    # ── Volatility Squeeze Detection ──────────────────────────────────────────
    # Squeeze = ATR at 30d low AND BB Width at 30d low → coiled spring
    result["squeeze_detected"] = False
    result["squeeze_warning"]  = ""
    atr_pct = result.get("atr_pctile")
    bbw_pct = result.get("bbw_pctile")
    if atr_pct is not None and bbw_pct is not None:
        if atr_pct <= 20 and bbw_pct <= 20:
            result["squeeze_detected"] = True
            result["squeeze_warning"]  = (
                "Volatility Squeeze Detected — Prepare for Sharp Movement. "
                f"ATR at {atr_pct:.0f}th percentile of 30-day range; "
                f"BB Width at {bbw_pct:.0f}th percentile. "
                "Typically precedes a high-velocity directional breakout."
            )

    return result


# Legacy alias kept for backward compatibility
def get_technical_trend(df) -> float:
    return get_technical_indicators(df)["score"]


def get_volatility_sparkline(df, window: int = 24) -> list[float]:
    """
    Returns a list of `window` hourly-equivalent ATR% values derived from
    daily OHLCV data (last `window` bars).  Used for Holdings sparklines.
    Values are percentage ATR relative to close (i.e. daily range / close * 100).
    """
    if df is None or len(df) < 5:
        return []
    closes = df["Close"].astype(float)
    if all(c in df.columns for c in ["High", "Low"]):
        highs  = df["High"].astype(float)
        lows   = df["Low"].astype(float)
        daily_range_pct = ((highs - lows) / closes * 100).round(3)
        return daily_range_pct.tail(window).tolist()
    return (closes.pct_change().abs() * 100).round(3).tail(window).tolist()


# ── XAI Confidence Interval ────────────────────────────────────────────────────
def get_confidence_interval(composite: float,
                             geo_unc: float,
                             tech_unc: float,
                             z: float = 1.96) -> tuple[float, float]:
    """
    95% CI using error propagation for f = 0.6*geo + 0.4*tech.
      σ_composite = sqrt((0.6*σ_geo)^2 + (0.4*σ_tech)^2)
    Returns (lower, upper) clamped to [-1, 1].
    """
    sigma = math.sqrt((0.6 * geo_unc) ** 2 + (0.4 * tech_unc) ** 2)
    lower = max(-1.0, composite - z * sigma)
    upper = min( 1.0, composite + z * sigma)
    return round(lower, 3), round(upper, 3)


# ── Investment Oracle ──────────────────────────────────────────────────────────
def get_recommendation(geo_score: float, tech_score: float,
                        geo_unc: float = 0.15,
                        tech_unc: float = 0.20,
                        macro_adj: float = 0.0,
                        macro_note: str = "",
                        ticker: str = "") -> dict:
    """
    Hybrid Oracle — 60% Geo + 40% Tech + FRED macro adjustment.

    macro_adj: additive shift from get_macro_oracle_adjustment()
               (e.g. -0.35 for QQQ in hawkish/high-CPI environment)
    """
    raw_composite = 0.60 * geo_score + 0.40 * tech_score
    composite     = max(-1.0, min(1.0, raw_composite + macro_adj))
    ci_lo, ci_hi  = get_confidence_interval(composite, geo_unc, tech_unc)
    confidence    = abs(composite)

    if composite >= 0.25:
        signal, color = "BUY",  "#00cc96"
        rationale = (
            f"Bullish composite ({composite:+.3f}): "
            f"Geo {geo_score:+.3f}×0.60 + Tech {tech_score:+.3f}×0.40"
            + (f" + Macro {macro_adj:+.3f}" if macro_adj else "") + ". "
            "Price technicals and macro environment favour accumulation."
        )
    elif composite <= -0.25:
        signal, color = "SELL", "#ef553b"
        rationale = (
            f"Bearish composite ({composite:+.3f}): "
            f"Geo {geo_score:+.3f}×0.60 + Tech {tech_score:+.3f}×0.40"
            + (f" + Macro {macro_adj:+.3f}" if macro_adj else "") + ". "
            "Risk-off conditions suggest reducing exposure."
        )
    else:
        signal, color = "HOLD", "#ffa500"
        rationale = (
            f"Neutral composite ({composite:+.3f}): "
            f"Geo {geo_score:+.3f}×0.60 + Tech {tech_score:+.3f}×0.40"
            + (f" + Macro {macro_adj:+.3f}" if macro_adj else "") + ". "
            "Await stronger directional signal."
        )

    if macro_note:
        rationale += f" | FRED: {macro_note}"

    geo_contribution  = round(0.60 * geo_score,  3)
    tech_contribution = round(0.40 * tech_score, 3)

    return {
        "signal":              signal,
        "color":               color,
        "confidence":          round(confidence, 3),
        "composite":           round(composite,  3),
        "raw_composite":       round(raw_composite, 3),
        "macro_adj":           round(macro_adj,  3),
        "ci_lower":            ci_lo,
        "ci_upper":            ci_hi,
        "ci_width":            round(ci_hi - ci_lo, 3),
        "geo_score":           round(geo_score,  3),
        "tech_score":          round(tech_score, 3),
        "geo_contribution":    geo_contribution,
        "tech_contribution":   tech_contribution,
        "macro_contribution":  round(macro_adj, 3),
        "geo_uncertainty":     round(geo_unc,  3),
        "tech_uncertainty":    round(tech_unc, 3),
        "rationale":           rationale,
        "ticker":              ticker,
    }


if __name__ == "__main__":
    for s in [
        "Fed raises rates amid persistent inflation",
        "NATO forces mobilize near Eastern European border",
        "Nifty hits record high on strong Q3 earnings season",
    ]:
        sc, u = score_geopolitical_risk_full(s)
        print(f"  {sc:+.3f} ±{u:.3f}  {s}")
