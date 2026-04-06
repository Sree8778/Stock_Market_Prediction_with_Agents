"""
Market Regime Detector
  detect_regime(df, geo_score, v_score, tech_ind, macro) -> dict
    {regime, label, color, description, confidence, factors}

Regime labels:
  CHAOS     — volatility > 2× 30-day mean AND veracity is HIGH (real news driving panic)
  BEARISH   — geo < -0.35 OR (tech < -0.20 AND macro hawkish)
  VOLATILE  — ATR at 30-day high but conditions don't meet CHAOS threshold
  BULLISH   — composite signals positive, low volatility
  STABLE    — everything neutral / mixed

A Gemini Flash qualitative description is generated and cached 6h.
"""

import os
import re
import time
from pathlib import Path

import diskcache
import numpy as np

_CACHE_DIR = str(Path(__file__).parent.parent / "data" / ".regime_cache")
_cache     = diskcache.Cache(_CACHE_DIR)
_TTL       = 21_600   # 6 h

# ── Key resolution ─────────────────────────────────────────────────────────────
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


_REGIME_PROMPT = """\
You are a senior market strategist. Describe the current market regime in exactly 2 sentences.
Be direct and prescriptive. No markdown, no headers.

Regime    : {regime}
Ticker    : {ticker}
Geo Score : {geo:+.3f} (Gemini-scored news sentiment)
Tech Score: {tech:+.3f}
ATR Pctile: {atr_pctile:.0f}th of 30-day range (higher = more volatile)
Veracity  : {v_score:.3f} (NB classifier — 0=fake, 1=real)
Macro     : {macro_regime} | Fed {fed_rate}% | CPI {cpi_yoy}%
Social    : {social:+.3f}

Write 2 sentences: (1) what is driving this regime, (2) what action it implies for investors.\
"""

_MOCK_DESCRIPTIONS = {
    "CHAOS":    "Extreme real-news driven volatility is dominating price action — verified headlines "
                "are creating sharp dislocations across assets. Investors should reduce leverage, "
                "widen stop-losses, and avoid entering new positions until volatility mean-reverts.",
    "BEARISH":  "Bearish technical and macro signals are aligned — geopolitical risk and restrictive "
                "monetary conditions are compressing risk appetite. Consider reducing exposure or "
                "adding defensive hedges (GLD, ITA) while waiting for a clearer reversal signal.",
    "VOLATILE": "Volatility is elevated relative to the 30-day baseline, suggesting market "
                "participants are pricing in uncertainty. Favour options strategies or tighter "
                "position sizing until the ATR normalises.",
    "BULLISH":  "A constructive confluence of positive geo sentiment, bullish technicals, and "
                "accommodative macro conditions defines the current regime. Momentum strategies "
                "and growth-oriented positions are favoured at this juncture.",
    "STABLE":   "Mixed signals across sentiment, technicals, and macro produce a neutral regime. "
                "Range-bound price action is likely — prefer mean-reversion setups and avoid "
                "chasing breakouts until a directional catalyst emerges.",
}


def detect_regime(
    df,
    geo_score:    float,
    v_score:      float,
    tech_ind:     dict,
    macro:        dict,
    ticker:       str = "",
    social_score: float = 0.0,
) -> dict:
    """
    Detect the current market regime and return a full intelligence dict.
    """
    closes  = df["Close"].astype(float) if df is not None else None

    # ── Factor 1: Realised volatility vs 30-day mean ──────────────────────────
    atr_pctile  = tech_ind.get("atr_pctile", 50.0) or 50.0
    bbw_pctile  = tech_ind.get("bbw_pctile", 50.0) or 50.0
    tech_score  = tech_ind.get("score", 0.0)

    # Compute rolling 30-day vol and compare to recent 5-day vol
    vol_ratio  = 1.0
    if closes is not None and len(closes) >= 35:
        rets      = closes.pct_change().dropna()
        vol_30    = float(rets.tail(30).std())
        vol_5     = float(rets.tail(5).std())
        vol_ratio = vol_5 / vol_30 if vol_30 > 0 else 1.0

    # ── Factor 2: News veracity ────────────────────────────────────────────────
    veracity_high = v_score >= 0.60   # NB is confident headline is real

    # ── Regime classification ──────────────────────────────────────────────────
    factors  = {}
    hawkish  = macro.get("hawkish", False)
    inf_high = macro.get("inflation_high", False)

    factors["vol_ratio"]     = round(vol_ratio, 2)
    factors["atr_pctile"]    = round(atr_pctile, 1)
    factors["veracity_high"] = veracity_high
    factors["geo_score"]     = round(geo_score, 3)
    factors["tech_score"]    = round(tech_score, 3)
    factors["hawkish"]       = hawkish

    if vol_ratio >= 2.0 and veracity_high and atr_pctile >= 60:
        regime     = "CHAOS"
        color      = "#ff1744"
        css_class  = "regime-chaos"
        confidence = min(1.0, vol_ratio / 3.0)
        emoji      = "🔴"
    elif geo_score <= -0.35 or (tech_score <= -0.20 and (hawkish or inf_high)):
        regime     = "BEARISH"
        color      = "#ff4757"
        css_class  = "regime-bear"
        confidence = min(1.0, (abs(geo_score) + abs(tech_score)) / 2)
        emoji      = "🔻"
    elif atr_pctile >= 70 or bbw_pctile >= 70:
        regime     = "VOLATILE"
        color      = "#ffa500"
        css_class  = "regime-neutral"
        confidence = min(1.0, atr_pctile / 100)
        emoji      = "⚡"
    elif geo_score >= 0.25 and tech_score >= 0.15:
        regime     = "BULLISH"
        color      = "#00e676"
        css_class  = "regime-dove"
        confidence = min(1.0, (geo_score + tech_score) / 2)
        emoji      = "🟢"
    else:
        regime     = "STABLE"
        color      = "#94a3b8"
        css_class  = "regime-neutral"
        confidence = 0.5
        emoji      = "🔵"

    # ── Gemini qualitative description ────────────────────────────────────────
    desc_key = f"regime_desc:{regime}:{ticker}:{round(geo_score,1)}:{round(tech_score,1)}"
    description = _cache.get(desc_key)

    if description is None:
        key_avail = bool(_get_api_key())
        if key_avail:
            try:
                import google.generativeai as genai
                genai.configure(api_key=_get_api_key())
                client = genai.GenerativeModel("gemini-2.5-flash")
                prompt = _REGIME_PROMPT.format(
                    regime=regime, ticker=ticker or "portfolio",
                    geo=geo_score, tech=tech_score,
                    atr_pctile=atr_pctile, v_score=v_score,
                    macro_regime=macro.get("regime","?"),
                    fed_rate=macro.get("fed_rate","?"),
                    cpi_yoy=macro.get("cpi_yoy","?"),
                    social=social_score,
                )
                resp = client.generate_content(prompt)
                description = resp.text.strip()
                _cache.set(desc_key, description, expire=_TTL)
            except Exception:
                description = _MOCK_DESCRIPTIONS.get(regime, "")
        else:
            description = _MOCK_DESCRIPTIONS.get(regime, "")

    return {
        "regime":      regime,
        "label":       f"{emoji} {regime}",
        "color":       color,
        "css_class":   css_class,
        "confidence":  round(confidence, 3),
        "description": description,
        "factors":     factors,
        "emoji":       emoji,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ingest  import load_or_fetch, get_macro_state
    from analyze import get_technical_indicators

    df   = load_or_fetch("SPY")
    ti   = get_technical_indicators(df)
    mac  = get_macro_state()
    reg  = detect_regime(df, geo_score=-0.2, v_score=0.8, tech_ind=ti, macro=mac, ticker="SPY")
    print(f"Regime: {reg['label']}  (conf {reg['confidence']:.0%})")
    print(f"Description: {reg['description']}")
