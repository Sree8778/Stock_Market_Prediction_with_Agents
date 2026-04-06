"""
Stock Oracle - Intelligence Sovereign Dashboard
Tabs: Live | Holdings | Oracle | Macro/FRED | CV Metrics
Sidebar: Multi-market + Bull/Bear Debate + Chatbot
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from filter  import score as veracity_score, cross_validate_async, HARDWARE_INFO, TEXTS as NB_CORPUS
from analyze import (
    score_geopolitical_risk_full, score_geopolitical_risk_mock,
    get_recommendation, get_technical_indicators, get_volatility_sparkline,
    _get_api_key,
)
from social_pulse import get_social_sentiment, refresh_social_pulse
from simulation  import run_monte_carlo_cached, build_cone_figure
from xai_shap    import compute_shap_values, build_shap_figure
from regime      import detect_regime
from ingest  import (
    fetch_realtime, load_or_fetch, get_dataset_report,
    get_usd_inr, is_indian, convert_to_inr,
    get_macro_state, get_macro_oracle_adjustment,
    USA_TICKERS, INDIA_TICKERS, ALL_TICKERS, MIN_SAMPLES,
)
from agents              import run_debate
from ticker_db           import resolve as ticker_resolve, search as ticker_search
from screener            import (
    build_screener_df, build_treemap, WATCHLIST, COLUMN_TOOLTIPS,
)
from analyst_view        import render_analyst_view
from correlation         import log_event, get_pearson, bootstrap_from_history
from recap               import get_daily_recap
from prescriptive_engine import (
    get_portfolio_analysis, generate_reasoning_card, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
)

ROOT           = Path(__file__).parent.parent
METRICS_PATH   = ROOT / "data" / "metrics.json"
PORTFOLIO_PATH = ROOT / "data" / "portfolio.json"

st.set_page_config(page_title="Stock Oracle", page_icon="📡",
                   layout="wide", initial_sidebar_state="expanded")

# ── Glassmorphism CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[data-testid="stAppViewContainer"]{
  background:radial-gradient(ellipse at top left,#0d1b35 0%,#060d1a 50%,#000 100%)!important;
  font-family:'Inter',sans-serif!important;color:#e2e8f0!important;}
[data-testid="stSidebar"]{background:rgba(6,13,26,0.93)!important;
  backdrop-filter:blur(28px)!important;border-right:1px solid rgba(99,102,241,0.15)!important;}
.glass{background:rgba(255,255,255,0.035);backdrop-filter:blur(20px);
  border:1px solid rgba(255,255,255,0.08);border-radius:18px;padding:22px 26px;
  margin-bottom:14px;box-shadow:0 8px 32px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,255,255,0.06);}
.glass-sm{background:rgba(255,255,255,0.03);backdrop-filter:blur(16px);
  border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:14px 18px;margin-bottom:10px;}
.glass-debate{background:rgba(255,255,255,0.025);backdrop-filter:blur(12px);border-radius:14px;
  padding:14px 18px;margin-bottom:8px;font-size:0.79rem;line-height:1.7;height:230px;overflow-y:auto;}
.glass-bull{border:1px solid rgba(0,255,157,0.22);}
.glass-bear{border:1px solid rgba(255,71,87,0.22);}
.signal-buy{color:#00ff9d;font-weight:800;font-size:2.6rem;letter-spacing:2px;text-shadow:0 0 28px rgba(0,255,157,0.55);}
.signal-sell{color:#ff4757;font-weight:800;font-size:2.6rem;letter-spacing:2px;text-shadow:0 0 28px rgba(255,71,87,0.55);}
.signal-hold{color:#ffa502;font-weight:800;font-size:2.6rem;letter-spacing:2px;text-shadow:0 0 28px rgba(255,165,2,0.55);}
.regime-hawk{background:rgba(255,71,87,0.14);color:#ff6b6b;border:1px solid rgba(255,71,87,0.3);
  border-radius:8px;padding:4px 14px;font-weight:600;font-size:0.8rem;}
.regime-dove{background:rgba(0,255,157,0.11);color:#00ff9d;border:1px solid rgba(0,255,157,0.25);
  border-radius:8px;padding:4px 14px;font-weight:600;font-size:0.8rem;}
.regime-neutral{background:rgba(255,165,2,0.11);color:#ffa502;border:1px solid rgba(255,165,2,0.25);
  border-radius:8px;padding:4px 14px;font-weight:600;font-size:0.8rem;}
.pill{display:inline-block;padding:3px 12px;border-radius:999px;font-size:0.7rem;font-weight:600;
  letter-spacing:0.05em;margin-left:6px;}
.pill-usa{background:rgba(99,102,241,0.2);color:#a5b4fc;border:1px solid rgba(99,102,241,0.3);}
.pill-india{background:rgba(251,146,60,0.2);color:#fdba74;border:1px solid rgba(251,146,60,0.3);}
.pos{color:#00ff9d;}.neg{color:#ff4757;}.neu{color:#ffa502;}
.section-title{font-size:0.68rem;font-weight:600;text-transform:uppercase;
  letter-spacing:0.12em;color:#475569;margin-bottom:8px;}
.ci-track{background:rgba(255,255,255,0.07);border-radius:999px;height:8px;
  position:relative;overflow:visible;margin:6px 0 2px;}
.ci-fill{height:8px;border-radius:999px;position:absolute;top:0;}
.ci-dot{width:14px;height:14px;border-radius:50%;position:absolute;top:-3px;
  border:2px solid #06101e;transform:translateX(-50%);}
.chat-user{background:rgba(99,102,241,0.16);border-radius:10px;padding:9px 13px;
  margin:4px 0;border-left:3px solid #6366f1;}
.chat-bot{background:rgba(0,204,150,0.09);border-radius:10px;padding:9px 13px;
  margin:4px 0;border-left:3px solid #00cc96;}
.badge{display:inline-block;padding:4px 14px;border-radius:999px;font-size:0.75rem;
  font-weight:700;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:4px;}
.badge-invest{background:rgba(0,230,118,0.18);color:#00e676;border:1px solid rgba(0,230,118,0.4);}
.badge-hold{background:rgba(255,165,0,0.15);color:#ffa500;border:1px solid rgba(255,165,0,0.35);}
.badge-sell{background:rgba(255,71,87,0.18);color:#ff4757;border:1px solid rgba(255,71,87,0.4);}
.badge-stop{background:rgba(255,23,68,0.22);color:#ff1744;border:1px solid rgba(255,23,68,0.5);
  animation:pulse-stop 1.6s ease-in-out infinite;}
.badge-partial{background:rgba(255,109,0,0.18);color:#ff6d00;border:1px solid rgba(255,109,0,0.4);}
.badge-profit{background:rgba(0,229,255,0.15);color:#00e5ff;border:1px solid rgba(0,229,255,0.35);}
@keyframes pulse-stop{0%,100%{box-shadow:0 0 0 0 rgba(255,23,68,0.4);}
  50%{box-shadow:0 0 0 6px rgba(255,23,68,0);}}
.holding-card{background:rgba(255,255,255,0.035);backdrop-filter:blur(18px);
  border:1px solid rgba(255,255,255,0.07);border-radius:16px;padding:18px 22px;
  margin-bottom:12px;transition:border-color 0.2s;}
.holding-card:hover{border-color:rgba(255,255,255,0.14);}
.recap-body{font-size:0.88rem;line-height:1.9;color:#cbd5e1;white-space:pre-wrap;}
.rebalance-tip{background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
  border-radius:10px;padding:12px 16px;margin:6px 0;font-size:0.82rem;color:#a5b4fc;}
.social-bar{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
  border-radius:10px;padding:10px 14px;margin-top:6px;}
.squeeze-warn{background:rgba(255,165,0,0.12);border:1px solid rgba(255,165,0,0.35);
  border-radius:10px;padding:10px 16px;margin:8px 0;font-size:0.8rem;color:#ffd166;
  font-weight:600;}
.regime-chaos{background:rgba(255,23,68,0.18);color:#ff1744;border:1px solid rgba(255,23,68,0.45);
  border-radius:8px;padding:5px 16px;font-weight:800;font-size:0.85rem;
  animation:pulse-stop 1.4s ease-in-out infinite;}
.regime-bear{background:rgba(255,71,87,0.14);color:#ff4757;border:1px solid rgba(255,71,87,0.3);
  border-radius:8px;padding:5px 16px;font-weight:700;font-size:0.85rem;}
.regime-sidebar{border-radius:10px;padding:10px 14px;margin-top:4px;font-size:0.82rem;}
.mc-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px;}
.mc-stat{background:rgba(255,255,255,0.04);border-radius:8px;padding:8px 12px;text-align:center;}
.mc-stat .label{font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:.06em;}
.mc-stat .value{font-size:0.95rem;font-weight:700;margin-top:2px;}
/* ── Screener grid ── */
.screener-header{font-size:0.65rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:#475569;padding:6px 0;}
.screener-row{display:grid;gap:6px;align-items:center;padding:9px 14px;
  border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.82rem;
  transition:background 0.15s;}
.screener-row:hover{background:rgba(255,255,255,0.04);}
.sig-buy{color:#00e676;font-weight:800;font-size:0.78rem;}
.sig-sell{color:#ff4757;font-weight:800;font-size:0.78rem;}
.sig-hold{color:#ffa500;font-weight:800;font-size:0.78rem;}
.chg-pos{color:#00e676;font-weight:600;}
.chg-neg{color:#ff4757;font-weight:600;}
.macro-high{color:#ff4757;font-weight:700;font-size:0.72rem;}
.macro-med{color:#ffa500;font-weight:700;font-size:0.72rem;}
.macro-low{color:#00e676;font-weight:700;font-size:0.72rem;}
.tooltip-col{position:relative;display:inline-block;cursor:help;
  border-bottom:1px dotted #475569;}
.filter-chip{display:inline-block;background:rgba(99,102,241,0.15);
  border:1px solid rgba(99,102,241,0.3);border-radius:999px;
  padding:3px 12px;font-size:0.72rem;color:#a5b4fc;margin:3px 3px 3px 0;}
.screener-stat{background:rgba(255,255,255,0.04);border-radius:10px;
  padding:10px 16px;text-align:center;}
.screener-summary{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px;}
[data-testid="stMetricValue"]{font-size:1.22rem!important;font-weight:700!important;}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,0.03)!important;border-radius:12px;gap:4px;}
.stTabs [data-baseweb="tab"]{border-radius:9px!important;color:#64748b!important;font-weight:500!important;}
.stTabs [aria-selected="true"]{background:rgba(99,102,241,0.2)!important;color:#e2e8f0!important;}
.stButton>button{background:rgba(99,102,241,0.15)!important;border:1px solid rgba(99,102,241,0.3)!important;
  border-radius:10px!important;color:#a5b4fc!important;font-weight:600!important;}
</style>""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("chat_history", []), ("cv_thread", None), ("cv_running", False),
             ("debate_result", None), ("debate_ticker", None), ("debate_running", False),
             ("selected_stock", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ────────────────────────────────────────────────────────────────────
def _geo(text):
    return score_geopolitical_risk_full(text) if _get_api_key() else score_geopolitical_risk_mock(text)

@st.cache_data(ttl=60,    show_spinner=False)
def _rt(t):     return fetch_realtime(t)
@st.cache_data(ttl=86400, show_spinner=False)
def _hist(t):   return load_or_fetch(t)
@st.cache_data(ttl=3600,  show_spinner=False)
def _macro_c(): return get_macro_state()

def _pf():    return json.loads(PORTFOLIO_PATH.read_text()) if PORTFOLIO_PATH.exists() else {"holdings": []}
def _spf(d):  PORTFOLIO_PATH.write_text(json.dumps(d, indent=2))
def _clr(v):  return "#00ff9d" if v >= 0 else "#ff4757"

# ── Live price fetcher (used exclusively by Holdings tab) ─────────────────────
# Precedence: 1-min realtime  →  last daily close  →  yfinance fast_info
# Indian tickers (.NS/.BO) are natively priced in INR by yfinance — no conversion needed.
# US tickers are priced in USD natively.
# Session-state cache lets "Refresh Prices" bypass @st.cache_data without a full rerun.

def _fetch_live_price(tkr: str) -> tuple[float, str]:
    """
    Returns (price, source_label).
    Price is in the ticker's native currency (INR for .NS/.BO, USD for US).
    """
    import yfinance as yf

    # 1. Try 1-min realtime feed
    try:
        df = fetch_realtime(tkr)
        if not df.empty:
            return float(df["Close"].iloc[-1]), "real-time 1m"
    except Exception:
        pass

    # 2. Try last daily close from cached historical CSV
    try:
        df = load_or_fetch(tkr)
        if not df.empty:
            return float(df["Close"].iloc[-1]), "daily close"
    except Exception:
        pass

    # 3. yfinance fast_info (quick API call, no full history)
    try:
        info = yf.Ticker(tkr).fast_info
        p = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
        if p:
            return float(p), "fast_info"
    except Exception:
        pass

    return None, "unavailable"


def _get_portfolio_prices(holdings: list, force: bool = False) -> dict:
    """
    Returns {ticker: (price, source)} for all portfolio holdings.
    Results cached in session_state for 60 s; force=True bypasses cache.
    """
    cache_key  = "pf_prices"
    ts_key     = "pf_prices_ts"
    now        = time.time()
    stale      = (now - st.session_state.get(ts_key, 0)) > 60

    if force or stale or cache_key not in st.session_state:
        prices = {}
        for h in holdings:
            tkr = h["ticker"]
            p, src = _fetch_live_price(tkr)
            prices[tkr] = (p, src)
        st.session_state[cache_key] = prices
        st.session_state[ts_key]    = now

    return st.session_state[cache_key]
def _pill(t):
    cls = "pill-india" if is_indian(t) else "pill-usa"
    lbl = "NSE/BSE"    if is_indian(t) else "NYSE/NASDAQ"
    return f'<span class="pill {cls}">{lbl}</span>'

def _candle(df, ticker, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=ticker,
        increasing=dict(line=dict(color="#00ff9d", width=1), fillcolor="rgba(0,255,157,0.1)"),
        decreasing=dict(line=dict(color="#ff4757", width=1), fillcolor="rgba(255,71,87,0.1)"),
    ))
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Vol",
                         opacity=0.14, marker_color="#6366f1", yaxis="y2"))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#94a3b8")),
        xaxis_rangeslider_visible=False, template="plotly_dark", height=460,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.015)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
        yaxis=dict(title="Price", side="right", gridcolor="rgba(255,255,255,0.04)"),
        yaxis2=dict(title="Vol", overlaying="y", side="left", showgrid=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=36, b=0),
    )
    return fig

def _gauge(val, rng, title, color, height=230):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        title={"text": title, "font": {"size": 11, "color": "#64748b"}},
        gauge={
            "axis": {"range": rng, "tickcolor": "#475569",
                     "tickfont": {"size": 8, "color": "#475569"}},
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [rng[0], rng[0]+(rng[1]-rng[0])*0.33], "color": "rgba(255,71,87,0.07)"},
                {"range": [rng[0]+(rng[1]-rng[0])*0.33, rng[0]+(rng[1]-rng[0])*0.67], "color": "rgba(255,165,0,0.05)"},
                {"range": [rng[0]+(rng[1]-rng[0])*0.67, rng[1]], "color": "rgba(0,255,157,0.07)"},
            ],
        },
        number={"font": {"size": 18, "color": color}, "valueformat": ".3f"},
    ))
    fig.update_layout(template="plotly_dark", height=height,
                      paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=16, r=16, t=44, b=8))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📡 Stock Oracle")
    st.markdown('<p class="section-title">Multi-Market</p>', unsafe_allow_html=True)
    market   = st.radio("Market", ["🇺🇸 USA", "🇮🇳 India"],
                        horizontal=True, label_visibility="collapsed")
    tmap     = USA_TICKERS if "USA" in market else INDIA_TICKERS
    ticker   = st.selectbox("Ticker", list(tmap.keys()),
                             format_func=lambda t: f"{t} — {tmap[t]}")
    auto_ref = st.toggle("Auto-refresh 30s", value=False)
    inr_rate = get_usd_inr()
    if not is_indian(ticker):
        st.markdown(
            f'<div class="glass-sm" style="font-size:0.78rem;text-align:center;">'
            f'💱 1 USD = ₹{inr_rate:.2f}</div>',
            unsafe_allow_html=True,
        )

    # ── Social Pulse Gauge ────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-title">Social Pulse — Retail Sentiment</p>',
                unsafe_allow_html=True)
    _sp = get_social_sentiment(ticker)
    _sp_score = _sp["score"]
    _sp_color = ("#00e676" if _sp_score >= 0.25 else
                 "#ff4757" if _sp_score <= -0.25 else "#ffa500")
    _sp_fill  = int((_sp_score + 1) / 2 * 100)   # 0-100 for CSS width
    st.markdown(f"""
<div class="social-bar">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <span style="font-size:0.82rem;font-weight:700;color:{_sp_color};">{_sp["label"]}</span>
    <span style="font-size:0.72rem;color:#475569;">{_sp['sources']} posts · {_sp['via']}</span>
  </div>
  <div style="background:rgba(255,255,255,0.07);border-radius:999px;height:7px;margin-bottom:6px;">
    <div style="width:{_sp_fill}%;height:7px;border-radius:999px;
         background:linear-gradient(90deg,#ff4757 0%,#ffa500 50%,#00e676 100%);
         opacity:0.85;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:0.67rem;color:#475569;">
    <span>-1.0 Bearish</span>
    <span style="color:{_sp_color};font-weight:600;">{_sp_score:+.3f}</span>
    <span>+1.0 Bullish</span>
  </div>
  <div style="font-size:0.71rem;color:#64748b;margin-top:5px;font-style:italic;">
    "{_sp['theme']}"
  </div>
  <div style="font-size:0.68rem;color:#334155;margin-top:3px;">
    Confidence: {'▓'*int(_sp['confidence']*10)}{'░'*(10-int(_sp['confidence']*10))}
    {_sp['confidence']:.0%}
    {'🔄' if _sp.get('cached') else '✨'}
  </div>
</div>""", unsafe_allow_html=True)

    # ── Market Regime Indicator ───────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-title">Market Regime</p>', unsafe_allow_html=True)
    # Use cached regime if computed; placeholder before pre-compute block runs
    _reg_now = st.session_state.get("cached_regime", {})
    if _reg_now:
        _rc    = _reg_now.get("color", "#94a3b8")
        _rlbl  = _reg_now.get("label", "STABLE")
        _rdesc = _reg_now.get("description", "")
        _rconf = _reg_now.get("confidence", 0.5)
        st.markdown(
            f'<div class="regime-sidebar" style="background:rgba(255,255,255,0.04);'
            f'border:1px solid {_rc}33;">'
            f'<div style="color:{_rc};font-weight:800;font-size:1rem;">{_rlbl}</div>'
            f'<div style="font-size:0.7rem;color:#64748b;margin-top:3px;">'
            f'Confidence: {_rconf:.0%}</div>'
            f'<div style="font-size:0.72rem;color:#94a3b8;margin-top:6px;line-height:1.5;">'
            f'{_rdesc[:180]}{"…" if len(_rdesc) > 180 else ""}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="regime-sidebar" style="background:rgba(255,255,255,0.04);'
            'border:1px solid rgba(255,255,255,0.07);color:#475569;font-size:0.8rem;">'
            'Computing regime…</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown('<p class="section-title">News Headline</p>', unsafe_allow_html=True)
    headline    = st.text_area(
        "", height=68, label_visibility="collapsed",
        value="Fed signals rates to stay higher for longer amid sticky inflation",
    )
    analyze_btn = st.button("Analyze Headline", use_container_width=True)

    st.divider()
    st.markdown('<p class="section-title">Validation</p>', unsafe_allow_html=True)
    cv_c1, cv_c2 = st.columns([3, 1])
    with cv_c1: run_cv = st.button("Run 10-Fold CV", use_container_width=True)
    with cv_c2:
        st.markdown("🔄" if st.session_state.cv_running
                    else ("✅" if METRICS_PATH.exists() else "⬜"))
    if run_cv and not st.session_state.cv_running:
        st.session_state.cv_thread  = cross_validate_async()
        st.session_state.cv_running = True
    if st.session_state.cv_running:
        th = st.session_state.cv_thread
        if th and not th.is_alive():
            st.session_state.cv_running = False
            st.toast("CV complete!", icon="✅")

    # ── Smart Filter (Screener) ────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-title">Smart Screener Filters</p>',
                unsafe_allow_html=True)
    sf_verified  = st.toggle("Verified News Only (Veracity > 85%)", value=False)
    sf_safehaven = st.toggle("Geopolitical Safe-Havens", value=False,
                             help="Show Gold, Defense, and low-geo-risk tickers")
    sf_momentum  = st.toggle("High-Momentum Tech", value=False,
                             help="Show Tech sector with Oracle BUY signal")
    sf_region    = st.radio("Region", ["All", "USA", "India"],
                            horizontal=True, label_visibility="visible")
    sf_signal    = st.selectbox("Oracle Signal", ["All", "BUY", "HOLD", "SELL"])

    # ── Bull vs Bear Debate Panel ──────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-title">Bull vs Bear Debate</p>', unsafe_allow_html=True)
    debate_btn = st.button("Run Analyst Debate", use_container_width=True,
                           disabled=st.session_state.debate_running)

    if debate_btn and not st.session_state.debate_running:
        st.session_state.debate_running = True
        st.session_state.debate_ticker  = ticker
        st.rerun()

    if st.session_state.debate_running and st.session_state.debate_ticker == ticker:
        with st.spinner("Agents deliberating..."):
            _g, _gu = _geo(headline)
            _hdf    = _hist(ticker)
            _td     = get_technical_indicators(_hdf)
            _ms     = _macro_c()
            _ma     = get_macro_oracle_adjustment(ticker, _ms)
            _rc     = get_recommendation(_g, _td["score"], _gu, _td["uncertainty"],
                                          _ma["adjustment"], _ma["note"], ticker)
            debate  = run_debate(
                ticker=ticker, geo=_g, tech=_td["score"],
                macro=_ms, tech_ind=_td, headline=headline,
                oracle_rec=_rc, ticker_desc=ALL_TICKERS.get(ticker, ticker),
            )
            st.session_state.debate_result  = debate
            st.session_state.debate_running = False

    dr = st.session_state.debate_result
    if dr and st.session_state.debate_ticker == ticker:
        v      = dr.get("verdict", {})
        winner = v.get("winner", "DRAW")
        wc     = "#00ff9d" if winner == "BULL" else "#ff4757" if winner == "BEAR" else "#ffa502"
        st.markdown(
            f'<div class="glass-sm" style="text-align:center;font-size:0.8rem;">'
            f'Verdict: <b style="color:{wc}">{v.get("verdict","HOLD")}</b> — '
            f'{winner} wins ({v.get("confidence",0):.0%})'
            f'{"  🔄" if dr.get("cached") else ""}</div>',
            unsafe_allow_html=True,
        )
        db1, db2 = st.columns(2)
        bull_html = (dr.get("bull") or "").replace("\n", "<br>")
        bear_html = (dr.get("bear") or "").replace("\n", "<br>")
        with db1:
            st.markdown(
                f'<div class="glass-debate glass-bull">'
                f'<b style="color:#00ff9d;">🐂 BULL</b><br><br>{bull_html}</div>',
                unsafe_allow_html=True,
            )
        with db2:
            st.markdown(
                f'<div class="glass-debate glass-bear">'
                f'<b style="color:#ff4757;">🐻 BEAR</b><br><br>{bear_html}</div>',
                unsafe_allow_html=True,
            )
        if dr.get("error"):
            st.caption("Mock mode active (Gemini quota)")

        # Verdict details
        if v.get("summary"):
            st.markdown(
                f'<div class="glass-sm" style="font-size:0.78rem;color:#94a3b8;">'
                f'<b>Key risk:</b> {v.get("key_risk","")}<br>'
                f'<b>Catalyst:</b> {v.get("key_catalyst","")}</div>',
                unsafe_allow_html=True,
            )

    # ── Chatbot ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-title">AI Advisor (Gemini 2.5 Flash)</p>',
                unsafe_allow_html=True)
    chat_in = st.chat_input("Ask about portfolio, signals, FRED macro...")

    for msg in st.session_state.chat_history[-6:]:
        cls = "chat-user" if msg["role"] == "user" else "chat-bot"
        lbl = "**You:**"   if msg["role"] == "user" else "**Oracle:**"
        st.markdown(f'<div class="{cls}">{lbl} {msg["content"]}</div>',
                    unsafe_allow_html=True)

    if chat_in:
        st.session_state.chat_history.append({"role": "user", "content": chat_in})
        pf_d = _pf()
        geo_s, _ = _geo(headline)
        v_s   = veracity_score(headline)
        ms2   = _macro_c()
        hld   = "\n".join(
            f"  - {h['ticker']}: {h['shares']} @ ${h['avg_cost']:.2f}"
            for h in pf_d.get("holdings", [])
        ) or "  (empty)"
        ctx = (
            f"You are Stock Oracle AI. Ticker:{ticker}. {'India' if is_indian(ticker) else 'USA'}.\n"
            f"USD/INR:{inr_rate:.2f}. Headline:'{headline}' veracity={v_s:.3f} geo={geo_s:+.3f}.\n"
            f"FRED regime:{ms2.get('regime','?')} | CPI:{ms2.get('cpi_yoy','?')}% "
            f"| Fed:{ms2.get('fed_rate','?')}% | TechPressure:{ms2.get('tech_pressure',0):.0%}\n"
            f"Portfolio:\n{hld}\n{datetime.now():%Y-%m-%d %H:%M}\n3 sentences max."
        )
        full = ctx + "\n\n" + "\n".join(
            f"{'User' if m['role']=='user' else 'Oracle'}: {m['content']}"
            for m in st.session_state.chat_history[-6:]
        ) + "\nOracle:"

        api_key = _get_api_key()
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                reply = genai.GenerativeModel("gemini-2.5-flash").generate_content(full).text.strip()
            except Exception as e:
                reply = (f"[{e.__class__.__name__}] "
                         f"FRED:{ms2.get('regime','?')} CPI:{ms2.get('cpi_yoy','?')}% "
                         f"Fed:{ms2.get('fed_rate','?')}%")
        else:
            reply = (f"Mock. FRED:{ms2.get('regime','?')} "
                     f"CPI:{ms2.get('cpi_yoy','?')}% Fed:{ms2.get('fed_rate','?')}%")
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE shared values (once per page render)
# ══════════════════════════════════════════════════════════════════════════════
geo_score, geo_unc = _geo(headline)
hist_df   = _hist(ticker)
tech_ind  = get_technical_indicators(hist_df)
tech      = tech_ind["score"]
tech_unc  = tech_ind["uncertainty"]
macro_st  = _macro_c()
macro_adj = get_macro_oracle_adjustment(ticker, macro_st)
rec       = get_recommendation(
    geo_score, tech, geo_unc, tech_unc,
    macro_adj["adjustment"], macro_adj["note"], ticker,
)
v_score  = veracity_score(headline)
cur      = "₹" if is_indian(ticker) else "$"
inr_rate = get_usd_inr()

# Background-refresh social pulse for all watchlist tickers (daemon thread, non-blocking)
if "social_pulse_started" not in st.session_state:
    refresh_social_pulse(list(ALL_TICKERS.keys()))
    st.session_state["social_pulse_started"] = True

# Volatility squeeze check (computed from pre-loaded tech_ind)
_squeeze_detected = tech_ind.get("squeeze_detected", False)
_squeeze_warning  = tech_ind.get("squeeze_warning", "")

# Social pulse for current ticker (fast — from cache)
_social_now = get_social_sentiment(ticker)

# Market Regime Detection
_regime = detect_regime(
    hist_df, geo_score, v_score, tech_ind, macro_st,
    ticker=ticker, social_score=_social_now["score"],
)

# Monte Carlo (cached 1h)
_mc = run_monte_carlo_cached(
    ticker, hist_df,
    geo_score   = geo_score,
    tech_score  = tech,
    atr_pct     = tech_ind.get("atr14_pct"),
    n_paths     = 1_000,
    horizon     = 30,
)

# Persist regime to session state so sidebar can read it before first run
st.session_state["cached_regime"] = _regime

# ══════════════════════════════════════════════════════════════════════════════
# ANALYST DRILL-DOWN ROUTING
# If a stock has been selected from the screener, render its full analyst page
# and stop — skipping the normal tab layout entirely.
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("selected_stock"):
    _sel = st.session_state["selected_stock"]
    _scr_row = {}
    # Try to get the cached screener row for this ticker so analyst_view has
    # pre-computed Oracle/Geo/Veracity values without re-fetching.
    try:
        _scr_df_cached = build_screener_df(headline, macro_st,
                                           tickers=list(WATCHLIST.keys()),
                                           force=False)
        if not _scr_df_cached.empty and _sel in _scr_df_cached["Ticker"].values:
            _scr_row = _scr_df_cached[_scr_df_cached["Ticker"] == _sel].iloc[0].to_dict()
    except Exception:
        pass
    render_analyst_view(_sel, headline, macro_st, _scr_row)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_screener, tab_live, tab_hold, tab_oracle_t, tab_macro, tab_recap, tab_cv = st.tabs([
    "🌍 Screener", "📊 Live", "💼 Holdings", "🔮 Oracle", "🏛️ Macro / FRED",
    "📰 Market Recap", "📋 CV Metrics",
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 0 — GLOBAL SCREENER
# ────────────────────────────────────────────────────────────────────────────
with tab_screener:
    st.markdown("## 🌍 Ultimate Global Screener")
    st.markdown(
        '<p style="color:#64748b;font-size:0.83rem;margin-bottom:4px;">'
        '25-ticker universe · MAG7 + US Blue-Chips + NIFTY50 Top 10 · '
        'Oracle signals updated every 15 min · Hover column headers for definitions.</p>',
        unsafe_allow_html=True,
    )

    # ── Load / refresh screener data ──────────────────────────────────────────
    scr_col1, scr_col2 = st.columns([1, 6])
    with scr_col1:
        scr_refresh = st.button("🔄 Refresh Grid", use_container_width=True)
    with scr_col2:
        # Active filter chips
        active_chips = []
        if sf_verified:  active_chips.append("✓ Veracity > 85%")
        if sf_safehaven: active_chips.append("🛡 Safe-Havens")
        if sf_momentum:  active_chips.append("⚡ High-Momentum Tech")
        if sf_region != "All": active_chips.append(f"📍 {sf_region}")
        if sf_signal != "All": active_chips.append(f"🎯 {sf_signal}")
        if active_chips:
            chips_html = " ".join(f'<span class="filter-chip">{c}</span>'
                                  for c in active_chips)
            st.markdown(chips_html, unsafe_allow_html=True)

    with st.spinner("Building screener grid… (first load ~30s, then cached 15 min)"):
        scr_df = build_screener_df(
            headline, macro_st,
            tickers=list(WATCHLIST.keys()),
            force=scr_refresh,
        )

    if scr_df.empty:
        st.warning("Could not load screener data. Check network / yfinance access.")
    else:
        # ── Apply Smart Filters ───────────────────────────────────────────────
        fdf = scr_df.copy()

        if sf_verified:
            fdf = fdf[fdf["Veracity%"] > 85]

        if sf_safehaven:
            safe_sectors  = {"Commod", "Defense"}
            safe_tickers  = {"GLD", "ITA", "JNJ", "XOM"}
            safe_geo_mask = fdf["Geo Score"] >= 0
            fdf = fdf[
                fdf["Sector"].isin(safe_sectors) |
                fdf["Ticker"].isin(safe_tickers) |
                safe_geo_mask
            ]

        if sf_momentum:
            fdf = fdf[(fdf["Sector"] == "Tech") & (fdf["Oracle"] == "BUY")]

        if sf_region != "All":
            fdf = fdf[fdf["Region"] == sf_region]

        if sf_signal != "All":
            fdf = fdf[fdf["Oracle"] == sf_signal]

        # ── Summary stats bar ─────────────────────────────────────────────────
        n_buy   = int((fdf["Oracle"] == "BUY").sum())
        n_sell  = int((fdf["Oracle"] == "SELL").sum())
        n_hold  = int((fdf["Oracle"] == "HOLD").sum())
        avg_geo = fdf["Geo Score"].mean() if not fdf.empty else 0.0
        st.markdown(f"""
<div class="screener-summary">
  <div class="screener-stat">
    <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;">BUY Signals</div>
    <div style="font-size:1.4rem;font-weight:800;color:#00e676;">{n_buy}</div>
  </div>
  <div class="screener-stat">
    <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;">SELL Signals</div>
    <div style="font-size:1.4rem;font-weight:800;color:#ff4757;">{n_sell}</div>
  </div>
  <div class="screener-stat">
    <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;">HOLD Signals</div>
    <div style="font-size:1.4rem;font-weight:800;color:#ffa500;">{n_hold}</div>
  </div>
  <div class="screener-stat">
    <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;">Avg Geo Score</div>
    <div style="font-size:1.4rem;font-weight:800;
         color:{'#00e676' if avg_geo>=0 else '#ff4757'};">{avg_geo:+.3f}</div>
  </div>
</div>""", unsafe_allow_html=True)

        # ── Geopolitical Treemap ──────────────────────────────────────────────
        st.markdown("#### Global Market Mood — Geopolitical Heatmap")
        _tooltip_tree = (
            "Size = approximate market capitalisation. "
            "Colour = Gemini AI geopolitical sentiment score "
            "(green = bullish, red = bearish). Click to drill down."
        )
        st.caption(_tooltip_tree)
        fig_tree = build_treemap(fdf)
        st.plotly_chart(fig_tree, use_container_width=True)

        st.divider()

        # ── Column tooltips header ────────────────────────────────────────────
        DISPLAY_COLS = ["Ticker", "Name", "Region", "Sector", "Price",
                        "24h Chg%", "Oracle", "Geo Score", "Veracity%",
                        "Macro Risk", "ATR%"]

        # Render tooltip header row
        header_html = '<div style="display:grid;grid-template-columns:' \
                      '90px 130px 60px 70px 90px 70px 60px 80px 80px 80px 55px 1fr;' \
                      'gap:4px;padding:6px 14px;border-bottom:1px solid rgba(255,255,255,0.1);">'
        for col in DISPLAY_COLS:
            tip = COLUMN_TOOLTIPS.get(col, "")
            header_html += (
                f'<span class="tooltip-col" title="{tip}" '
                f'style="font-size:0.62rem;font-weight:700;text-transform:uppercase;'
                f'color:#475569;letter-spacing:.08em;">{col}</span>'
            )
        header_html += '<span style="font-size:0.62rem;font-weight:700;text-transform:uppercase;color:#475569;">AI Signal Reason</span>'
        header_html += '</div>'
        st.markdown(header_html, unsafe_allow_html=True)

        # ── Grid rows ─────────────────────────────────────────────────────────
        # Sort: SELL first (highest risk), then HOLD, then BUY; within each by |composite|
        sort_order = {"SELL": 0, "HOLD": 1, "BUY": 2}
        fdf_sorted = fdf.copy()
        fdf_sorted["_sort_sig"] = fdf_sorted["Oracle"].map(sort_order).fillna(1)
        fdf_sorted = fdf_sorted.sort_values(
            ["_sort_sig", "_composite"], ascending=[True, True]
        )

        for _, row in fdf_sorted.iterrows():
            chg      = row["_chg_raw"]
            sig      = row["Oracle"]
            geo      = row["Geo Score"]
            macro_r  = row["Macro Risk"]
            chg_cls  = "chg-pos" if chg >= 0 else "chg-neg"
            sig_cls  = f"sig-{sig.lower()}"
            mac_cls  = f"macro-{'high' if macro_r=='HIGH' else 'med' if macro_r=='MEDIUM' else 'low'}"
            geo_col  = "#00e676" if geo >= 0.2 else "#ff4757" if geo <= -0.2 else "#ffa500"
            ver_col  = "#00e676" if row["Veracity%"] >= 85 else "#ffa500" if row["Veracity%"] >= 60 else "#ff4757"
            region_flag = "🇺🇸" if row["Region"] == "USA" else "🇮🇳"
            atr_col  = "#ff4757" if row["ATR%"] > 3 else "#ffa500" if row["ATR%"] > 1.5 else "#00e676"

            _row_left, _row_btn = st.columns([11, 1])
            with _row_left:
                st.markdown(f"""
<div class="screener-row" style="display:grid;grid-template-columns:90px 130px 60px 70px 90px 70px 60px 80px 80px 80px 55px 1fr;gap:4px;padding:9px 14px;">
  <span style="font-weight:700;font-size:0.83rem;">{row['Ticker']}</span>
  <span style="color:#94a3b8;font-size:0.78rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{row['Name']}">{row['Name']}</span>
  <span style="font-size:0.75rem;color:#64748b;">{region_flag}</span>
  <span style="font-size:0.72rem;color:#64748b;">{row['Sector']}</span>
  <span style="font-size:0.82rem;font-weight:600;">{row['Price']}</span>
  <span class="{chg_cls}" style="font-size:0.8rem;">{chg:+.2f}%</span>
  <span class="{sig_cls}">{sig}</span>
  <span style="color:{geo_col};font-size:0.8rem;font-weight:600;">{geo:+.3f}</span>
  <span style="color:{ver_col};font-size:0.8rem;">{row['Veracity%']:.0f}%</span>
  <span class="{mac_cls}">{macro_r}</span>
  <span style="color:{atr_col};font-size:0.78rem;">{row['ATR%']:.2f}%</span>
  <span style="font-size:0.72rem;color:#64748b;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
        title="{row['Signal']}">{row['Signal'][:90]}{"…" if len(row['Signal'])>90 else ""}</span>
</div>""", unsafe_allow_html=True)
            with _row_btn:
                if st.button("→", key=f"drill_{row['Ticker']}", help=f"Open {row['Ticker']} analyst view"):
                    st.session_state["selected_stock"] = row["Ticker"]
                    st.rerun()

        st.divider()

        # ── Sector breakdown bar chart ────────────────────────────────────────
        sc1, sc2 = st.columns(2)
        with sc1:
            if not fdf.empty:
                sec_grp = fdf.groupby("Sector")["_composite"].mean().sort_values()
                s_colors = ["#00e676" if v >= 0.1 else "#ff4757" if v <= -0.1 else "#ffa500"
                            for v in sec_grp.values]
                fig_sec = go.Figure(go.Bar(
                    x=sec_grp.values, y=sec_grp.index,
                    orientation="h", marker_color=s_colors,
                    text=[f"{v:+.3f}" for v in sec_grp.values],
                    textposition="outside", textfont=dict(size=10, color="#94a3b8"),
                ))
                fig_sec.update_layout(
                    title="Oracle Composite by Sector",
                    template="plotly_dark", height=300,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.015)",
                    xaxis=dict(range=[-1.1, 1.1], gridcolor="rgba(255,255,255,0.04)",
                               zeroline=True, zerolinecolor="rgba(255,255,255,0.1)"),
                    margin=dict(l=0, r=60, t=44, b=0),
                )
                st.plotly_chart(fig_sec, use_container_width=True)

        with sc2:
            if not fdf.empty:
                sig_counts = fdf["Oracle"].value_counts()
                sig_colors_map = {"BUY": "#00e676", "HOLD": "#ffa500", "SELL": "#ff4757"}
                fig_sig = go.Figure(go.Pie(
                    labels=sig_counts.index.tolist(),
                    values=sig_counts.values.tolist(),
                    hole=0.55,
                    marker=dict(colors=[sig_colors_map.get(s, "#6366f1") for s in sig_counts.index]),
                    textfont=dict(size=11, color="white"),
                    hovertemplate="%{label}: %{value} tickers (%{percent})<extra></extra>",
                ))
                fig_sig.update_layout(
                    title="Signal Distribution",
                    template="plotly_dark", height=300,
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=44, b=0),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_sig, use_container_width=True)

        # ── Column glossary (beginner layer) ──────────────────────────────────
        with st.expander("📖 Column Glossary — What does each column mean?"):
            for col, tip in COLUMN_TOOLTIPS.items():
                st.markdown(
                    f'<div class="glass-sm" style="padding:8px 14px;margin-bottom:4px;">'
                    f'<b style="color:#a5b4fc;">{col}</b>'
                    f'<span style="color:#64748b;"> — </span>'
                    f'<span style="color:#94a3b8;font-size:0.85rem;">{tip}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Export ────────────────────────────────────────────────────────────
        export_cols = [c for c in DISPLAY_COLS if c in fdf.columns] + ["Signal"]
        csv_data = fdf[export_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Export Screener CSV",
            data=csv_data,
            file_name=f"oracle_screener_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            use_container_width=False,
        )


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — LIVE DASHBOARD
# ────────────────────────────────────────────────────────────────────────────
with tab_live:
    st.markdown(
        f'<h2 style="margin-bottom:2px">{ticker} {_pill(ticker)} '
        f'<span style="font-size:0.85rem;color:#475569">'
        f'{ALL_TICKERS.get(ticker,"")}</span></h2>',
        unsafe_allow_html=True,
    )
    rt_df = _rt(ticker)
    if rt_df.empty:
        st.info("Market closed — showing daily history.")
        rt_df = hist_df.tail(120)

    if not rt_df.empty:
        last  = rt_df.iloc[-1]
        prev  = rt_df.iloc[-2] if len(rt_df) > 1 else last
        delta = float(last["Close"]) - float(prev["Close"])
        pct   = delta / float(prev["Close"]) * 100
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Last", f"{cur}{float(last['Close']):,.2f}", f"{delta:+.2f} ({pct:+.2f}%)")
        c2.metric("Open", f"{cur}{float(last['Open']):,.2f}")
        c3.metric("High", f"{cur}{float(last['High']):,.2f}")
        c4.metric("Low",  f"{cur}{float(last['Low']):,.2f}")
        c5.metric("Vol",  f"{int(last['Volume']):,}")
        if not is_indian(ticker):
            c6.metric("INR", f"₹{convert_to_inr(float(last['Close'])):,.0f}")

    st.plotly_chart(_candle(rt_df, ticker, f"{ticker} — 1-Min Real-Time"),
                    use_container_width=True)

    # ── Volatility Squeeze Warning ─────────────────────────────────────────────
    if _squeeze_detected:
        st.markdown(
            f'<div class="squeeze-warn">'
            f'⚡ {_squeeze_warning}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        _atr_pct = tech_ind.get("atr_pctile")
        _bbw_pct = tech_ind.get("bbw_pctile")
        if _atr_pct is not None:
            _vol_label = ("HIGH" if _atr_pct >= 70 else "MODERATE" if _atr_pct >= 30 else "LOW")
            _vol_color = ("#ff4757" if _atr_pct >= 70 else "#ffa500" if _atr_pct >= 30 else "#00e676")
            st.markdown(
                f'<div style="font-size:0.75rem;color:#475569;margin-bottom:6px;">'
                f'Volatility: <span style="color:{_vol_color};font-weight:600;">{_vol_label}</span>'
                f' — ATR-14: {cur}{tech_ind.get("atr14","N/A")} ({tech_ind.get("atr14_pct","?"):.2f}% of price)'
                f' | ATR at <b>{_atr_pct:.0f}th</b> pctile of 30-day range'
                f' | BB Width at <b>{_bbw_pct:.0f}th</b> pctile'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── 24h Sentiment Heat Trend ───────────────────────────────────────────────
    st.markdown("#### 24-Hour Global Sentiment Heat Trend — USA & India")
    usa_scores, usa_lbls = [], []
    ind_scores, ind_lbls = [], []
    for t in list(USA_TICKERS.keys()):
        try:
            s = get_technical_indicators(_hist(t))["score"]
            blended = round(0.5 * s + 0.5 * geo_score, 3)
        except Exception:
            blended = 0.0
        usa_scores.append(blended)
        usa_lbls.append(f"{t}: {blended:+.2f}")
    for t in list(INDIA_TICKERS.keys()):
        try:
            s = get_technical_indicators(_hist(t))["score"]
            blended = round(0.5 * s + 0.5 * geo_score, 3)
        except Exception:
            blended = 0.0
        ind_scores.append(blended)
        ind_lbls.append(f"{t}: {blended:+.2f}")

    colorscale = [[0, "#ff4757"], [0.5, "#ffa502"], [1, "#00ff9d"]]
    fig_heat = go.Figure()
    fig_heat.add_trace(go.Bar(
        y=["🇺🇸 USA"] * len(usa_scores), x=usa_scores,
        orientation="h", name="USA",
        marker=dict(color=usa_scores, colorscale=colorscale, cmin=-1, cmax=1,
                    line=dict(width=0)),
        text=usa_lbls, textposition="inside",
        hovertemplate="%{text}<extra></extra>",
    ))
    fig_heat.add_trace(go.Bar(
        y=["🇮🇳 India"] * len(ind_scores), x=ind_scores,
        orientation="h", name="India",
        marker=dict(color=ind_scores, colorscale=colorscale, cmin=-1, cmax=1,
                    line=dict(width=0)),
        text=ind_lbls, textposition="inside",
        hovertemplate="%{text}<extra></extra>",
    ))
    fig_heat.update_layout(
        title=f"Heat Trend — {datetime.now():%H:%M:%S} | Tech(50%) + Geo(50%) blend",
        template="plotly_dark", height=230, barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
        xaxis=dict(range=[-1, 1], gridcolor="rgba(255,255,255,0.04)",
                   title="Blended Score"),
        showlegend=False, margin=dict(l=0, r=0, t=44, b=0),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Scores = 50% price technicals + 50% headline geo-sentiment | Cached 60s")

    # ── Sentiment vs Price Correlation Scatter ─────────────────────────────────
    with st.expander("🔗 Sentiment vs. Price Correlation (Pearson r)"):
        # Bootstrap from price history on first visit so chart isn't empty
        bootstrap_from_history(ticker, hist_df, geo_score)
        # Log the current analysis event with live price
        _live_px, _ = _fetch_live_price(ticker)
        if _live_px:
            log_event(ticker, headline, geo_score, _live_px)

        corr = get_pearson(ticker)
        corr_r = corr["r"]
        corr_n = corr["n"]

        if corr_r is not None and corr["data"]:
            sig_color = ("#00e676" if corr_r > 0.3 else
                         "#ff4757" if corr_r < -0.3 else "#ffa500")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Pearson r", f"{corr_r:+.4f}")
            sc2.metric("p-value",   f"{corr['p_value']:.4f}",
                       "Significant" if corr["significant"] else "Not significant")
            sc3.metric("Samples", corr_n, f"(max {100})")

            xs  = [d["geo"]       for d in corr["data"]]
            ys  = [d["price_ret"] for d in corr["data"]]
            hls = [d["headline"]  for d in corr["data"]]

            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(
                    color=xs, colorscale=[[0,"#ff4757"],[0.5,"#ffa502"],[1,"#00ff9d"]],
                    cmin=-1, cmax=1, size=9, opacity=0.8,
                    line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
                ),
                text=hls, hovertemplate="Geo: %{x:+.3f}<br>Ret: %{y:+.3f}%<br>%{text}<extra></extra>",
                name="Events",
            ))
            # Trend line
            if corr_n >= 5:
                import numpy as np
                m_coef = np.polyfit(xs, ys, 1)
                x_line = [min(xs), max(xs)]
                y_line = [m_coef[0]*x + m_coef[1] for x in x_line]
                fig_sc.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode="lines",
                    line=dict(color=sig_color, width=2, dash="dot"),
                    name=f"Trend (r={corr_r:+.3f})",
                ))
            fig_sc.update_layout(
                title=f"{ticker} — Gemini Sentiment Score vs. Price Return% | {corr['interpretation']}",
                template="plotly_dark", height=320,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                xaxis=dict(title="Geo Sentiment Score", gridcolor="rgba(255,255,255,0.04)",
                           zeroline=True, zerolinecolor="rgba(255,255,255,0.1)"),
                yaxis=dict(title="Price Return % (vs baseline)", gridcolor="rgba(255,255,255,0.04)",
                           zeroline=True, zerolinecolor="rgba(255,255,255,0.1)"),
                margin=dict(l=0, r=0, t=44, b=0),
            )
            st.plotly_chart(fig_sc, use_container_width=True)
            st.caption(
                f"Each point = one headline analysis. X = Gemini geo score, "
                f"Y = price return vs first logged price. "
                f"r={corr_r:+.4f} | {'p<0.05 ✓' if corr['significant'] else 'p≥0.05'} | "
                f"n={corr_n} samples"
            )
        else:
            st.info(
                "Analyzing headlines builds the correlation dataset. "
                f"Currently {corr_n} sample(s) — need ≥ 3. "
                "Analyze more headlines in the sidebar to populate this chart."
            )

    with st.expander("📈 5-Year Daily + SMA 20/50/200"):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hist_df.index, y=hist_df["Close"],
                                   mode="lines", name="Close",
                                   line=dict(color="#6366f1", width=1.2)))
        for w, c in [(20, "#ffa502"), (50, "#00ff9d"), (200, "#ff4757")]:
            fig2.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df["Close"].rolling(w).mean(),
                mode="lines", name=f"SMA{w}",
                line=dict(color=c, width=1, dash="dot"),
            ))
        fig2.update_layout(
            title=f"{ticker} — 5Y Daily",
            template="plotly_dark", height=340,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
            margin=dict(l=0, r=0, t=36, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            f"{len(hist_df):,} rows | {hist_df.index[0].date()} to "
            f"{hist_df.index[-1].date()} | min required: {MIN_SAMPLES:,}"
        )

    # ── Monte Carlo Probability Cone ───────────────────────────────────────────
    with st.expander("🎲 Monte Carlo Probability Cone (1,000 paths · 30-day horizon)"):
        if _mc.get("error"):
            st.warning(f"Simulation error: {_mc['error']}")
        else:
            s = _mc["stats"]
            # Stats row
            st.markdown(f"""
<div class="mc-stats">
  <div class="mc-stat">
    <div class="label">Prob Up (30d)</div>
    <div class="value" style="color:{'#00e676' if s['prob_up']>=0.5 else '#ff4757'}">
      {s['prob_up']:.1%}</div>
  </div>
  <div class="mc-stat">
    <div class="label">Expected Return</div>
    <div class="value" style="color:{'#00e676' if s['exp_return']>=0 else '#ff4757'}">
      {s['exp_return']:+.2f}%</div>
  </div>
  <div class="mc-stat">
    <div class="label">Annual Vol</div>
    <div class="value" style="color:#ffa500">{s['annual_vol']:.1f}%</div>
  </div>
  <div class="mc-stat">
    <div class="label">Bear (5th pctile)</div>
    <div class="value" style="color:#ff4757">{cur}{s['loss_5pct']:,.2f}</div>
  </div>
  <div class="mc-stat">
    <div class="label">Bull (95th pctile)</div>
    <div class="value" style="color:#00e676">{cur}{s['gain_95pct']:,.2f}</div>
  </div>
  <div class="mc-stat">
    <div class="label">Geo Drift Adj</div>
    <div class="value" style="color:{'#00e676' if s['geo_adj']>=0 else '#ff4757'}">
      {s['geo_adj']:+.5f}/day</div>
  </div>
</div>""", unsafe_allow_html=True)

            fig_cone = build_cone_figure(_mc["cone_data"], ticker, cur)
            st.plotly_chart(fig_cone, use_container_width=True)
            st.caption(
                f"GBM params: μ={s['mu']:+.6f}/day | σ={s['sigma']:.6f}/day | "
                f"Geo bias={s['geo_adj']:+.6f} | Tech bias={s['tech_adj']:+.6f} | "
                f"{'🔄 Cached' if _mc.get('cached') else '✨ Fresh simulation'}"
            )
            if _mc.get("warning"):
                st.warning(_mc["warning"])


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — HOLDINGS
# ────────────────────────────────────────────────────────────────────────────
with tab_hold:
    st.markdown("## 💼 My Holdings")
    pf_data  = _pf()
    holdings = pf_data.get("holdings", [])

    with st.expander("Add / Remove Holding"):
        # ── Fuzzy ticker search (outside form so it can react live) ───────────
        raw_query = st.text_input(
            "Search ticker or company name",
            placeholder="e.g. apple, reliance, QQQ, TCS",
            key="hold_raw_query",
        )
        resolved_sym = resolved_name = resolved_ccy = None
        if raw_query.strip():
            hits = ticker_search(raw_query.strip(), limit=6)
            if hits:
                options = [f"{h['symbol']} — {h['name']} ({h['exchange']}, {h['currency']})" for h in hits]
                chosen = st.selectbox("Select match", options, key="hold_pick")
                resolved_sym = hits[options.index(chosen)]["symbol"]
                resolved_name = hits[options.index(chosen)]["name"]
                resolved_ccy  = hits[options.index(chosen)]["currency"]
                st.markdown(
                    f'<span style="color:#00cc96;font-size:0.85rem;">✓ Resolved: '
                    f'<b>{resolved_sym}</b> — {resolved_name} ({resolved_ccy})</span>',
                    unsafe_allow_html=True,
                )
            else:
                # Try direct resolve as fallback
                res = ticker_resolve(raw_query.strip())
                if res:
                    resolved_sym, resolved_name, _, resolved_ccy = res
                    st.markdown(
                        f'<span style="color:#00cc96;font-size:0.85rem;">✓ Resolved: '
                        f'<b>{resolved_sym}</b> — {resolved_name} ({resolved_ccy})</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(f"No match for '{raw_query}'. Try the exact symbol (e.g. AAPL, RELIANCE.NS).")

        with st.form("add_form"):
            fa1, fa2, fa3 = st.columns(3)
            # Show resolved symbol or let user type directly
            default_sym = resolved_sym or raw_query.strip().upper()
            nt = fa1.text_input("Ticker", value=default_sym, help="Auto-filled from search above")
            ns = fa2.number_input("Shares", min_value=0.0, step=1.0, value=1.0)
            ccy_label = f"Avg Cost ({resolved_ccy or 'USD/INR'})"
            nc = fa3.number_input(ccy_label, min_value=0.01, step=0.01, value=100.0)
            use_live = st.checkbox("Auto-fill cost with live market price", value=True)
            if st.form_submit_button("Save"):
                final_sym = nt.strip().upper()
                if not final_sym:
                    st.error("Enter a ticker symbol.")
                else:
                    final_cost = nc
                    if use_live:
                        with st.spinner(f"Fetching live price for {final_sym}…"):
                            live_p, live_src = _fetch_live_price(final_sym)
                        if live_p is not None:
                            final_cost = live_p
                            ccy = "₹" if is_indian(final_sym) else "$"
                            st.success(f"Live price: {ccy}{live_p:,.2f} ({live_src})")
                        else:
                            st.warning("Could not fetch live price — using your entered cost.")
                    ex = [h for h in holdings if h["ticker"] == final_sym]
                    if ex:
                        ex[0].update({"shares": ns, "avg_cost": final_cost})
                    else:
                        holdings.append({"ticker": final_sym, "shares": ns, "avg_cost": final_cost})
                    _spf({"holdings": holdings})
                    _rt.clear()
                    # Invalidate price cache so new holding shows up immediately
                    st.session_state.pop("pf_prices", None)
                    st.session_state.pop("pf_prices_ts", None)
                    st.success(f"Saved {final_sym}")
                    st.rerun()
        opts = [h["ticker"] for h in holdings]
        if opts:
            del_t = st.selectbox("Remove", ["—"] + opts)
            if st.button("Remove") and del_t != "—":
                _spf({"holdings": [h for h in holdings if h["ticker"] != del_t]})
                st.rerun()

    if not holdings:
        st.markdown('<div class="glass">No holdings. Add positions above.</div>',
                    unsafe_allow_html=True)
    else:
        # ── Refresh Prices button ──────────────────────────────────────────────
        rb1, rb2, rb3 = st.columns([2, 1, 4])
        with rb1:
            refresh_clicked = st.button("🔄 Refresh Prices", use_container_width=True)
        with rb2:
            ts = st.session_state.get("pf_prices_ts", 0)
            age_s = int(time.time() - ts) if ts else None
            if age_s is not None:
                st.markdown(
                    f'<div style="font-size:0.72rem;color:#64748b;padding-top:10px;">'
                    f'Updated {age_s}s ago</div>',
                    unsafe_allow_html=True,
                )

        with st.spinner("Fetching live prices...") if refresh_clicked else st.empty():
            prices = _get_portfolio_prices(holdings, force=refresh_clicked)

        # Resolve prices with fallbacks
        resolved_prices = {}
        for h in holdings:
            tkr = h["ticker"]
            p, src = prices.get(tkr, (None, "unavailable"))
            if p is None:
                try:
                    df_fb = load_or_fetch(tkr)
                    p = float(df_fb["Close"].iloc[-1]) if not df_fb.empty else None
                    src = "daily close (offline)"
                except Exception:
                    p = None
            if p is None:
                p = h["avg_cost"]
                src = "avg cost (no data)"
            resolved_prices[tkr] = (p, src)

        # ── Prescriptive engine ────────────────────────────────────────────────
        tech_map_pf = {}
        for h in holdings:
            tkr = h["ticker"]
            try:
                tech_map_pf[tkr] = get_technical_indicators(_hist(tkr))
            except Exception:
                tech_map_pf[tkr] = {"score": 0.0, "rsi14": None, "macd": None,
                                     "sma_signal": 0.0, "uncertainty": 0.2}

        pf_analysis = get_portfolio_analysis(
            holdings, resolved_prices, geo_score, tech_map_pf, macro_st,
        )
        analyses  = pf_analysis["analyses"]
        total_val = pf_analysis["total_value"]

        # ── Summary banner ─────────────────────────────────────────────────────
        tc_sum = sum(h["shares"] * h["avg_cost"] for h in holdings)
        tv_sum = total_val
        tp_sum = tv_sum - tc_sum
        pp_sum = tp_sum / tc_sum * 100 if tc_sum else 0.0

        g1, g2, g3, g4 = st.columns(4)
        ccy_lbl = "₹" if all(is_indian(h["ticker"]) for h in holdings) else "$"
        g1.metric("Portfolio Value", f"{ccy_lbl}{tv_sum:,.2f}")
        g2.metric("Cost Basis",      f"{ccy_lbl}{tc_sum:,.2f}")
        g3.metric("Total P&L",
                  f"{'+'if tp_sum>=0 else ''}{ccy_lbl}{tp_sum:,.2f}",
                  f"{'+' if pp_sum>=0 else ''}{pp_sum:.2f}%")
        g4.metric("Portfolio Bias",  pf_analysis["portfolio_action"].split("—")[0].strip())
        st.divider()

        # ── Action-First Holding Cards ─────────────────────────────────────────
        st.markdown("### Action Dashboard")

        BADGE_MAP = {
            "INVEST MORE":  ("badge-invest", "INVEST MORE"),
            "HOLD":         ("badge-hold",   "HOLD"),
            "SELL":         ("badge-sell",   "SELL"),
            "STOP-LOSS":    ("badge-stop",   "STOP-LOSS ⚠"),
            "PARTIAL SELL": ("badge-partial","PARTIAL SELL"),
            "TAKE PROFIT":  ("badge-profit", "TAKE PROFIT"),
        }
        # Sort by priority: CRITICAL first, then HIGH, MEDIUM, LOW
        PRIORITY_ORD = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        analyses_sorted = sorted(analyses, key=lambda a: PRIORITY_ORD.get(a["priority"], 9))

        for a in analyses_sorted:
            tkr   = a["ticker"]
            ccy2  = a["ccy"]
            badge_cls, badge_lbl = BADGE_MAP.get(a["action"], ("badge-hold", a["action"]))
            pnl_color = "#00e676" if a["pnl"] >= 0 else "#ff4757"
            _, src_lbl = resolved_prices.get(tkr, (None, ""))

            # Compute volatility sparkline for this holding
            _spark_df  = tech_map_pf.get(tkr)
            try:
                _spark_vals = get_volatility_sparkline(_hist(tkr), window=24)
            except Exception:
                _spark_vals = []

            # Get social pulse for this holding (from cache, instant)
            _sp_h = get_social_sentiment(tkr)

            st.markdown(f"""
<div class="holding-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
    <div>
      <span style="font-size:1.1rem;font-weight:700;">{tkr}</span>
      {_pill(tkr)}
      <span class="badge {badge_cls}" style="margin-left:10px;">{badge_lbl}</span>
    </div>
    <div style="text-align:right;font-size:0.8rem;color:#64748b;">{src_lbl}</div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px;">
    <div><div class="section-title">Live Price</div>
         <div style="font-size:1.05rem;font-weight:600;">{ccy2}{a['price']:,.2f}</div></div>
    <div><div class="section-title">Avg Cost</div>
         <div style="font-size:1.05rem;">{ccy2}{a['avg_cost']:,.2f}</div></div>
    <div><div class="section-title">Mkt Value</div>
         <div style="font-size:1.05rem;">{ccy2}{a['mkt_val']:,.2f}</div></div>
    <div><div class="section-title">P&amp;L</div>
         <div style="font-size:1.05rem;font-weight:700;color:{pnl_color};">
           {'+'if a['pnl']>=0 else ''}{ccy2}{a['pnl']:,.2f}
           &nbsp;({'+' if a['pct_pnl']>=0 else ''}{a['pct_pnl']:.2f}%)</div></div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px;
       font-size:0.78rem;color:#64748b;">
    <div>Shares: <b style="color:#94a3b8;">{a['shares']:g}</b></div>
    <div>Weight: <b style="color:#94a3b8;">{a['weight']:.1%}</b></div>
    <div>Stop-Loss: <b style="color:#ff6b6b;">{ccy2}{a['stop_loss']:,.2f}</b></div>
    <div>Take-Profit: <b style="color:#00e5ff;">{ccy2}{a['take_profit']:,.2f}</b></div>
  </div>
  <div style="margin-top:8px;font-size:0.75rem;color:#475569;">
    Geo: <b style="color:{'#00e676' if a['geo_score']>0.2 else '#ff4757' if a['geo_score']<-0.2 else '#ffa500'};">{a['geo_score']:+.3f}</b>
    &nbsp;|&nbsp;Tech: <b style="color:{'#00e676' if a['tech_score']>0.2 else '#ff4757' if a['tech_score']<-0.2 else '#ffa500'};">{a['tech_score']:+.3f}</b>
    &nbsp;|&nbsp;RSI-14: <b>{a.get('rsi14') or 'N/A'}</b>
    &nbsp;|&nbsp;Regime: <b>{a['regime']}</b>
    &nbsp;|&nbsp;Social: <b style="color:{'#00e676' if _sp_h['score']>0.1 else '#ff4757' if _sp_h['score']<-0.1 else '#ffa500'};">{_sp_h['score']:+.2f} {_sp_h['label']}</b>
  </div>
</div>""", unsafe_allow_html=True)

            # ── Volatility Sparkline ───────────────────────────────────────────
            if _spark_vals:
                _spark_color = "#ff4757" if max(_spark_vals) > 3.0 else "#ffa502" if max(_spark_vals) > 1.5 else "#00e676"
                _spark_max   = max(_spark_vals)
                _spark_avg   = sum(_spark_vals) / len(_spark_vals)
                fig_spark = go.Figure()
                fig_spark.add_trace(go.Scatter(
                    y=_spark_vals,
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color=_spark_color, width=1.5),
                    fillcolor=f"rgba({'255,71,87' if _spark_color=='#ff4757' else '255,165,2' if _spark_color=='#ffa502' else '0,230,118'},0.10)",
                    hovertemplate="Day -%{x}<br>Range: %{y:.2f}%<extra></extra>",
                    showlegend=False,
                ))
                fig_spark.update_layout(
                    height=70, margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    annotations=[dict(
                        text=f"24-Day Volatility Trend | Avg: {_spark_avg:.2f}% | Peak: {_spark_max:.2f}%",
                        xref="paper", yref="paper", x=0, y=1,
                        showarrow=False, font=dict(size=9, color="#475569"),
                        align="left",
                    )],
                )
                st.plotly_chart(fig_spark, use_container_width=True, key=f"spark_{tkr}")

            with st.expander(f"🧠 AI Reasoning — {tkr}"):
                rc_key = f"rc_{tkr}_{a['action']}"
                if rc_key not in st.session_state:
                    with st.spinner("Generating reasoning card..."):
                        from ticker_db import resolve as _tr
                        _res = _tr(tkr)
                        desc = _res[1] if _res else tkr
                        card = generate_reasoning_card(
                            tkr, a, v_score, macro_st, desc,
                        )
                        st.session_state[rc_key] = card
                st.markdown(
                    st.session_state[rc_key].replace("\n", "\n\n"),
                    unsafe_allow_html=False,
                )

        st.divider()

        # ── Suggested Rebalance ────────────────────────────────────────────────
        st.markdown("### 🔁 Suggested Rebalance")
        for tip in pf_analysis["rebalance_suggestions"]:
            st.markdown(
                f'<div class="rebalance-tip">💡 {tip}</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Charts ─────────────────────────────────────────────────────────────
        pc1, pc2 = st.columns(2)
        with pc1:
            bar_colors = ["#00e676" if a["pnl"] >= 0 else "#ff4757" for a in analyses]
            fig_pnl = go.Figure(go.Bar(
                x=[a["ticker"] for a in analyses],
                y=[a["pnl"]    for a in analyses],
                marker_color=bar_colors,
                text=[f"{'+' if a['pnl']>=0 else ''}{a['ccy']}{a['pnl']:,.0f}"
                      for a in analyses],
                textposition="auto",
            ))
            fig_pnl.update_layout(
                title="Unrealised P&L (live prices)",
                template="plotly_dark", height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.015)",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
        with pc2:
            fig_pie = go.Figure(go.Pie(
                labels=[a["ticker"]  for a in analyses],
                values=[a["mkt_val"] for a in analyses],
                hole=0.5,
                marker=dict(colors=["#6366f1","#00ff9d","#ff4757","#ffa502","#a78bfa"]),
                hovertemplate="%{label}<br>%{value:,.2f}<br>%{percent}<extra></extra>",
            ))
            fig_pie.update_layout(
                title="Portfolio Allocation (live market value)",
                template="plotly_dark", height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_pie, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — ORACLE (XAI + FRED macro)
# ────────────────────────────────────────────────────────────────────────────
with tab_oracle_t:
    st.markdown(f"## 🔮 Investment Oracle — {ticker} {_pill(ticker)}")

    regime  = macro_st.get("regime", "UNKNOWN")
    reg_cls = ("regime-hawk" if regime == "HAWKISH" else
               "regime-dove" if regime == "DOVISH" else "regime-neutral")
    st.markdown(
        f'<span class="{reg_cls}">🏛️ {regime}</span> &nbsp;'
        f'<span style="color:#64748b;font-size:0.8rem;">'
        f'Fed {macro_st.get("fed_rate","?")}% | CPI {macro_st.get("cpi_yoy","?")}% | '
        f'Macro adj: {macro_adj["adjustment"]:+.3f} | '
        f'{macro_adj["note"][:65]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    sig_cls = f"signal-{rec['signal'].lower()}"
    ci_lo, ci_hi = rec["ci_lower"], rec["ci_upper"]
    mid_pct  = (rec["composite"] + 1) / 2 * 100
    ci_plo   = max(0,  (ci_lo + 1) / 2 * 100)
    ci_phi   = min(100,(ci_hi + 1) / 2 * 100)

    st.markdown(f"""
<div class="glass" style="text-align:center;padding:26px 30px;">
  <span class="{sig_cls}">{rec['signal']}</span>
  <div style="margin-top:8px;color:#94a3b8;font-size:0.82rem;">
    Composite: <b style="color:#e2e8f0">{rec['composite']:+.3f}</b>
    (raw {rec['raw_composite']:+.3f} + macro {rec['macro_adj']:+.3f})
    &nbsp;|&nbsp; Confidence: <b style="color:#e2e8f0">{rec['confidence']:.1%}</b>
    &nbsp;|&nbsp; 95% CI: <b style="color:#e2e8f0">[{ci_lo:+.3f}, {ci_hi:+.3f}]</b>
  </div>
  <div style="margin-top:14px;">
    <div class="ci-track">
      <div class="ci-fill" style="left:{ci_plo:.1f}%;width:{ci_phi-ci_plo:.1f}%;
           background:linear-gradient(90deg,rgba(99,102,241,0.22),rgba(99,102,241,0.5));"></div>
      <div class="ci-dot" style="left:{mid_pct:.1f}%;background:{rec['color']};"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.68rem;
         color:#475569;margin-top:4px;">
      <span>-1.0 Bearish</span><span>0.0</span><span>+1.0 Bullish</span>
    </div>
  </div>
  <p style="margin-top:12px;color:#94a3b8;font-size:0.8rem;font-style:italic;">
    {rec['rationale']}</p>
</div>""", unsafe_allow_html=True)

    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        lbl = "REAL" if v_score >= 0.5 else "FAKE"
        clr = "#00ff9d" if v_score >= 0.5 else "#ff4757"
        st.plotly_chart(_gauge(round(v_score*100,1), [0,100], f"Veracity — {lbl}", clr),
                        use_container_width=True)
        st.caption(f"NB:{v_score:.4f} | {HARDWARE_INFO['backend']}")
    with gc2:
        gl = "Bullish" if geo_score > 0.2 else "Bearish" if geo_score < -0.2 else "Neutral"
        gc = "#00ff9d" if geo_score>0.2 else "#ff4757" if geo_score<-0.2 else "#ffa502"
        st.plotly_chart(_gauge(round(geo_score,3), [-1,1], f"Geo — {gl}", gc),
                        use_container_width=True)
        st.caption(f"{'Gemini 2.5 Flash' if _get_api_key() else 'Keyword Mock'} | +-{geo_unc:.3f}")
    with gc3:
        tl = "Bullish" if tech > 0.2 else "Bearish" if tech < -0.2 else "Neutral"
        tc2 = "#00ff9d" if tech>0.2 else "#ff4757" if tech<-0.2 else "#ffa502"
        st.plotly_chart(_gauge(round(tech,3), [-1,1], f"Technical — {tl}", tc2),
                        use_container_width=True)
        st.caption(f"SMA/RSI/MACD/BB | +-{tech_unc:.3f}")

    st.markdown("#### Explainable AI — Factor Contributions (XAI)")
    xa1, xa2 = st.columns([3, 2])
    with xa1:
        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Geo x0.60", "Tech x0.40", "Macro adj", "Composite"],
            y=[rec["geo_contribution"], rec["tech_contribution"],
               rec["macro_contribution"], 0],
            text=[f"{v:+.3f}" for v in [rec["geo_contribution"],
                  rec["tech_contribution"], rec["macro_contribution"], 0]],
            textposition="outside",
            connector={"line": {"color": "rgba(255,255,255,0.07)"}},
            increasing={"marker": {"color": "#00ff9d"}},
            decreasing={"marker": {"color": "#ff4757"}},
            totals={"marker":  {"color": rec["color"]}},
        ))
        fig_wf.update_layout(
            title="Contributions: Geo 60% + Tech 40% + FRED Macro",
            template="plotly_dark", height=290,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
            yaxis=dict(range=[-1.2, 1.2], gridcolor="rgba(255,255,255,0.04)"),
            margin=dict(l=0, r=0, t=44, b=0),
        )
        st.plotly_chart(fig_wf, use_container_width=True)
    with xa2:
        rsi_d    = tech_ind.get("rsi14",   "N/A")
        sma20_d  = tech_ind.get("sma20",   "N/A")
        sma50_d  = tech_ind.get("sma50",   "N/A")
        macd_d   = tech_ind.get("macd",    "N/A")
        bb_u     = tech_ind.get("bb_upper","N/A")
        bb_l     = tech_ind.get("bb_lower","N/A")
        atr14_d  = tech_ind.get("atr14",   "N/A")
        atr_pct  = tech_ind.get("atr14_pct","N/A")
        bbw_d    = tech_ind.get("bb_width","N/A")
        atr_ptle = tech_ind.get("atr_pctile", "N/A")
        bbw_ptle = tech_ind.get("bbw_pctile", "N/A")
        squeeze  = tech_ind.get("squeeze_detected", False)
        sigma    = ((0.6 * geo_unc)**2 + (0.4 * tech_unc)**2) ** 0.5
        sq_html  = (
            '<br><span style="color:#ffd166;font-weight:700;">⚡ Volatility Squeeze Detected</span>'
            if squeeze else ""
        )
        st.markdown(f"""
<div class="glass" style="font-size:0.79rem;line-height:1.85;">
<span style="font-size:0.67rem;color:#475569;text-transform:uppercase;">Indicators</span><br>
RSI-14: <b>{rsi_d}</b> | SMA20: {cur}{sma20_d} | SMA50: {cur}{sma50_d}<br>
MACD: <b>{macd_d}</b> | BB: {cur}{bb_l} - {cur}{bb_u}<br>
<br>
<span style="font-size:0.67rem;color:#475569;text-transform:uppercase;">Volatility</span><br>
ATR-14: <b>{cur}{atr14_d}</b> ({atr_pct}% of price) | Pctile: <b>{atr_ptle}</b><br>
BB Width: <b>{bbw_d}</b> | Pctile: <b>{bbw_ptle}</b>{sq_html}<br>
<br>
<span style="font-size:0.67rem;color:#475569;text-transform:uppercase;">Oracle Formula</span><br>
f = 0.60*{geo_score:+.3f} + 0.40*{tech:+.3f} + {macro_adj['adjustment']:+.3f}<br>
&nbsp; = <b style="color:{rec['color']}">{rec['composite']:+.3f}</b><br>
<br>
<span style="font-size:0.67rem;color:#475569;text-transform:uppercase;">95% CI (error propagation)</span><br>
sigma = sqrt((0.6*{geo_unc:.3f})^2+(0.4*{tech_unc:.3f})^2) = {sigma:.4f}<br>
CI = [{ci_lo:+.3f}, {ci_hi:+.3f}] | width = {rec['ci_width']:.3f}
</div>""", unsafe_allow_html=True)

    # ── SHAP XAI — Why this signal? ───────────────────────────────────────────
    with st.expander("🔍 Why this signal? — SHAP Feature Attribution"):
        _sp_oracle = get_social_sentiment(ticker)
        _shap_vals = compute_shap_values(
            rec       = rec,
            tech_ind  = tech_ind,
            v_score   = v_score,
            social_score = _sp_oracle["score"],
            geo_unc   = geo_unc,
        )
        fig_shap = build_shap_figure(_shap_vals)
        st.plotly_chart(fig_shap, use_container_width=True)

        # Narrative summary
        top_pos = [(f, v) for f, v in zip(_shap_vals["features"], _shap_vals["values"]) if v > 0.001]
        top_neg = [(f, v) for f, v in zip(_shap_vals["features"], _shap_vals["values"]) if v < -0.001]
        top_pos.sort(key=lambda x: -x[1])
        top_neg.sort(key=lambda x: x[1])
        pos_txt = " · ".join(f"{f} ({v:+.3f})" for f, v in top_pos[:3]) or "none"
        neg_txt = " · ".join(f"{f} ({v:+.3f})" for f, v in top_neg[:3]) or "none"
        st.markdown(
            f'<div class="glass-sm" style="font-size:0.78rem;">'
            f'<b style="color:#00e676;">Bullish drivers:</b> {pos_txt}<br>'
            f'<b style="color:#ff4757;">Bearish drivers:</b> {neg_txt}<br>'
            f'<b>Explained: {_shap_vals["explained"]:+.4f}</b> | '
            f'Prediction: {_shap_vals["prediction"]:+.4f} | '
            f'Residual: {_shap_vals["residual"]:+.4f}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Regime card in Oracle tab ──────────────────────────────────────────────
    with st.expander(f"🌐 Market Regime — {_regime['label']}"):
        _rc2 = _regime["color"]
        st.markdown(
            f'<div class="glass" style="border-color:{_rc2}33;">'
            f'<div style="font-size:1.1rem;font-weight:800;color:{_rc2};">'
            f'{_regime["label"]}</div>'
            f'<div style="font-size:0.75rem;color:#64748b;margin:4px 0 10px;">'
            f'Confidence: {_regime["confidence"]:.0%}</div>'
            f'<div style="font-size:0.85rem;color:#cbd5e1;line-height:1.7;">'
            f'{_regime["description"]}</div>'
            f'<div style="margin-top:12px;font-size:0.72rem;color:#475569;">'
            f'Vol ratio: {_regime["factors"].get("vol_ratio","?")}× | '
            f'ATR pctile: {_regime["factors"].get("atr_pctile","?")} | '
            f'Veracity high: {"Yes" if _regime["factors"].get("veracity_high") else "No"}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.info(f'Analysed: "{headline}"')


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — MACRO / FRED
# ────────────────────────────────────────────────────────────────────────────
with tab_macro:
    st.markdown("## 🏛️ FRED Macro Intelligence Dashboard")
    ms = macro_st
    reg_cls2 = ("regime-hawk"    if ms.get("regime") == "HAWKISH" else
                "regime-dove"    if ms.get("regime") == "DOVISH"  else "regime-neutral")
    st.markdown(
        f'<span class="{reg_cls2}" style="font-size:1rem;padding:6px 20px;">'
        f'🏛️ {ms.get("regime","UNKNOWN")} REGIME</span>'
        f'<span style="margin-left:12px;color:#64748b;font-size:0.84rem;">'
        f'{ms.get("regime_note","")}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("CPI YoY", f"{ms.get('cpi_yoy','N/A')}%", delta_color="inverse")
    m2.metric("Fed Rate", f"{ms.get('fed_rate','N/A')}%",
              f"{'Up Rising' if ms.get('rate_rising') else 'Stable'}")
    m3.metric("Breakeven", f"{ms.get('be_inflation','N/A')}%")
    m4.metric("Unemployment", f"{ms.get('unemployment','N/A')}%")
    m5.metric("Tech Pressure", f"{ms.get('tech_pressure',0):.0%}",
              delta_color="inverse")
    st.divider()

    fc1, fc2 = st.columns(2)
    with fc1:
        cpi_dict = ms.get("cpi_series", {})
        if cpi_dict:
            cpi_s = pd.Series(cpi_dict)
            cpi_s.index = pd.to_datetime(list(cpi_s.index))
            fig_cpi = go.Figure()
            fig_cpi.add_trace(go.Scatter(
                x=cpi_s.index, y=cpi_s.values, mode="lines+markers",
                name="CPI", line=dict(color="#6366f1", width=2),
                marker=dict(size=4),
            ))
            fig_cpi.add_hline(
                y=float(cpi_s.iloc[-1]), line_dash="dot", line_color="#ffa502",
                annotation_text=f"Latest:{float(cpi_s.iloc[-1]):.1f}",
            )
            fig_cpi.update_layout(
                title="CPI — CPIAUCSL (FRED)", template="plotly_dark", height=290,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                yaxis_title="Index", margin=dict(l=0, r=0, t=44, b=0),
            )
            st.plotly_chart(fig_cpi, use_container_width=True)
    with fc2:
        fed_dict = ms.get("fed_series", {})
        if fed_dict:
            fed_s = pd.Series(fed_dict)
            fed_s.index = pd.to_datetime(list(fed_s.index))
            c_fed = "#ff4757" if ms.get("hawkish") else "#00ff9d"
            fig_fed = go.Figure()
            fig_fed.add_trace(go.Scatter(
                x=fed_s.index, y=fed_s.values, mode="lines+markers",
                name="Fed Rate", line=dict(color=c_fed, width=2.5),
                fill="tozeroy",
                fillcolor=f"rgba({'255,71,87' if ms.get('hawkish') else '0,255,157'},0.08)",
            ))
            fig_fed.add_hline(
                y=4.5, line_dash="dot", line_color="#475569",
                annotation_text="4.5% hawkish threshold",
                annotation_font_color="#475569",
            )
            fig_fed.update_layout(
                title="Federal Funds Rate — FEDFUNDS (FRED)",
                template="plotly_dark", height=290,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                yaxis_title="Rate %", margin=dict(l=0, r=0, t=44, b=0),
            )
            st.plotly_chart(fig_fed, use_container_width=True)

    st.markdown("#### Oracle Macro Adjustment — All USA Tickers")
    rows_m = []
    for t in list(USA_TICKERS.keys()):
        adj2 = get_macro_oracle_adjustment(t, ms)
        rows_m.append({
            "Ticker": t, "Regime": ms.get("regime","?"),
            "Macro Adj": f"{adj2['adjustment']:+.3f}",
            "Tech Pressure": f"{adj2['tech_pressure']:.0%}",
            "Note": adj2["note"][:72],
        })
    st.dataframe(pd.DataFrame(rows_m), use_container_width=True, hide_index=True)
    st.caption(
        "Macro adj is added after Geo*0.60 + Tech*0.40. "
        "QQQ/growth tickers penalised in HAWKISH regime (Fed>4.5% + CPI>3.5%). "
        "Defense tickers get a small boost. Current regime is DOVISH so adj=0."
    )


# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — MARKET RECAP (Agentic Daily Summary)
# ────────────────────────────────────────────────────────────────────────────
with tab_recap:
    st.markdown("## 📰 Daily Global Market Intelligence Brief")
    st.markdown(
        '<p style="color:#64748b;font-size:0.85rem;">Powered by Gemini 2.5 Flash '
        '— synthesises Oracle signals, FRED macro, and recent headlines into a '
        '300-word analyst-grade summary.</p>',
        unsafe_allow_html=True,
    )

    # Build ticker score dict for all watchlist tickers
    recap_scores: dict = {}
    for t in list(ALL_TICKERS.keys()):
        try:
            ti2  = get_technical_indicators(_hist(t))
            ma2  = get_macro_oracle_adjustment(t, macro_st)
            rec2 = get_recommendation(
                geo_score, ti2["score"], geo_unc, ti2["uncertainty"],
                ma2["adjustment"], ma2["note"], t,
            )
            recap_scores[t] = {
                "composite": rec2["composite"],
                "signal":    rec2["signal"],
                "desc":      ALL_TICKERS.get(t, t)[:40],
            }
        except Exception:
            recap_scores[t] = {"composite": 0.0, "signal": "HOLD", "desc": t}

    recap_col1, recap_col2 = st.columns([1, 2])
    with recap_col1:
        regen_btn = st.button("🔄 Generate Fresh Recap", use_container_width=True)
        st.markdown("")
        # Sentiment table
        for t, v in recap_scores.items():
            sig  = v["signal"]
            comp = v["composite"]
            s_c  = "#00e676" if sig == "BUY" else "#ff4757" if sig == "SELL" else "#ffa500"
            st.markdown(
                f'<div class="glass-sm" style="padding:8px 14px;margin-bottom:6px;'
                f'display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-size:0.82rem;font-weight:600;">{t}</span>'
                f'<span style="color:{s_c};font-weight:700;font-size:0.82rem;">'
                f'{sig} {comp:+.2f}</span></div>',
                unsafe_allow_html=True,
            )

    with recap_col2:
        with st.spinner("Loading market summary..."):
            recent_headlines = [headline] if headline.strip() else []
            recap_result = get_daily_recap(
                recap_scores, macro_st,
                headlines=recent_headlines,
                force=regen_btn,
            )

        reg_c = macro_st.get("regime_color", "#ffa502")
        st.markdown(
            f'<div style="display:flex;gap:12px;margin-bottom:14px;flex-wrap:wrap;">'
            f'<span style="color:{reg_c};font-weight:700;font-size:0.85rem;">'
            f'🏛️ {recap_result["regime"]}</span>'
            f'<span style="color:#64748b;font-size:0.8rem;">'
            f'{"🔄 Cached" if recap_result.get("cached") else "✨ Fresh"} | '
            f'{recap_result["date"]} | '
            f'{recap_result["headlines_used"]} headline(s) used</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="glass recap-body">{recap_result["summary"]}</div>',
            unsafe_allow_html=True,
        )
        if recap_result.get("error"):
            st.caption(f"Mock mode active ({recap_result['error'][:60]})")


# ────────────────────────────────────────────────────────────────────────────
# TAB 6 — CV METRICS
# ────────────────────────────────────────────────────────────────────────────
with tab_cv:
    st.markdown("## 📋 10-Fold CV — Multinomial Naive Bayes Veracity Filter")

    report = get_dataset_report(
        list(USA_TICKERS.keys())[:3] + list(INDIA_TICKERS.keys())[:2]
    )
    nb_ok = len(NB_CORPUS) >= 50

    cols_d = st.columns(1 + len(report[:4]))
    with cols_d[0]:
        st.markdown(
            f'<div class="glass-sm" style="text-align:center;">'
            f'<div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;">NB Corpus</div>'
            f'<div style="font-size:1.3rem;font-weight:700;">{"OK" if nb_ok else "WARN"} {len(NB_CORPUS)}</div>'
            f'<div style="font-size:0.68rem;color:#475569;">headlines (min 50)</div></div>',
            unsafe_allow_html=True,
        )
    for col, row in zip(cols_d[1:], report[:4]):
        pc = "pill-india" if row["market"] == "India" else "pill-usa"
        col.markdown(
            f'<div class="glass-sm" style="text-align:center;">'
            f'<div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;">{row["ticker"]}</div>'
            f'<div style="font-size:1.3rem;font-weight:700;">{"OK" if row["ok"] else "WARN"} {row["n"]:,}</div>'
            f'<div style="font-size:0.68rem;color:#475569;">rows (min {MIN_SAMPLES:,})</div>'
            f'<span class="pill {pc}">{row["market"]}</span></div>',
            unsafe_allow_html=True,
        )
    st.divider()

    if st.session_state.cv_running:
        st.info("CV running in background — refresh to see results.")
    elif not METRICS_PATH.exists():
        st.info("Click Run 10-Fold CV in the sidebar to start.")
    else:
        m  = json.loads(METRICS_PATH.read_text())
        mk = [k for k in m if isinstance(m[k], dict) and "mean" in m[k]]

        cols_m = st.columns(len(mk))
        for col, key in zip(cols_m, mk):
            v   = m[key]; pct = v["mean"] * 100
            cls = "pos" if pct >= 75 else "neu" if pct >= 60 else "neg"
            col.markdown(
                f'<div class="glass" style="text-align:center;">'
                f'<div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;">'
                f'{key.capitalize()}</div>'
                f'<div style="font-size:1.4rem;font-weight:700;" class="{cls}">'
                f'{pct:.1f}%</div>'
                f'<div style="font-size:0.7rem;color:#475569;">+- {v["std"]*100:.1f}%</div></div>',
                unsafe_allow_html=True,
            )

        hw = m.get("hardware", {})
        if hw:
            st.markdown(
                f'<div class="glass-sm" style="font-size:0.78rem;color:#64748b;">'
                f'Backend: {hw.get("backend","CPU")} | '
                f'GPU: {"Yes" if hw.get("gpu_available") else "No"} | '
                f'n_jobs=-1 | Corpus: {hw.get("corpus_size",0)} headlines | '
                f'{m.get("n_folds",10)}-fold stratified CV</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### Per-Fold Breakdown")
        for key in mk:
            v    = m[key]
            clrs = ["#00ff9d" if f >= v["mean"] else "#6366f1" for f in v["folds"]]
            fig_f = go.Figure()
            fig_f.add_trace(go.Bar(
                x=[f"F{i+1}" for i in range(len(v["folds"]))],
                y=v["folds"], marker_color=clrs,
                text=[f"{x:.3f}" for x in v["folds"]], textposition="auto",
            ))
            fig_f.add_hline(
                y=v["mean"], line_dash="dot", line_color="#ffa502",
                annotation_text=f"mean={v['mean']:.4f}",
                annotation_position="top right",
                annotation_font_color="#ffa502",
            )
            fig_f.add_hrect(
                y0=v["mean"] - v["std"], y1=v["mean"] + v["std"],
                fillcolor="rgba(255,165,2,0.05)", line_width=0,
            )
            fig_f.update_layout(
                title=f"{key.capitalize()}  mean={v['mean']:.4f}  std={v['std']:.4f}",
                template="plotly_dark", height=220,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                yaxis=dict(range=[0, 1.08], gridcolor="rgba(255,255,255,0.04)"),
                margin=dict(l=0, r=0, t=44, b=0),
            )
            st.plotly_chart(fig_f, use_container_width=True)

        st.caption(
            "Classifier: Multinomial NB | TF-IDF ngram(1,2) alpha=0.5 | "
            "10-fold Stratified CV | n_jobs=-1 (all cores)"
        )


# ── Auto-refresh ───────────────────────────────────────────────────────────────
if auto_ref:
    time.sleep(30)
    st.rerun()
