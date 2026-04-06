"""
Analyst Suite — Drill-Down View (View B)
render_analyst_view(ticker, headline, macro_st, headline_geo, v_score)

Triggered when st.session_state["selected_stock"] is set.
Renders:
  - Floating Back button
  - Large live price header
  - Left column: 1m chart, technicals, SHAP, Monte Carlo
  - Right column: Agentic chat with full stock context
  - Bull vs Bear debate (full width below)
"""

import time
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))


# ── Agentic chat with full stock context ────────────────────────────────────
_ANALYST_SYSTEM = """\
You are an elite equity analyst AI for {ticker} ({name}).
You have access to the following real-time intelligence:

PRICE DATA:
  Current Price : {cur}{price:,.2f}
  24h Change    : {chg:+.2f}%
  Region        : {region}

ORACLE SIGNAL: {signal} (composite {composite:+.3f})
  Geo Sentiment : {geo:+.3f} (Gemini-scored)
  Tech Score    : {tech:+.3f} (SMA/RSI/MACD/BB)
  RSI-14        : {rsi}
  MACD          : {macd}
  ATR-14%       : {atr_pct}%
  Squeeze       : {squeeze}

NEWS & VERACITY:
  Headline      : "{headline}"
  Veracity      : {veracity:.0%} (NB classifier — {veracity_label})
  Geo Score     : {geo:+.3f}

MACRO ENVIRONMENT:
  FRED Regime   : {regime} | Fed Rate: {fed_rate}% | CPI: {cpi_yoy}%
  Macro Adj     : {macro_adj:+.3f} | {macro_note}
  Tech Pressure : {tech_pressure:.0%}

SOCIAL PULSE:
  Retail Sentiment: {social_score:+.3f} ({social_label})
  Dominant Theme  : "{social_theme}"

PORTFOLIO CONTEXT:
  Stop-Loss level  : {cur}{stop_loss:,.2f} (-5% trailing)
  Take-Profit level: {cur}{take_profit:,.2f} (+15% from current)

Answer the user's question in 3-4 sentences. Be direct and specific to {ticker}.
Cite numbers. Use plain language — assume the user is an informed retail investor.
"""


def _build_chat_context(ticker: str, data: dict) -> str:
    """Build the full context string for the Gemini analyst chat."""
    return _ANALYST_SYSTEM.format(**data)


def _call_gemini_chat(prompt: str) -> str:
    """Call Gemini with rate-limit handling. Returns response text."""
    import os, re, google.generativeai as genai
    from pathlib import Path as P

    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
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
        return "API key not configured. Set GEMINI_API_KEY in .streamlit/secrets.toml."

    genai.configure(api_key=key)
    client = genai.GenerativeModel("gemini-2.5-flash")
    for attempt in range(3):
        try:
            return client.generate_content(prompt).text.strip()
        except Exception as exc:
            err = str(exc)
            if "429" in err or "quota" in err.lower():
                import re as _re
                m2 = _re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err)
                wait = int(m2.group(1)) + 1 if m2 else 20
                if attempt < 2:
                    time.sleep(wait)
                    continue
            return f"[{exc.__class__.__name__}] {str(exc)[:120]}"
    return "Gemini unavailable — please retry."


def render_analyst_view(
    selected_ticker: str,
    headline:        str,
    macro_st:        dict,
    screner_row:     dict | None = None,
) -> None:
    """
    Full analyst drill-down view for a single stock.
    Call from app.py when st.session_state['selected_stock'] is set.
    """
    from ingest  import (load_or_fetch, fetch_realtime, is_indian,
                          get_usd_inr, get_macro_oracle_adjustment)
    from analyze import (get_technical_indicators, score_geopolitical_risk_full,
                          score_geopolitical_risk_mock, get_recommendation,
                          get_volatility_sparkline, _get_api_key)
    from filter  import score as veracity_score
    from social_pulse import get_social_sentiment
    from simulation   import run_monte_carlo_cached, build_cone_figure
    from xai_shap     import compute_shap_values, build_shap_figure
    from agents       import run_debate
    from screener     import WATCHLIST

    meta   = WATCHLIST.get(selected_ticker, {})
    name   = meta.get("name", selected_ticker)
    region = meta.get("region", "USA")
    ccy    = "₹" if is_indian(selected_ticker) else "$"

    # ── Back button (floating top-left) ───────────────────────────────────────
    back_col, title_col = st.columns([1, 9])
    with back_col:
        if st.button("← Screener", use_container_width=True,
                     help="Return to Global Screener"):
            st.session_state.pop("selected_stock", None)
            st.session_state.pop(f"analyst_chat_{selected_ticker}", None)
            st.rerun()

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner(f"Loading {selected_ticker} intelligence…"):
        hist_df  = load_or_fetch(selected_ticker)
        rt_df    = fetch_realtime(selected_ticker)

        if hist_df is None or hist_df.empty:
            st.error(f"No price data available for {selected_ticker}.")
            return

        closes  = hist_df["Close"].astype(float)
        price   = float(closes.iloc[-1])
        prev_c  = float(closes.iloc[-2]) if len(closes) > 1 else price
        chg_pct = (price - prev_c) / prev_c * 100

        tech_ind = get_technical_indicators(hist_df)
        tech     = tech_ind.get("score", 0.0)
        tech_unc = tech_ind.get("uncertainty", 0.2)

        try:
            geo, geo_unc = score_geopolitical_risk_full(headline)
        except Exception:
            geo, geo_unc = score_geopolitical_risk_mock(headline)

        v_score  = veracity_score(headline)
        macro_adj_d = get_macro_oracle_adjustment(selected_ticker, macro_st)
        rec      = get_recommendation(geo, tech, geo_unc, tech_unc,
                                       macro_adj_d["adjustment"],
                                       macro_adj_d["note"], selected_ticker)
        sp       = get_social_sentiment(selected_ticker)

        stop_loss   = price * 0.95
        take_profit = price * 1.15

    # ── Header — large price ticker ───────────────────────────────────────────
    with title_col:
        sig_color = rec["color"]
        chg_color = "#00e676" if chg_pct >= 0 else "#ff4757"
        region_flag = "🇺🇸" if region == "USA" else "🇮🇳"
        st.markdown(f"""
<div style="display:flex;align-items:baseline;gap:18px;flex-wrap:wrap;margin-bottom:4px;">
  <span style="font-size:1.6rem;font-weight:800;">{region_flag} {selected_ticker}</span>
  <span style="font-size:1rem;color:#64748b;">{name}</span>
  <span style="font-size:2rem;font-weight:800;color:#e2e8f0;">{ccy}{price:,.2f}</span>
  <span style="font-size:1rem;font-weight:700;color:{chg_color};">{chg_pct:+.2f}%</span>
  <span style="font-size:1.3rem;font-weight:800;color:{sig_color};
    text-shadow:0 0 18px {sig_color}88;">{rec['signal']}</span>
  <span style="font-size:0.8rem;color:#475569;">composite {rec['composite']:+.3f}</span>
</div>""", unsafe_allow_html=True)

    st.divider()

    # ── Main two-column layout ────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2], gap="large")

    # ══ LEFT — Charts & Analytics ═══════════════════════════════════════════
    with left_col:

        # 1-min chart (or daily fallback)
        display_df = rt_df if (rt_df is not None and not rt_df.empty) else hist_df.tail(120)
        chart_title = (f"{selected_ticker} — 1-Min Real-Time"
                       if (rt_df is not None and not rt_df.empty)
                       else f"{selected_ticker} — Daily (Market Closed)")

        fig_price = go.Figure()
        if rt_df is not None and not rt_df.empty:
            fig_price.add_trace(go.Scatter(
                x=display_df.index, y=display_df["Close"],
                mode="lines", line=dict(color="#6366f1", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.07)",
                name="Price",
            ))
        else:
            fig_price.add_trace(go.Candlestick(
                x=display_df.index,
                open=display_df["Open"], high=display_df["High"],
                low=display_df["Low"],  close=display_df["Close"],
                increasing=dict(line=dict(color="#00e676", width=1),
                                fillcolor="rgba(0,230,118,0.1)"),
                decreasing=dict(line=dict(color="#ff4757", width=1),
                                fillcolor="rgba(255,71,87,0.1)"),
                name=selected_ticker,
            ))
        # Stop-loss / take-profit lines
        fig_price.add_hline(y=stop_loss,   line_dash="dot", line_color="#ff4757",
                            annotation_text=f"Stop {ccy}{stop_loss:.2f}",
                            annotation_font_color="#ff4757", annotation_font_size=9)
        fig_price.add_hline(y=take_profit, line_dash="dot", line_color="#00e5ff",
                            annotation_text=f"Target {ccy}{take_profit:.2f}",
                            annotation_font_color="#00e5ff", annotation_font_size=9)
        fig_price.update_layout(
            title=chart_title,
            template="plotly_dark", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.015)",
            xaxis_rangeslider_visible=False,
            yaxis=dict(side="right", gridcolor="rgba(255,255,255,0.04)"),
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # Volatility sparkline
        spark_vals = get_volatility_sparkline(hist_df, window=30)
        if spark_vals:
            spark_max   = max(spark_vals)
            spark_color = "#ff4757" if spark_max > 3 else "#ffa502" if spark_max > 1.5 else "#00e676"
            fig_spark = go.Figure(go.Scatter(
                y=spark_vals, mode="lines", fill="tozeroy",
                line=dict(color=spark_color, width=1.5),
                fillcolor=f"rgba({'255,71,87' if spark_color=='#ff4757' else '255,165,2' if spark_color=='#ffa502' else '0,230,118'},0.10)",
            ))
            fig_spark.update_layout(
                height=80, margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                title=dict(text=f"30-Day Volatility Trend (Daily Range%) | Peak {spark_max:.2f}%",
                           font=dict(size=9, color="#475569")),
            )
            st.plotly_chart(fig_spark, use_container_width=True, key=f"av_spark_{selected_ticker}")

        # Technical indicators panel
        rsi_v  = tech_ind.get("rsi14", "N/A")
        atr_v  = tech_ind.get("atr14_pct", "N/A")
        macd_v = tech_ind.get("macd", "N/A")
        bb_w   = tech_ind.get("bb_width", "N/A")
        sqz    = tech_ind.get("squeeze_detected", False)
        atr_p  = tech_ind.get("atr_pctile", "N/A")
        sma_s  = tech_ind.get("sma_signal", 0)
        sma_lbl= "Above SMA50 (Bullish)" if sma_s > 0 else "Below SMA50 (Bearish)"
        st.markdown(f"""
<div class="glass-sm" style="font-size:0.8rem;line-height:2.0;">
<span style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;">
Technical Indicators</span><br>
RSI-14: <b style="color:{'#ff4757' if isinstance(rsi_v,float) and rsi_v>70 else '#00e676' if isinstance(rsi_v,float) and rsi_v<30 else '#e2e8f0'}">{rsi_v}</b>
&nbsp;·&nbsp; MACD: <b>{macd_v}</b>
&nbsp;·&nbsp; BB Width: <b>{bb_w}</b><br>
ATR-14%: <b>{atr_v}</b> ({atr_p}th pctile)
&nbsp;·&nbsp; SMA Cross: <b>{sma_lbl}</b><br>
Composite: <b style="color:{rec['color']}">{rec['composite']:+.3f}</b>
&nbsp;·&nbsp; Confidence: <b>{rec['confidence']:.0%}</b>
{'&nbsp;·&nbsp;<span style="color:#ffd166;font-weight:700;">⚡ Squeeze</span>' if sqz else ""}
</div>""", unsafe_allow_html=True)

        # SHAP chart
        st.markdown("##### SHAP Feature Attribution")
        shap_vals = compute_shap_values(
            rec=rec, tech_ind=tech_ind,
            v_score=v_score, social_score=sp["score"], geo_unc=geo_unc,
        )
        fig_shap = build_shap_figure(shap_vals)
        fig_shap.update_layout(height=300, margin=dict(l=0, r=60, t=44, b=0))
        st.plotly_chart(fig_shap, use_container_width=True, key=f"av_shap_{selected_ticker}")

        # Monte Carlo cone
        mc = run_monte_carlo_cached(
            selected_ticker, hist_df, geo, tech,
            atr_pct=tech_ind.get("atr14_pct"), n_paths=1_000, horizon=30,
        )
        if not mc.get("error"):
            s = mc["stats"]
            st.markdown(f"""
<div style="display:flex;gap:12px;flex-wrap:wrap;margin:6px 0 10px;">
  <div class="glass-sm" style="padding:6px 14px;flex:1;text-align:center;">
    <div style="font-size:0.62rem;color:#475569;text-transform:uppercase;">Prob Up</div>
    <div style="font-weight:800;color:{'#00e676' if s['prob_up']>=.5 else '#ff4757'};">
      {s['prob_up']:.0%}</div>
  </div>
  <div class="glass-sm" style="padding:6px 14px;flex:1;text-align:center;">
    <div style="font-size:0.62rem;color:#475569;text-transform:uppercase;">E[Return]</div>
    <div style="font-weight:800;color:{'#00e676' if s['exp_return']>=0 else '#ff4757'};">
      {s['exp_return']:+.2f}%</div>
  </div>
  <div class="glass-sm" style="padding:6px 14px;flex:1;text-align:center;">
    <div style="font-size:0.62rem;color:#475569;text-transform:uppercase;">Annual Vol</div>
    <div style="font-weight:800;color:#ffa500;">{s['annual_vol']:.1f}%</div>
  </div>
  <div class="glass-sm" style="padding:6px 14px;flex:1;text-align:center;">
    <div style="font-size:0.62rem;color:#475569;text-transform:uppercase;">Bear (5%)</div>
    <div style="font-weight:800;color:#ff4757;">{ccy}{s['loss_5pct']:,.2f}</div>
  </div>
</div>""", unsafe_allow_html=True)
            fig_cone = build_cone_figure(mc["cone_data"], selected_ticker, ccy)
            fig_cone.update_layout(height=280)
            st.plotly_chart(fig_cone, use_container_width=True,
                            key=f"av_cone_{selected_ticker}")

    # ══ RIGHT — AI Advisor Chat ══════════════════════════════════════════════
    with right_col:
        st.markdown("#### AI Analyst Chat")
        st.markdown(
            f'<div class="glass-sm" style="font-size:0.75rem;color:#64748b;margin-bottom:8px;">'
            f'Full context loaded: {selected_ticker} · {rec["signal"]} · '
            f'Geo {geo:+.3f} · Veracity {v_score:.0%} · '
            f'{macro_st.get("regime","?")} regime · Social {sp["score"]:+.2f}</div>',
            unsafe_allow_html=True,
        )

        # Build data dict for context template
        ctx_data = dict(
            ticker=selected_ticker, name=name, cur=ccy, price=price,
            chg=chg_pct, region=region,
            signal=rec["signal"], composite=rec["composite"],
            geo=geo, tech=tech,
            rsi=tech_ind.get("rsi14", "N/A"),
            macd=tech_ind.get("macd", "N/A"),
            atr_pct=tech_ind.get("atr14_pct", "N/A"),
            squeeze="YES" if tech_ind.get("squeeze_detected") else "No",
            headline=headline,
            veracity=v_score,
            veracity_label="REAL" if v_score >= 0.5 else "FAKE",
            regime=macro_st.get("regime", "?"),
            fed_rate=macro_st.get("fed_rate", "?"),
            cpi_yoy=macro_st.get("cpi_yoy", "?"),
            macro_adj=macro_adj_d.get("adjustment", 0.0),
            macro_note=macro_adj_d.get("note", "")[:80],
            tech_pressure=macro_st.get("tech_pressure", 0.0),
            social_score=sp["score"],
            social_label=sp["label"],
            social_theme=sp.get("theme", ""),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        system_ctx = _build_chat_context(selected_ticker, ctx_data)

        # Chat history per ticker
        chat_key = f"analyst_chat_{selected_ticker}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        # Render history
        chat_container = st.container(height=420)
        with chat_container:
            for msg in st.session_state[chat_key]:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-user"><b>You:</b> {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="chat-bot"><b>Oracle:</b> {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
            if not st.session_state[chat_key]:
                st.markdown(
                    f'<div class="chat-bot"><b>Oracle:</b> '
                    f'I have full context on <b>{selected_ticker}</b>. '
                    f'Current signal is <b style="color:{sig_color}">{rec["signal"]}</b> '
                    f'({rec["composite"]:+.3f} composite). '
                    f'Ask me anything — price targets, risk factors, macro impact, '
                    f'or whether to buy now.</div>',
                    unsafe_allow_html=True,
                )

        # Chat input
        chat_in = st.chat_input(
            f"Ask about {selected_ticker}: price target, risk, macro impact…",
            key=f"chat_input_{selected_ticker}",
        )
        if chat_in:
            st.session_state[chat_key].append({"role": "user", "content": chat_in})
            full_prompt = (
                system_ctx + "\n\n"
                + "\n".join(
                    f"{'User' if m['role']=='user' else 'Oracle'}: {m['content']}"
                    for m in st.session_state[chat_key][-8:]
                )
                + "\nOracle:"
            )
            with st.spinner("Thinking…"):
                reply = _call_gemini_chat(full_prompt)
            st.session_state[chat_key].append({"role": "assistant", "content": reply})
            st.rerun()

        # Quick-action buttons
        st.markdown('<div style="margin-top:8px;">', unsafe_allow_html=True)
        qa_cols = st.columns(2)
        quick_questions = [
            f"What's the price target for {selected_ticker}?",
            f"What are the biggest risks right now?",
            f"Should I buy now or wait?",
            f"How does the macro affect {selected_ticker}?",
        ]
        for i, q in enumerate(quick_questions):
            with qa_cols[i % 2]:
                if st.button(q, key=f"qa_{selected_ticker}_{i}",
                             use_container_width=True):
                    st.session_state[chat_key].append({"role": "user", "content": q})
                    full_prompt = (
                        system_ctx + f"\nUser: {q}\nOracle:"
                    )
                    with st.spinner("Thinking…"):
                        reply = _call_gemini_chat(full_prompt)
                    st.session_state[chat_key].append({"role": "assistant", "content": reply})
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Social pulse card
        sp_color = "#00e676" if sp["score"] >= 0.2 else "#ff4757" if sp["score"] <= -0.2 else "#ffa500"
        sp_fill  = int((sp["score"] + 1) / 2 * 100)
        st.markdown(f"""
<div class="social-bar" style="margin-top:12px;">
  <div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">
    Social Pulse — Retail Sentiment</div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
    <span style="color:{sp_color};font-weight:700;font-size:0.85rem;">{sp['label']}</span>
    <span style="font-size:0.72rem;color:#475569;">{sp['score']:+.3f} · {sp['via']}</span>
  </div>
  <div style="background:rgba(255,255,255,0.07);border-radius:999px;height:6px;margin-bottom:5px;">
    <div style="width:{sp_fill}%;height:6px;border-radius:999px;
         background:linear-gradient(90deg,#ff4757 0%,#ffa500 50%,#00e676 100%);"></div>
  </div>
  <div style="font-size:0.71rem;color:#64748b;font-style:italic;">"{sp['theme']}"</div>
</div>""", unsafe_allow_html=True)

    # ── Bull vs Bear Debate (full width) ─────────────────────────────────────
    st.divider()
    st.markdown("#### Bull vs Bear Analyst Debate")
    debate_key = f"av_debate_{selected_ticker}"

    db_c1, db_c2 = st.columns([1, 6])
    with db_c1:
        run_debate_btn = st.button("Run Debate", key=f"debate_btn_{selected_ticker}",
                                   use_container_width=True)
    with db_c2:
        st.markdown(
            '<span style="font-size:0.78rem;color:#475569;">'
            'Two Gemini Flash agents debate the bull and bear case · Cached 24h</span>',
            unsafe_allow_html=True,
        )

    if run_debate_btn or debate_key in st.session_state:
        if run_debate_btn or debate_key not in st.session_state:
            with st.spinner("Agents deliberating…"):
                from screener import WATCHLIST
                dr = run_debate(
                    ticker=selected_ticker,
                    geo=geo, tech=tech,
                    macro=macro_st, tech_ind=tech_ind,
                    headline=headline,
                    oracle_rec=rec,
                    ticker_desc=WATCHLIST.get(selected_ticker, {}).get("name", selected_ticker),
                )
                st.session_state[debate_key] = dr

        dr = st.session_state[debate_key]
        v  = dr.get("verdict", {})
        winner   = v.get("winner", "DRAW")
        wc       = "#00e676" if winner == "BULL" else "#ff4757" if winner == "BEAR" else "#ffa502"

        st.markdown(
            f'<div class="glass-sm" style="text-align:center;font-size:0.85rem;margin-bottom:10px;">'
            f'Verdict: <b style="color:{wc}">{v.get("verdict","HOLD")}</b> — '
            f'{winner} wins ({v.get("confidence",0):.0%})'
            f'{"  🔄" if dr.get("cached") else "  ✨"}</div>',
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

        if v.get("summary"):
            st.markdown(
                f'<div class="glass-sm" style="font-size:0.8rem;color:#94a3b8;margin-top:8px;">'
                f'<b>Summary:</b> {v["summary"]}<br>'
                f'<b>Key Risk:</b> {v.get("key_risk","")}&nbsp;·&nbsp;'
                f'<b>Catalyst:</b> {v.get("key_catalyst","")}</div>',
                unsafe_allow_html=True,
            )
