import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import json, os, pickle
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="360° Corporate Valuation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── STYLING ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0a0e1a; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    background: linear-gradient(135deg, #e8d5a3, #c9a84c, #f0e68c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}
.hero-sub {
    color: #7a8399;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Score cards */
.score-card {
    background: linear-gradient(145deg, #131929, #1a2235);
    border: 1px solid #2a3550;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.score-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #c9a84c, #f0e68c);
}
.score-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: #e8d5a3;
    line-height: 1;
}
.score-label {
    color: #5a6680;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.4rem;
}
.score-sub {
    color: #7a8399;
    font-size: 0.85rem;
    margin-top: 0.3rem;
}

/* Section headers */
.section-head {
    font-family: 'Playfair Display', serif;
    color: #e8d5a3;
    font-size: 1.4rem;
    border-bottom: 1px solid #2a3550;
    padding-bottom: 0.6rem;
    margin-bottom: 1.2rem;
}

/* Verdict box */
.verdict-buy {
    background: linear-gradient(135deg, #0d2818, #1a3d26);
    border: 1px solid #2d6a42;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.verdict-hold {
    background: linear-gradient(135deg, #1a1a0d, #2d2d12);
    border: 1px solid #5a5a1a;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.verdict-sell {
    background: linear-gradient(135deg, #1a0d0d, #2d1212);
    border: 1px solid #6a2d2d;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.verdict-text {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 700;
}

/* Report section */
.report-section {
    background: #131929;
    border: 1px solid #2a3550;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.report-section h3 {
    color: #c9a84c;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
}
.report-section p { color: #a0aec0; line-height: 1.8; font-size: 0.95rem; }

/* Input styling */
.stTextInput input {
    background: #131929 !important;
    border: 1px solid #2a3550 !important;
    color: #e8d5a3 !important;
    border-radius: 10px !important;
    font-size: 1.1rem !important;
    padding: 0.8rem 1rem !important;
}
.stButton button {
    background: linear-gradient(135deg, #c9a84c, #e8d5a3) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2.5rem !important;
    width: 100% !important;
    letter-spacing: 1px !important;
}
.stButton button:hover { opacity: 0.9; transform: translateY(-1px); }

/* Tag pills */
.tag-green {
    display:inline-block; background:#0d2818; color:#4ade80;
    border:1px solid #2d6a42; border-radius:20px;
    padding:3px 14px; font-size:0.8rem; margin:3px;
}
.tag-red {
    display:inline-block; background:#1a0d0d; color:#f87171;
    border:1px solid #6a2d2d; border-radius:20px;
    padding:3px 14px; font-size:0.8rem; margin:3px;
}
.tag-yellow {
    display:inline-block; background:#1a1a0d; color:#fbbf24;
    border:1px solid #5a5a1a; border-radius:20px;
    padding:3px 14px; font-size:0.8rem; margin:3px;
}
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ───────────────────────────────────────────────────
def get_ticker_symbol(company_input):
    import requests
    company_input = company_input.strip()
    
    # Agar already valid ticker format hai (e.g. AAPL, HDFCBANK.NS)
    if company_input.upper() == company_input and len(company_input) <= 10:
        return company_input.upper()
    
    # Yahoo Finance search API
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": company_input, "quotesCount": 5, "newsCount": 0}
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=5)
        data = r.json()
        
        quotes = data.get("quotes", [])
        if quotes:
            # Pehla result lo
            symbol = quotes[0].get("symbol", company_input.upper())
            name = quotes[0].get("shortname", "")
            st.caption(f"Found: {name} → {symbol}")
            return symbol
    except:
        pass
    
    return company_input.upper()

@st.cache_data(ttl=3600)
def fetch_all_data(ticker, period="2y"):
    stock = yf.Ticker(ticker)
    prices = stock.history(period=period)
    info = stock.info
    return prices, info

def compute_features(prices):
    df = prices.copy()
    df["return_1d"] = df["Close"].pct_change()
    df["return_20d"] = df["Close"].pct_change(20)
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df["ma_50"] = df["Close"].rolling(50).mean()
    df["ma_200"] = df["Close"].rolling(200).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))
    df.dropna(inplace=True)
    return df

def compute_health_score(info, df):
    score = 5.0
    reasons = []

    pe = info.get("trailingPE")
    roe = info.get("returnOnEquity")
    de = info.get("debtToEquity")
    margins = info.get("profitMargins")
    rev_growth = info.get("revenueGrowth")
    rsi = df["rsi"].iloc[-1] if len(df) > 0 else 50

    if pe and 5 < pe < 30: score += 0.8; reasons.append(("P/E ratio healthy", "green"))
    elif pe and pe > 50: score -= 0.8; reasons.append(("P/E ratio too high", "red"))

    if roe and roe > 0.15: score += 1.0; reasons.append(("Strong ROE > 15%", "green"))
    elif roe and roe < 0: score -= 1.2; reasons.append(("Negative ROE", "red"))

    if de and de < 1: score += 0.7; reasons.append(("Low debt ratio", "green"))
    elif de and de > 3: score -= 0.8; reasons.append(("High debt", "red"))

    if margins and margins > 0.15: score += 0.8; reasons.append(("Healthy margins > 15%", "green"))
    elif margins and margins < 0: score -= 1.0; reasons.append(("Negative margins", "red"))

    if rev_growth and rev_growth > 0.1: score += 0.7; reasons.append(("Revenue growing > 10%", "green"))
    elif rev_growth and rev_growth < 0: score -= 0.5; reasons.append(("Revenue declining", "red"))

    if 40 < rsi < 65: score += 0.5; reasons.append(("RSI in healthy range", "green"))
    elif rsi > 75: score -= 0.3; reasons.append(("RSI overbought", "yellow"))
    elif rsi < 30: score -= 0.3; reasons.append(("RSI oversold", "yellow"))

    score = round(min(max(score, 1), 10), 1)
    return score, reasons

def get_verdict(health_score, info):
    pe = info.get("forwardPE") or info.get("trailingPE") or 999
    roe = info.get("returnOnEquity") or 0
    rev_growth = info.get("revenueGrowth") or 0

    buy_signals = sum([health_score >= 7, roe > 0.12, rev_growth > 0.05, pe < 35])
    sell_signals = sum([health_score < 4, roe < 0, rev_growth < -0.05, pe > 60])

    if buy_signals >= 3: return "BUY", "#4ade80", "verdict-buy"
    elif sell_signals >= 2: return "SELL", "#f87171", "verdict-sell"
    else: return "HOLD", "#fbbf24", "verdict-hold"

def generate_ai_report(info, df, health_score, verdict, api_key):
    client = OpenAI(api_key=api_key)
    ticker = info.get("symbol", "")
    name = info.get("shortName", ticker)
    rsi = round(df["rsi"].iloc[-1], 1)
    vol = round(df["volatility_20d"].iloc[-1] * 100, 2)
    ret_1m = round(df["return_20d"].iloc[-1] * 100, 2)
    current_price = round(df["Close"].iloc[-1], 2)

    prompt = f"""You are a senior Wall Street investment analyst. Generate a concise but insightful investment report for {name} ({ticker}).

Data:
- Current Price: {current_price}
- Health Score: {health_score}/10
- Verdict: {verdict}
- RSI: {rsi}
- 20-day Volatility: {vol}%
- 1-month Return: {ret_1m}%
- P/E: {info.get('trailingPE', 'N/A')}
- ROE: {info.get('returnOnEquity', 'N/A')}
- Revenue Growth: {info.get('revenueGrowth', 'N/A')}
- Profit Margins: {info.get('profitMargins', 'N/A')}
- Debt/Equity: {info.get('debtToEquity', 'N/A')}
- Sector: {info.get('sector', 'N/A')}

Write EXACTLY in this JSON format (no markdown, pure JSON):
{{
  "executive_summary": "3-4 sentence overview",
  "financial_analysis": "Analyze the ratios in 3-4 sentences",
  "technical_analysis": "Analyze RSI, volatility, momentum in 2-3 sentences",
  "bull_case": "2-3 sentence bull case",
  "bear_case": "2-3 sentence bear case",
  "recommendation": "Final 2-3 sentence recommendation with price target insight"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1500,
        messages=[
            {"role": "system", "content": "You are a professional investment analyst. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    text = response.choices[0].message.content.strip()
    # Clean JSON
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"): text = text[4:]
    return json.loads(text.strip())


# ─── CHARTS ────────────────────────────────────────────────────

def price_chart(df, name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        fill="tozeroy", fillcolor="rgba(201,168,76,0.08)",
        line=dict(color="#c9a84c", width=2),
        name="Price"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ma_50"],
        line=dict(color="#60a5fa", width=1, dash="dot"),
        name="MA 50"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ma_200"],
        line=dict(color="#f87171", width=1, dash="dot"),
        name="MA 200"
    ))
    fig.update_layout(
        title=f"{name} — Price History",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#7a8399", title_font_color="#e8d5a3",
        legend=dict(bgcolor="#131929", bordercolor="#2a3550"),
        xaxis=dict(gridcolor="#1a2235", showgrid=True),
        yaxis=dict(gridcolor="#1a2235", showgrid=True),
        height=380, margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def volume_chart(df):
    colors = ["#4ade80" if r >= 0 else "#f87171" for r in df["return_1d"]]
    fig = go.Figure(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors, opacity=0.7, name="Volume"
    ))
    fig.update_layout(
        title="Volume",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#7a8399", title_font_color="#e8d5a3",
        xaxis=dict(gridcolor="#1a2235"),
        yaxis=dict(gridcolor="#1a2235"),
        height=200, margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    return fig

def rsi_chart(df):
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,113,113,0.1)", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(74,222,128,0.1)", line_width=0)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi"],
        line=dict(color="#a78bfa", width=2), name="RSI"
    ))
    fig.add_hline(y=70, line_dash="dot", line_color="#f87171", line_width=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#4ade80", line_width=1)
    fig.update_layout(
        title="RSI (14)",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#7a8399", title_font_color="#e8d5a3",
        xaxis=dict(gridcolor="#1a2235"),
        yaxis=dict(gridcolor="#1a2235", range=[0, 100]),
        height=220, margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    return fig

def gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 10], "tickcolor": "#7a8399"},
            "bar": {"color": "#c9a84c"},
            "bgcolor": "#131929",
            "bordercolor": "#2a3550",
            "steps": [
                {"range": [0, 4], "color": "#2d1212"},
                {"range": [4, 7], "color": "#1a1a0d"},
                {"range": [7, 10], "color": "#0d2818"},
            ],
            "threshold": {
                "line": {"color": "#e8d5a3", "width": 3},
                "thickness": 0.8, "value": score
            }
        },
        number={"font": {"color": "#e8d5a3", "size": 48}, "suffix": "/10"}
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117", font_color="#7a8399",
        height=260, margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig


# ─── MAIN APP ──────────────────────────────────────────────────

# Hero
st.markdown('<div class="hero-title">360° Corporate Valuation</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-Powered Investment Intelligence</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Input row
col1, col2 = st.columns([3, 1.5])
with col1:
    company_input = st.text_input("Company", placeholder="Enter ticker or company name  (e.g. HDFC, AAPL, TCS, RELIANCE)", label_visibility="collapsed")
with col2:
    period = st.selectbox("", ["1y", "2y", "5y"], index=1, label_visibility="collapsed")

analyze_btn = st.button("GENERATE REPORT →")

st.markdown("---")

# ─── ANALYSIS ──────────────────────────────────────────────────
if analyze_btn and company_input:
    ticker = get_ticker_symbol(company_input)

    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            prices, info = fetch_all_data(ticker, period)
            if prices.empty:
                st.error(f"No data found for '{company_input}'. Try the exact ticker symbol.")
                st.stop()
            df = compute_features(prices)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

    name = info.get("shortName", ticker)
    current_price = round(df["Close"].iloc[-1], 2)
    currency = info.get("currency", "USD")
    health_score, reasons = compute_health_score(info, df)
    verdict, verdict_color, verdict_class = get_verdict(health_score, info)

    # ── TOP METRICS ROW ──
    st.markdown(f'<div class="section-head">{name} — {ticker}</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        (f"{currency} {current_price:,}", "Current Price", ""),
        (f"{health_score}/10", "Health Score", ""),
        (f"{round(df['rsi'].iloc[-1], 1)}", "RSI (14)", "Overbought >70"),
        (f"{round(df['volatility_20d'].iloc[-1]*100, 1)}%", "20d Volatility", ""),
        (f"{round(df['return_20d'].iloc[-1]*100, 1)}%", "1M Return", ""),
    ]
    for col, (val, label, sub) in zip([m1, m2, m3, m4, m5], metrics):
        with col:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-value">{val}</div>
                <div class="score-label">{label}</div>
                <div class="score-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── VERDICT + GAUGE ──
    v1, v2 = st.columns([1, 2])
    with v1:
        st.markdown(f"""
        <div class="{verdict_class}">
            <div style="color:#7a8399;font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem">AI Recommendation</div>
            <div class="verdict-text" style="color:{verdict_color}">{verdict}</div>
            <div style="color:#7a8399;font-size:0.85rem;margin-top:0.8rem">Based on ML + Fundamental Analysis</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        # Signal tags
        for reason, color in reasons[:6]:
            st.markdown(f'<span class="tag-{color}">{reason}</span>', unsafe_allow_html=True)

    with v2:
        st.plotly_chart(gauge_chart(health_score), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PRICE CHART ──
    st.markdown('<div class="section-head">Price & Technical Analysis</div>', unsafe_allow_html=True)
    st.plotly_chart(price_chart(df, name), use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(rsi_chart(df.tail(180)), use_container_width=True)
    with c2:
        st.plotly_chart(volume_chart(df.tail(180)), use_container_width=True)

    # ── FUNDAMENTALS TABLE ──
    st.markdown('<div class="section-head">Key Fundamentals</div>', unsafe_allow_html=True)
    fund_data = {
        "Metric": ["P/E Ratio", "Forward P/E", "Price/Book", "ROE", "ROA", "Debt/Equity", "Current Ratio", "Profit Margin", "Revenue Growth", "EV/EBITDA"],
        "Value": [
            info.get("trailingPE", "N/A"), info.get("forwardPE", "N/A"),
            info.get("priceToBook", "N/A"), info.get("returnOnEquity", "N/A"),
            info.get("returnOnAssets", "N/A"), info.get("debtToEquity", "N/A"),
            info.get("currentRatio", "N/A"), info.get("profitMargins", "N/A"),
            info.get("revenueGrowth", "N/A"), info.get("enterpriseToEbitda", "N/A"),
        ]
    }
    fund_df = pd.DataFrame(fund_data)
    fund_df["Value"] = fund_df["Value"].apply(
        lambda x: f"{round(float(x)*100, 2)}%" if isinstance(x, float) and abs(x) < 1 and x != "N/A"
        else (round(x, 2) if isinstance(x, float) else x)
    )
    f1, f2 = st.columns(2)
    with f1:
        st.dataframe(fund_df.iloc[:5], hide_index=True, use_container_width=True)
    with f2:
        st.dataframe(fund_df.iloc[5:], hide_index=True, use_container_width=True)

    # ── AI REPORT ──
    st.markdown('<div class="section-head">AI Investment Report</div>', unsafe_allow_html=True)

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY not found in .env file.")

    else:
        with st.spinner("Generating AI analysis..."):
            try:
                report = generate_ai_report(info, df, health_score, verdict, os.getenv("OPENAI_API_KEY"))

                sections = [
                    ("EXECUTIVE SUMMARY", report.get("executive_summary", "")),
                    ("FINANCIAL ANALYSIS", report.get("financial_analysis", "")),
                    ("TECHNICAL ANALYSIS", report.get("technical_analysis", "")),
                    ("BULL CASE 🟢", report.get("bull_case", "")),
                    ("BEAR CASE 🔴", report.get("bear_case", "")),
                    ("RECOMMENDATION", report.get("recommendation", "")),
                ]
                for title, content in sections:
                    st.markdown(f"""
                    <div class="report-section">
                        <h3>{title}</h3>
                        <p>{content}</p>
                    </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"AI report error: {e}")

    # Footer
    st.markdown("""
    <div style="text-align:center;color:#3a4560;font-size:0.75rem;margin-top:3rem;padding-top:1rem;border-top:1px solid #1a2235">
    This report is for informational purposes only and does not constitute financial advice.
    </div>""", unsafe_allow_html=True)

elif analyze_btn:
    st.warning("Please enter a company name or ticker symbol.")