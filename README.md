# 360° Corporate Valuation
### AI-Powered Investment Intelligence Platform

A full-stack machine learning project that performs comprehensive financial health checks on any publicly traded company and generates professional investment reports using AI.

---

## What It Does

Enter any stock ticker (AAPL, TCS.NS, RELIANCE.NS) and get:

- **Health Score** — ML-based company health rating out of 10
- **BUY / HOLD / SELL** recommendation
- **Technical Analysis** — RSI, Moving Averages, Volatility, Volume
- **Fundamental Analysis** — P/E, ROE, Debt/Equity, Margins, Growth
- **Anomaly Detection** — Flags unusual trading periods
- **AI Report** — GPT-4 generated professional investment narrative

---

## ML Pipeline

| Phase | What Happens |
|---|---|
| Phase 1 | Data collection — yfinance, SEC EDGAR, NewsAPI, FRED |
| Phase 2 | Feature engineering — 14+ technical & fundamental features + NLP sentiment |
| Phase 3 | Model training — Random Forest, Gradient Boosting, Isolation Forest |
| Phase 4 | AI report generation — GPT-4 powered investment narrative |
| Phase 5 | Streamlit dashboard — Interactive web app |

---

## Tech Stack

- **ML** — Scikit-learn (Random Forest, Gradient Boosting, Isolation Forest)
- **Data** — yfinance, SEC EDGAR API, NewsAPI, FRED API
- **NLP** — VADER Sentiment Analysis
- **Frontend** — Streamlit + Plotly
- **AI** — OpenAI GPT-4o
- **Language** — Python

---

## Run Locally

```bash
git clone https://github.com/ishitasehgal218-ai/360-Corporate-Valuation/tree/main
cd 360-Corporate-Valuation
pip install -r requirements.txt
streamlit run app.py
```

Create a `.env` file:
---

## Live Demo

🔗 [Click here to open the app](https://360-corporate-valuation-9mqjficxcavgtjhbswoeu9.streamlit.app/)

---

*This project was built as part of an AI/ML research initiative. Reports are for informational purposes only and do not constitute financial advice.*