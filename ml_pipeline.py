"""
ML Pipeline - 360° Corporate Valuation
This script demonstrates the complete ML pipeline used in this project.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ── STEP 1: DATA COLLECTION ──────────────────────────────────
print("Step 1: Collecting data...")

ticker = "HCLTECH.NS"
stock = yf.Ticker(ticker)
prices = stock.history(period="2y")
info = stock.info

print(f"  Fetched {len(prices)} days of price data for {ticker}")

# ── STEP 2: FEATURE ENGINEERING ──────────────────────────────
print("\nStep 2: Engineering features...")

df = prices.copy()

# Technical indicators
df["return_1d"]      = df["Close"].pct_change()
df["return_5d"]      = df["Close"].pct_change(5)
df["return_20d"]     = df["Close"].pct_change(20)
df["volatility_20d"] = df["return_1d"].rolling(20).std()
df["volatility_60d"] = df["return_1d"].rolling(60).std()
df["ma_50"]          = df["Close"].rolling(50).mean()
df["ma_200"]         = df["Close"].rolling(200).mean()
df["ma_cross"]       = (df["ma_50"] > df["ma_200"]).astype(int)
df["volume_ratio"]   = df["Volume"] / df["Volume"].rolling(20).mean()

# RSI
delta = df["Close"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df["rsi"] = 100 - (100 / (1 + gain / loss))

# Fundamental features (from yfinance info)
df["pe_ratio"]     = info.get("trailingPE", 0)
df["roe"]          = info.get("returnOnEquity", 0)
df["debt_equity"]  = info.get("debtToEquity", 0)
df["profit_margin"]= info.get("profitMargins", 0)
df["rev_growth"]   = info.get("revenueGrowth", 0)
df["beta"]         = info.get("beta", 1)

df.dropna(inplace=True)
print(f"  Feature matrix shape: {df.shape}")

# ── STEP 3: TARGET VARIABLE ───────────────────────────────────
print("\nStep 3: Creating target labels...")

# Health label: 1 = Healthy, 0 = At-Risk
df["target"] = (
    (df["return_20d"] > 0) &
    (df["rsi"] > 40) &
    (df["rsi"] < 70)
).astype(int)

print(f"  Healthy days:  {(df['target']==1).sum()}")
print(f"  At-Risk days:  {(df['target']==0).sum()}")

# ── STEP 4: TRAIN/TEST SPLIT ──────────────────────────────────
feature_cols = [
    "return_1d", "return_5d", "return_20d",
    "volatility_20d", "volatility_60d",
    "ma_cross", "volume_ratio", "rsi",
    "pe_ratio", "roe", "debt_equity",
    "profit_margin", "rev_growth", "beta"
]

X = df[feature_cols].fillna(0)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nStep 4: Train/Test split")
print(f"  Train: {len(X_train)} samples")
print(f"  Test:  {len(X_test)} samples")

# ── STEP 5: MODEL 1 — HEALTH CLASSIFIER ──────────────────────
print("\nStep 5: Training Health Classifier (Random Forest)...")

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)

print(f"  Accuracy: {acc:.2%}")
print(classification_report(y_test, y_pred))

feat_imp = pd.Series(clf.feature_importances_, index=feature_cols)
print(f"  Top 5 features: {list(feat_imp.nlargest(5).index)}")

# ── STEP 6: MODEL 2 — VALUATION REGRESSOR ────────────────────
print("\nStep 6: Training Valuation Regressor (Gradient Boosting)...")

df["future_close"] = df["Close"].shift(-20)
df.dropna(subset=["future_close"], inplace=True)

X_reg = df[feature_cols].fillna(0)
y_reg = df["future_close"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42, shuffle=False
)

reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
reg.fit(X_tr, y_tr)

r2  = reg.score(X_te, y_te)
mae = np.mean(np.abs(reg.predict(X_te) - y_te))

print(f"  R2 Score: {r2:.4f}")
print(f"  MAE:      {mae:.2f}")

# ── STEP 7: MODEL 3 — ANOMALY DETECTOR ───────────────────────
print("\nStep 7: Training Anomaly Detector (Isolation Forest)...")

iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_reg.fillna(0))

preds          = iso.predict(X_reg.fillna(0))
anomaly_count  = (preds == -1).sum()
anomaly_dates  = df.index[preds == -1].strftime("%Y-%m-%d").tolist()

print(f"  Anomalies detected: {anomaly_count} days ({anomaly_count/len(X_reg):.1%})")
print(f"  Sample dates: {anomaly_dates[:5]}")

# ── STEP 8: FINAL INSIGHTS ────────────────────────────────────
print("\n" + "="*50)
print("PIPELINE COMPLETE — Final Insights")
print("="*50)

latest      = X_reg.iloc[-1:]
health_pred = clf.predict(scaler.transform(latest))[0]
price_pred  = reg.predict(latest)[0]
current     = df["Close"].iloc[-1]

print(f"  Company:            {info.get('shortName', ticker)}")
print(f"  Current Price:      ${current:.2f}")
print(f"  Predicted Price:    ${price_pred:.2f} (20 days ahead)")
print(f"  Health Status:      {'HEALTHY ✅' if health_pred == 1 else 'AT RISK ⚠️'}")
print(f"  Anomaly Periods:    {len(anomaly_dates)} detected")
print(f"  Classifier Acc:     {acc:.2%}")
print(f"  Regressor R2:       {r2:.4f}")