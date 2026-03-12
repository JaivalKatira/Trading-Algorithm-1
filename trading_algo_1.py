import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Trading Decision Dashboard", layout="wide")

st.title("📈 Probabilistic Trading Decision Dashboard")

# ─────────────────────────────
# SECTOR BASED STOCK DATABASE
# ─────────────────────────────

SECTORS = {

"Indices": {
"NIFTY 50":"^NSEI",
"BANK NIFTY":"^NSEBANK",
"NIFTY MIDCAP":"^CNX100",
"NIFTY NEXT 50":"^NSMIDCP"
},

"IT": {
"TCS":"TCS.NS",
"Infosys":"INFY.NS",
"Wipro":"WIPRO.NS",
"HCL Tech":"HCLTECH.NS",
"Tech Mahindra":"TECHM.NS"
},

"Banking": {
"HDFC Bank":"HDFCBANK.NS",
"ICICI Bank":"ICICIBANK.NS",
"Kotak Bank":"KOTAKBANK.NS",
"Axis Bank":"AXISBANK.NS",
"SBI":"SBIN.NS"
},

"FMCG": {
"HUL":"HINDUNILVR.NS",
"ITC":"ITC.NS",
"Nestle":"NESTLEIND.NS",
"Dabur":"DABUR.NS"
},

"Pharma": {
"Sun Pharma":"SUNPHARMA.NS",
"Dr Reddy":"DRREDDY.NS",
"Cipla":"CIPLA.NS",
"Divis Lab":"DIVISLAB.NS"
},

"Auto": {
"Maruti":"MARUTI.NS",
"Tata Motors":"TATAMOTORS.NS",
"M&M":"M&M.NS",
"Hero Moto":"HEROMOTOCO.NS"
},

"Energy": {
"Reliance":"RELIANCE.NS",
"ONGC":"ONGC.NS",
"BPCL":"BPCL.NS"
},

"Infra": {
"L&T":"LT.NS",
"Ultratech Cement":"ULTRACEMCO.NS",
"Adani Ports":"ADANIPORTS.NS"
},

"Metals": {
"Tata Steel":"TATASTEEL.NS",
"Hindalco":"HINDALCO.NS",
"JSW Steel":"JSWSTEEL.NS"
}

}

# ─────────────────────────────
# SELECTION UI
# ─────────────────────────────

sector = st.sidebar.selectbox("Select Sector", list(SECTORS.keys()))

stocks = SECTORS[sector]

stock_name = st.sidebar.selectbox("Select Stock", list(stocks.keys()))

ticker = stocks[stock_name]

st.write(f"Selected Ticker: **{ticker}**")

# ─────────────────────────────
# DOWNLOAD DATA
# ─────────────────────────────

df = yf.download(ticker, period="5y", auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.dropna()

# ─────────────────────────────
# INDICATORS
# ─────────────────────────────

df["SMA50"] = df["Close"].rolling(50).mean()

# RSI
delta = df["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["RSI"] = 100 - (100/(1+rs))

# Returns + direction
df["Return"] = df["Close"].pct_change()
df["Dir"] = np.where(df["Return"]>0,1,-1)

# streak logic
df["change"] = df["Dir"].ne(df["Dir"].shift()).cumsum()
df["streak"] = (df.groupby("change").cumcount()+1)*df["Dir"]

# ATR
tr = np.maximum(
df["High"]-df["Low"],
np.maximum(
abs(df["High"]-df["Close"].shift()),
abs(df["Low"]-df["Close"].shift())
)
)

df["ATR"] = tr.rolling(14).mean()
df["ATR_mean"] = df["ATR"].rolling(20).mean()

# momentum
df["momentum"] = df["Close"]/df["Close"].shift(10)

df = df.dropna()

# ─────────────────────────────
# MARKOV PROBABILITY
# ─────────────────────────────

df_prob = df.copy()
df_prob["next"] = df_prob["Dir"].shift(-1)

prob_map = {}

for s,grp in df_prob.groupby("streak"):

    if len(grp)>=5:
        prob_map[s]=(grp["next"]==1).sum()/len(grp)

# ─────────────────────────────
# LATEST DATA
# ─────────────────────────────

latest = df.iloc[-1]

st.subheader("Latest Market Stats")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Close", round(latest["Close"],2))
col2.metric("RSI", round(latest["RSI"],2))
col3.metric("Momentum", round(latest["momentum"],2))
col4.metric("ATR", round(latest["ATR"],2))

# ─────────────────────────────
# SIGNAL LOGIC
# ─────────────────────────────

trend_ok = latest["Close"] > latest["SMA50"]
rsi_ok = 35 < latest["RSI"] < 70
momentum_ok = latest["momentum"] > 1.02
atr_expand = latest["ATR"] > latest["ATR_mean"]

streak = latest["streak"]

p_up = prob_map.get(streak,0.5)

st.subheader("📊 Trade Recommendation")

if trend_ok and rsi_ok and momentum_ok and atr_expand and p_up>=0.5:

    st.success("✅ LONG TRADE")

    st.write("Probability of Up Move:", round(p_up*100,2), "%")

elif not trend_ok and p_up<0.5:

    st.error("🔻 SHORT TRADE")

    st.write("Probability of Down Move:", round((1-p_up)*100,2), "%")

else:

    st.warning("⚠️ NO TRADE")

    st.write("Market conditions not favorable.")

# ─────────────────────────────
# PRICE CHART
# ─────────────────────────────

st.subheader("Price Chart")

st.line_chart(df["Close"])