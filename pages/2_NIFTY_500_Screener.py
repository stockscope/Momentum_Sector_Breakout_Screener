import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
from io import BytesIO

matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide")
st.title("📊 Momentum Sector Breakout Screener (NIFTY 500)")
st.markdown("Identifies breakout, retest, and pullback setups in top-performing sectors with momentum, volume, and risk filters.")

with st.expander("🧠 **Screening Criteria Used**", expanded=True):
    st.markdown("""
    - **Universe**: NIFTY 500 stocks
    - **Top Sectors**: Based on average **1-week return**
    - **Setup Detection**:
        - 📈 **Breakout**: Close ≥ 98% of 20-day or 52-week high
        - 🔁 **Retest**: Close between 0.97–1.03 × 50 EMA
        - ↩️ **Pullback**: Price dips to 20 EMA after recent high
    - **Momentum**:
        - RSI between 40–70
        - Positive ADR and price > 50 EMA
    - **Volume & Liquidity**:
        - Volume Spike > 1.5× 20D Avg Volume
        - RVOL > 1.2
    - **Risk Management**:
        - Reward:Risk > 1.5
        - Calculated Stop-Loss (3% below current price)
    - **Extras**:
        - Timezone-safe datetime
        - Excel export
        - Color-coded setups
    """)

with st.spinner("🔍 Screening in progress... please wait"):

    csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty500list.csv"
    df_nifty500 = pd.read_csv(csv_url)
    df_nifty500 = df_nifty500[~df_nifty500['Symbol'].str.contains("DUMMY", na=False)]
    df_nifty500['Ticker'] = df_nifty500['Symbol'].str.strip() + ".NS"

    tickers = df_nifty500['Ticker'].tolist()
    sector_map = dict(zip(df_nifty500['Ticker'], df_nifty500['Industry']))

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365)
    data = yf.download(tickers, start=start_date, end=end_date, interval='1d', group_by='ticker', auto_adjust=False, progress=False)

    results = []
    sector_perf = {}

    for ticker in tickers:
        try:
            df = data[ticker].copy().dropna()
            df['50EMA'] = df['Adj Close'].ewm(span=50).mean()
            df['20EMA'] = df['Adj Close'].ewm(span=20).mean()
            df['20D_High'] = df['High'].rolling(window=20).max()
            df['52W_High'] = df['High'].rolling(window=252).max()
            df['Avg_Vol_20D'] = df['Volume'].rolling(window=20).mean()
            df['Daily_Range'] = df['High'] - df['Low']
            df['ADR'] = df['Daily_Range'].rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + df['Adj Close'].pct_change().add(1).rolling(window=14).apply(lambda x: np.mean(x[x > 0]) / np.mean(-x[x < 0]) if np.mean(-x[x < 0]) != 0 else 1)))

            latest = df.iloc[-1]
            prev_day = df.iloc[-2]
            prev_week = df.iloc[-6]
            month_ago = df.iloc[-22]

            return_1d = ((latest['Adj Close'] - prev_day['Adj Close']) / prev_day['Adj Close']) * 100
            return_1w = ((latest['Adj Close'] - prev_week['Adj Close']) / prev_week['Adj Close']) * 100
            return_1m = ((latest['Adj Close'] - month_ago['Adj Close']) / month_ago['Adj Close']) * 100

            sector = sector_map[ticker]
            sector_perf.setdefault(sector, []).append(return_1w)

            vol_spike = latest['Volume'] > 1.5 * latest['Avg_Vol_20D']
            near_52w_high = latest['Adj Close'] >= 0.95 * latest['52W_High']
            rvol = latest['Volume'] / latest['Avg_Vol_20D'] if latest['Avg_Vol_20D'] > 0 else 0

            setup = ""
            if latest['Adj Close'] >= 0.98 * max(latest['20D_High'], latest['52W_High']):
                setup = "📈 Breakout"
            elif latest['Adj Close'] >= 0.97 * latest['50EMA'] and latest['Adj Close'] <= 1.03 * latest['50EMA']:
                setup = "🔁 Retest"
            elif latest['Adj Close'] <= latest['20EMA'] * 1.02:
                setup = "↩️ Pullback"

            stop_loss = round(latest['Adj Close'] * 0.97, 2)
            target = round(max(latest['20D_High'], latest['52W_High']), 2)
            reward_risk = round((target - latest['Adj Close']) / (latest['Adj Close'] - stop_loss), 2) if latest['Adj Close'] - stop_loss > 0 else 0

            if 40 <= latest['RSI'] <= 70 and latest['Adj Close'] > latest['50EMA'] and rvol > 1.2:
                results.append({
                    'Ticker': ticker,
                    'Sector': sector,
                    'Price': round(latest['Adj Close'], 2),
                    'Return_1D': round(return_1d, 2),
                    'Return_1W': round(return_1w, 2),
                    'Return_1M': round(return_1m, 2),
                    'RSI': round(latest['RSI'], 2),
                    'ADR': round(latest['ADR'], 2),
                    'RVOL': round(rvol, 2),
                    '50EMA': round(latest['50EMA'], 2),
                    '20D_High': round(latest['20D_High'], 2),
                    '52W_High': round(latest['52W_High'], 2),
                    'Setup': setup,
                    'Volume': round(latest['Volume'] / 1e6, 2),
                    'Avg_Vol_20D': round(latest['Avg_Vol_20D'] / 1e6, 2),
                    'Vol_Spike': vol_spike,
                    'Near_52W_High': near_52w_high,
                    'Reward:Risk': reward_risk,
                    'Stop_Loss': stop_loss
                })
        except Exception:
            continue

    df_all = pd.DataFrame(results)

    sector_perf_avg = {k: sum(v)/len(v) for k, v in sector_perf.items()}
    top_sectors = sorted(sector_perf_avg.items(), key=lambda x: x[1], reverse=True)[:5]
    top_sector_names = [s[0] for s in top_sectors]

st.markdown("### 🏆 Top Performing Sectors (1W Avg Return)")
cols = st.columns(len(top_sectors))
for i, (name, perf) in enumerate(top_sectors):
    cols[i].metric(label=name, value=f"{perf:.2f} %")

filtered = df_all[df_all['Sector'].isin(top_sector_names)]
filtered = filtered[filtered['Setup'].isin(['📈 Breakout', '🔁 Retest', '↩️ Pullback'])]
filtered = filtered[filtered['Reward:Risk'] > 1.5]

st.markdown("### 📈 Stock Setups in Leading Sectors")
st.dataframe(filtered.style.apply(lambda row: [
    'background-color: lightgreen' if row['Setup'] == '📈 Breakout' else
    'background-color: lightblue' if row['Setup'] == '🔁 Retest' else
    'background-color: orange' if row['Setup'] == '↩️ Pullback' else ''
]*len(row), axis=1), use_container_width=True)

# Export
st.download_button("📤 Export to Excel", data=BytesIO(filtered.to_excel(index=False, engine='openpyxl')).getvalue(), file_name="nifty500_screened.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Histogram
st.markdown("### 📊 Return Distribution of Selected Stocks")
fig, ax = plt.subplots(figsize=(12, 5))
sns.histplot(filtered['Return_1W'], kde=True, color='blue', label='1W Return')
sns.histplot(filtered['Return_1M'], kde=True, color='green', label='1M Return')
plt.legend()
plt.title("Histogram of Weekly & Monthly Returns", fontsize=14)
plt.xlabel("Return (%)")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)
st.pyplot(fig)
