import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide")
st.title("📊 Momentum Sector Breakout Screener (NIFTY 200)")
st.markdown("Identifies breakout or retest setups in top-performing sectors based on trend, volume, and returns.")

with st.expander("🧠 **Screening Criteria Used**", expanded=True):
    st.markdown("""
    - **Universe**: NIFTY 200 stocks
    - **Top Sectors**: Based on average **1-week return**
    - **Setup Detection**:
        - 📈 **Breakout**: Close ≥ 98% of 20-day high
        - 🔁 **Retest**: Close ≥ 50 EMA and ≤ 103% of 50 EMA
    - **Filters**:
        - Stocks must belong to one of the top 5 performing sectors (1W return)
        - Volume Spike: Volume > 1.5× average 20-day volume
        - Proximity to 52W High is highlighted
    - **Displayed Metrics**:
        - Price, 1D/1W/1M Returns, Volume, Setup Type
    """)

# Wrap the screening process with spinner
with st.spinner("🔍 Screening in progress... please wait"):

    # Load NIFTY 200 CSV
    csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty200list.csv"
    df_nifty200 = pd.read_csv(csv_url)
    df_nifty200.columns = df_nifty200.columns.str.strip()
    df_nifty200 = df_nifty200[~df_nifty200['Symbol'].str.contains("DUMMY", na=False)]
    df_nifty200['Symbol'] = df_nifty200['Symbol'].str.strip()
    df_nifty200['Ticker'] = df_nifty200['Symbol'] + ".NS"

    tickers = df_nifty200['Ticker'].tolist()
    sector_map = dict(zip(df_nifty200['Ticker'], df_nifty200['Industry']))

    # Download historical data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    data = yf.download(tickers, start=start_date, end=end_date, interval='1d',
                       group_by='ticker', auto_adjust=False, progress=False)

    results = []
    sector_perf = {}

    for ticker in tickers:
        try:
            df = data[ticker].copy()
            df.dropna(inplace=True)

            df['50EMA'] = df['Adj Close'].ewm(span=50, adjust=False).mean()
            df['20D_High'] = df['High'].rolling(window=20).max()
            df['52W_High'] = df['High'].rolling(window=252).max()
            df['Avg_Vol_20D'] = df['Volume'].rolling(window=20).mean()

            if len(df) < 22:
                continue

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

            setup = ""
            if latest['Adj Close'] >= 0.98 * latest['20D_High']:
                setup = "Breakout"
            elif latest['Adj Close'] >= latest['50EMA'] and latest['Adj Close'] <= 1.03 * latest['50EMA']:
                setup = "Retest"

            results.append({
                'Ticker': ticker,
                'Sector': sector,
                'Price': round(latest['Adj Close'], 2),
                'Return_1D': round(return_1d, 2),
                'Return_1W': round(return_1w, 2),
                'Return_1M': round(return_1m, 2),
                '50EMA': round(latest['50EMA'], 2),
                '20D_High': round(latest['20D_High'], 2),
                '52W_High': round(latest['52W_High'], 2),
                'Near_52W_High': near_52w_high,
                'Setup': setup,
                'Volume': round(latest['Volume'] / 1e6, 1),
                'Avg_Vol_20D': round(latest['Avg_Vol_20D'] / 1e6, 1),
                'Vol_Spike': vol_spike
            })
        except Exception:
            continue

    df_all = pd.DataFrame(results)

    # Compute sector performance
    sector_perf_avg = {k: sum(v) / len(v) for k, v in sector_perf.items()}
    top_sectors = sorted(sector_perf_avg.items(), key=lambda x: x[1], reverse=True)[:5]
    top_sector_names = [s[0] for s in top_sectors]

# Show top sectors
st.markdown("### 🏆 Top Performing Sectors (1W Avg Return)")
cols = st.columns(len(top_sectors))
for i, (name, perf) in enumerate(top_sectors):
    cols[i].metric(label=name, value=f"{perf:.2f} %")

# Filter results
df_filtered = df_all[
    (df_all['Sector'].isin(top_sector_names)) &
    (df_all['Setup'].isin(['Breakout', 'Retest']))
].sort_values(by=['Vol_Spike', 'Return_1M'], ascending=[False, False])

# Show table
st.markdown("### 📈 Top Stock Setups in Leading Sectors")
st.dataframe(df_filtered[['Ticker', 'Sector', 'Price', 'Return_1W', 'Return_1M', 'Setup', 'Vol_Spike']].head(20),
             use_container_width=True)

# Histogram plot
st.markdown("### 📊 Return Distribution of Selected Stocks")
fig, ax = plt.subplots(figsize=(12, 5))
sns.histplot(data=df_filtered, x='Return_1W', kde=True, color='blue', label='1W Return')
sns.histplot(data=df_filtered, x='Return_1M', kde=True, color='green', label='1M Return')
plt.legend()
plt.title("Histogram of Weekly & Monthly Returns", fontsize=14)
plt.xlabel("Return (%)")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)
st.pyplot(fig)
