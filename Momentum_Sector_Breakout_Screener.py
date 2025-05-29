
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Top Sector Momentum Stocks Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("📊 Top Sector Momentum Stocks Screener (NIFTY 500)")

# Screening Criteria
st.markdown("### 📌 Screening Criteria Used")
st.markdown("""
- **Universe**: NIFTY 500 Stocks  
- **Sector Ranking**: Top 5 sectors by 1-week average return  
- **Setup Types**:  
  - **Breakout**: Price ≥ 98% of 20-day high  
  - **Retest**: Price near 50 EMA (within +3%)  
- **Volume Spike**: Volume > 1.5× 20-day average volume  
- **Near 52W High**: Price ≥ 95% of 52-week high  
""")

# Show spinner during screening
with st.spinner("🔍 Screening in progress... please wait"):

    # Load NIFTY 500 CSV from GitHub
    csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty500list.csv"
    df_nifty500 = pd.read_csv(csv_url)
    df_nifty500.columns = df_nifty500.columns.str.strip()
    df_nifty500 = df_nifty500[~df_nifty500['Symbol'].str.contains("DUMMY", na=False)]
    df_nifty500['Symbol'] = df_nifty500['Symbol'].str.strip()
    df_nifty500['Ticker'] = df_nifty500['Symbol'] + ".NS"

    tickers = df_nifty500['Ticker'].tolist()
    sector_map = dict(zip(df_nifty500['Ticker'], df_nifty500['Industry']))

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    data = yf.download(tickers, start=start_date, end=end_date, interval='1d', group_by='ticker', auto_adjust=False, progress=False)

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
        except Exception as e:
            st.warning(f"⚠️ Skipped {ticker}: {e}")

    df_all = pd.DataFrame(results)
    sector_perf_avg = {k: sum(v)/len(v) for k, v in sector_perf.items()}
    top_sectors = sorted(sector_perf_avg.items(), key=lambda x: x[1], reverse=True)[:5]
    top_sector_names = [s[0] for s in top_sectors]

    st.markdown("### 🏆 Top Performing Sectors (1W Avg Return)")
    for name, perf in top_sectors:
        st.write(f"- {name}: {perf:.2f}%")

    df_filtered = df_all[(df_all['Sector'].isin(top_sector_names)) & (df_all['Setup'].isin(['Breakout', 'Retest']))]
    df_filtered = df_filtered.sort_values(by=['Vol_Spike', 'Return_1M'], ascending=[False, False])

    st.markdown("### 📈 Trade Candidates from Leading Sectors")
    st.dataframe(df_filtered[['Ticker', 'Sector', 'Price', 'Return_1W', 'Return_1M', 'Setup', 'Vol_Spike']], use_container_width=True)

    # Distribution plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=df_filtered, x='Return_1W', kde=True, color='orange', label='1W Return')
    sns.histplot(data=df_filtered, x='Return_1M', kde=True, color='green', label='1M Return')
    plt.legend()
    plt.title("📊 Return Distribution of Momentum Candidates")
    plt.xlabel("Return (%)")
    plt.grid(True)
    st.pyplot(fig)
