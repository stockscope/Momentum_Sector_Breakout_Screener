import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pytz

matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide")
st.title("📊 Momentum Sector Breakout Screener (NIFTY 500)")
st.markdown("Identifies breakout, retest, and pullback setups in top-performing sectors based on trend, volume, and returns.")

with st.expander("🧠 **Screening Criteria Used**", expanded=True):
    st.markdown("""
    - **Universe**: NIFTY 500 stocks
    - **Top Sectors**: Based on average **1-week return**
    - **Setup Detection**:
        - 📈 **Breakout**: Close ≥ 98% of multi-timeframe highs (20D/50D/100D/200D/252D/500D)
        - 🔁 **Retest**: Close ≥ 50 EMA and ≤ 103% of 50 EMA
        - 🔄 **Pullback**: Close ≤ 97% of 20D High and ≥ 20 EMA
    - **Filters**:
        - Stocks must belong to one of the top 5 performing sectors (1W return)
        - Volume Spike: Volume > 1.5× average 20-day volume
        - Proximity to 52W High is highlighted
    - **Displayed Metrics**:
        - Price, Returns, Volume, Setup Type, Breakout Level
    """)

# Timezone-safe datetime
default_timezone = pytz.timezone("Asia/Kolkata")
end_date = datetime.now(default_timezone)
start_date = end_date - timedelta(days=550)

# Load CSV
csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty500list.csv"
df_nifty500 = pd.read_csv(csv_url)
df_nifty500.columns = df_nifty500.columns.str.strip()
df_nifty500 = df_nifty500[~df_nifty500['Symbol'].str.contains("DUMMY", na=False)]
df_nifty500['Symbol'] = df_nifty500['Symbol'].str.strip()
df_nifty500['Ticker'] = df_nifty500['Symbol'] + ".NS"

# Map Ticker -> Sector and Ticker -> Name
tickers = df_nifty500['Ticker'].tolist()
sector_map = dict(zip(df_nifty500['Ticker'], df_nifty500['Industry']))
name_map = dict(zip(df_nifty500['Ticker'], df_nifty500['Company Name']))

@st.cache_data(show_spinner=False)
def fetch_yf_data(tickers):
    return yf.download(tickers, start=start_date, end=end_date, interval='1d', group_by='ticker', auto_adjust=False, progress=False)

with st.spinner("🔍 Screening in progress... please wait"):
    data = fetch_yf_data(tickers)

    results = []
    sector_perf = {}

    for ticker in tickers:
        try:
            df = data[ticker].copy()
            df.dropna(inplace=True)

            df['50EMA'] = df['Adj Close'].ewm(span=50, adjust=False).mean()
            df['20EMA'] = df['Adj Close'].ewm(span=20, adjust=False).mean()
            for window in [20, 50, 100, 200, 252, 500]:
                df[f'{window}D_High'] = df['High'].rolling(window=window).max()

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
            breakout_level = ""
            for window in [500, 252, 200, 100, 50, 20]:
                if latest['Adj Close'] >= 0.98 * latest[f'{window}D_High']:
                    setup = f"Breakout ({window}D)"
                    breakout_level = f"{window}D"
                    break
            if not setup and latest['Adj Close'] >= latest['50EMA'] and latest['Adj Close'] <= 1.03 * latest['50EMA']:
                setup = "Retest"
            if not setup and latest['Adj Close'] <= 0.97 * latest['20D_High'] and latest['Adj Close'] >= latest['20EMA']:
                setup = "Pullback"

            results.append({
                'Ticker': ticker,
                'Name': name_map.get(ticker, ""),
                'Sector': sector,
                'Price': round(latest['Adj Close'], 2),
                'Return_1D': round(return_1d, 2),
                'Return_1W': round(return_1w, 2),
                'Return_1M': round(return_1m, 2),
                'Setup': setup,
                'Breakout_Level': breakout_level,
                'Volume (Mn)': round(latest['Volume'] / 1e6, 1),
                'Avg_Vol_20D (Mn)': round(latest['Avg_Vol_20D'] / 1e6, 1),
                'Vol_Spike': vol_spike,
                'Near_52W_High': near_52w_high
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

df_filtered = df_all[(df_all['Sector'].isin(top_sector_names)) & (df_all['Setup'] != "")]
df_filtered = df_filtered.sort_values(by=['Vol_Spike', 'Return_1M'], ascending=[False, False])

st.markdown("### 📈 Top Stock Setups in Leading Sectors")
setup_color = {
    'Breakout (20D)': 'green', 'Breakout (50D)': 'darkgreen', 'Breakout (100D)': 'seagreen',
    'Breakout (200D)': 'forestgreen', 'Breakout (252D)': 'limegreen', 'Breakout (500D)': 'springgreen',
    'Retest': 'orange', 'Pullback': 'blue'
}

def highlight_setup(val):
    color = setup_color.get(val, '')
    return f'background-color: {color}; color: white' if color else ''

st.dataframe(df_filtered[['Ticker', 'Name', 'Sector', 'Price', 'Return_1W', 'Return_1M', 'Setup', 'Breakout_Level', 'Vol_Spike']].style.applymap(highlight_setup, subset=['Setup']), use_container_width=True)

csv = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Results as CSV", csv, "breakout_screener_results.csv", "text/csv")

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
