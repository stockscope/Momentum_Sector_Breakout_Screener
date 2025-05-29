import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from io import BytesIO

matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide")
st.title("📊 Momentum Sector Breakout Screener (NIFTY 500)")
st.markdown("Identifies breakout or retest setups in top-performing sectors based on trend, volume, and returns.")

with st.expander("🧠 **Screening Criteria Used**", expanded=True):
    st.markdown("""
    - **Universe**: NIFTY 500 stocks
    - **Top Sectors**: Based on average **1-week return**
    - **Setup Detection**:
        - 📈 **Breakout**: Close ≥ 98% of 20-day high and near 52W high
        - 🔁 **Retest**: Close ≥ 50 EMA and ≤ 103% of 50 EMA
        - 📉 **Pullback**: Price above all EMAs but short-term MA < long-term MA
    - **Filters**:
        - RSI (14) between 50–70
        - ADR > 3%
        - Avg Volume > 500K
        - RVOL > 1.5
        - Beta < 2.5
    - **Displayed Metrics**:
        - Price, Returns, EMAs, RSI, ADR, RVOL, Volume, Float (if available), Setup Type
    """)

with st.spinner("🔍 Screening in progress... please wait"):
    csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty500list.csv"
    df_nifty500 = pd.read_csv(csv_url)
    df_nifty500.columns = df_nifty500.columns.str.strip()
    df_nifty500 = df_nifty500[~df_nifty500['Symbol'].str.contains("DUMMY", na=False)]
    df_nifty500['Symbol'] = df_nifty500['Symbol'].str.strip()
    df_nifty500['Ticker'] = df_nifty500['Symbol'] + ".NS"

    tickers = df_nifty500['Ticker'].tolist()
    symbol_name_map = dict(zip(df_nifty500['Ticker'], df_nifty500['Company Name']))
    sector_map = dict(zip(df_nifty500['Ticker'], df_nifty500['Industry']))

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    data = yf.download(tickers, start=start_date, end=end_date, interval='1d', group_by='ticker', auto_adjust=False, progress=False)

    results = []
    sector_perf = {}

    for ticker in tickers:
        try:
            df = data[ticker].copy()
            df.dropna(inplace=True)

            df['10MA'] = df['Adj Close'].rolling(window=10).mean()
            df['20EMA'] = df['Adj Close'].ewm(span=20, adjust=False).mean()
            df['50EMA'] = df['Adj Close'].ewm(span=50, adjust=False).mean()
            df['200EMA'] = df['Adj Close'].ewm(span=200, adjust=False).mean()
            df['20D_High'] = df['High'].rolling(window=20).max()
            df['52W_High'] = df['High'].rolling(window=252).max()
            df['Avg_Vol_20D'] = df['Volume'].rolling(window=20).mean()
            df['Daily_Change'] = df['High'] - df['Low']
            df['ADR'] = df['Daily_Change'].rolling(window=20).mean() / df['Adj Close'] * 100

            delta = df['Adj Close'].diff()
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).rolling(window=14).mean()
            avg_loss = pd.Series(loss).rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))

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
            avg_volume_check = latest['Avg_Vol_20D'] > 500000
            rsi_ok = 50 <= latest['RSI'] <= 70
            adr_ok = latest['ADR'] > 3

            info = yf.Ticker(ticker).info
            beta = info.get('beta', np.nan)
            float_shares = info.get('floatShares', np.nan)

            setup = ""
            if (latest['Adj Close'] >= 0.98 * latest['20D_High'] and near_52w_high):
                setup = "Breakout"
            elif (latest['Adj Close'] >= latest['50EMA'] and latest['Adj Close'] <= 1.03 * latest['50EMA']):
                setup = "Retest"
            elif (latest['Adj Close'] > latest['20EMA'] and latest['20EMA'] > latest['50EMA'] and latest['10MA'] < latest['20EMA']):
                setup = "Pullback"

            if setup and vol_spike and avg_volume_check and rsi_ok and adr_ok and (not np.isnan(beta) and beta < 2.5):
                stop_loss_price = round(latest['Adj Close'] * 0.93, 2)
                reward = latest['52W_High'] - latest['Adj Close']
                risk = latest['Adj Close'] - stop_loss_price
                rr_ratio = round(reward / risk, 2) if risk > 0 else np.nan

                results.append({
                    'Ticker': ticker,
                    'Name': symbol_name_map.get(ticker, ""),
                    'Sector': sector,
                    'Price': round(latest['Adj Close'], 2),
                    'Return_1D': round(return_1d, 2),
                    'Return_1W': round(return_1w, 2),
                    'Return_1M': round(return_1m, 2),
                    'RSI_14': round(latest['RSI'], 1),
                    'ADR_%': round(latest['ADR'], 2),
                    '50EMA': round(latest['50EMA'], 2),
                    '52W_High': round(latest['52W_High'], 2),
                    'Vol_Spike': vol_spike,
                    'Avg_Vol_20D': round(latest['Avg_Vol_20D'] / 1e6, 2),
                    'Volume': round(latest['Volume'] / 1e6, 2),
                    'Float': float_shares,
                    'Beta': beta,
                    'Setup': setup,
                    'Stop_Loss': stop_loss_price,
                    'Reward:Risk': rr_ratio
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

df_filtered = df_all[(df_all['Sector'].isin(top_sector_names)) & (df_all['Setup'].isin(['Breakout', 'Retest', 'Pullback']))]
df_filtered = df_filtered.sort_values(by=['Vol_Spike', 'Return_1M'], ascending=[False, False])

st.markdown("### 📈 Top Stock Setups in Leading Sectors")
setup_colors = {"Breakout": "#34c759", "Retest": "#007aff", "Pullback": "#ff9500"}
df_display = df_filtered.copy()
df_display['Setup'] = df_display['Setup'].apply(lambda x: f"<span style='color:{setup_colors.get(x, 'black')}'>{x}</span>")
st.write(df_display[['Ticker', 'Name', 'Sector', 'Price', 'Return_1W', 'Return_1M', 'RSI_14', 'ADR_%', 'Setup']].head(20).to_html(escape=False, index=False), unsafe_allow_html=True)

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

st.markdown("### 📥 Download Filtered Results")
@st.cache_data

def convert_df(df):
    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    processed_data = output.getvalue()
    return processed_data

download_data = convert_df(df_filtered)
st.download_button(
    label="📄 Download as Excel",
    data=download_data,
    file_name="nifty500_momentum_screen.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
