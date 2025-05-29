import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np

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
        - Volume Spike: Volume > 1.5× average 20-day volume (used for sorting, True values at top)
        - Proximity to 52W High is highlighted (shown as a column)
    - **Displayed Metrics**:
        - Price, 1D/1W/1M Returns, Volume, Setup Type, Near 52W High, Volume Spike
    """)

@st.cache_data(ttl=timedelta(hours=1))
def run_screening_process(today_iso_format):
    csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty200list.csv"
    df_nifty200 = pd.read_csv(csv_url)
    df_nifty200.columns = df_nifty200.columns.str.strip()
    df_nifty200 = df_nifty200[~df_nifty200['Symbol'].str.contains("DUMMY", na=False)]
    df_nifty200['Symbol'] = df_nifty200['Symbol'].str.strip()
    df_nifty200['Industry'] = df_nifty200['Industry'].fillna('Unknown')
    df_nifty200['Ticker'] = df_nifty200['Symbol'] + ".NS"

    tickers = df_nifty200['Ticker'].tolist()
    sector_map = dict(zip(df_nifty200['Ticker'], df_nifty200['Industry']))

    if not tickers:
        return pd.DataFrame(), {}

    end_date = datetime.fromisoformat(today_iso_format)
    start_date = end_date - timedelta(days=400)

    stock_data = yf.download(tickers, start=start_date, end=end_date, interval='1d', group_by='ticker', auto_adjust=False, progress=False)

    results = []
    sector_returns_collector = {}

    for ticker in tickers:
        try:
            if ticker not in stock_data or not isinstance(stock_data[ticker], pd.DataFrame) or stock_data[ticker].empty:
                continue
            df = stock_data[ticker].copy()
            df.dropna(subset=['Adj Close', 'High', 'Low', 'Open', 'Volume'], inplace=True)
            if len(df) < 22:
                continue

            # Indicators
            df['50EMA'] = df['Adj Close'].ewm(span=50, adjust=False).mean()
            df['20D_High'] = df['High'].rolling(window=20).max()
            df['52W_High'] = df['High'].rolling(window=252).max()
            df['Avg_Vol_20D'] = df['Volume'].rolling(window=20).mean()

            # RSI Calculation
            delta = df['Adj Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            if df[['50EMA', '20D_High', '52W_High', 'Avg_Vol_20D', 'RSI']].iloc[-1].isnull().any():
                continue

            latest = df.iloc[-1]
            prev_day = df.iloc[-2]
            prev_week = df.iloc[-6]
            month_ago = df.iloc[-22]

            if prev_day['Adj Close'] == 0 or prev_week['Adj Close'] == 0 or month_ago['Adj Close'] == 0:
                continue

            return_1d = ((latest['Adj Close'] - prev_day['Adj Close']) / prev_day['Adj Close']) * 100
            return_1w = ((latest['Adj Close'] - prev_week['Adj Close']) / prev_week['Adj Close']) * 100
            return_1m = ((latest['Adj Close'] - month_ago['Adj Close']) / month_ago['Adj Close']) * 100

            sector = sector_map.get(ticker, 'Unknown')
            if pd.notna(return_1w):
                sector_returns_collector.setdefault(sector, []).append(return_1w)

            vol_spike = False
            if pd.notna(latest['Avg_Vol_20D']) and latest['Avg_Vol_20D'] > 0:
                vol_spike = latest['Volume'] > 1.5 * latest['Avg_Vol_20D']
            elif pd.notna(latest['Volume']):
                vol_spike = latest['Volume'] > 0

            near_52w_high = False
            if pd.notna(latest['52W_High']) and latest['52W_High'] > 0:
                near_52w_high = latest['Adj Close'] >= 0.95 * latest['52W_High']

            setup = ""
            if pd.notna(latest['20D_High']) and latest['Adj Close'] >= 0.98 * latest['20D_High']:
                setup = "Breakout"
            elif pd.notna(latest['50EMA']) and latest['Adj Close'] >= latest['50EMA'] and latest['Adj Close'] <= 1.03 * latest['50EMA']:
                setup = "Retest"

            if any(pd.isna(val) for val in [latest['Adj Close'], return_1d, return_1w, return_1m, latest['Volume']]):
                continue

            results.append({
                'Ticker': ticker,
                'Sector': sector,
                'Price': round(latest['Adj Close'], 2),
                'Return_1D': round(return_1d, 2),
                'Return_1W': round(return_1w, 2),
                'Return_1M': round(return_1m, 2),
                '50EMA': round(latest['50EMA'], 2) if pd.notna(latest['50EMA']) else float('nan'),
                '20D_High': round(latest['20D_High'], 2) if pd.notna(latest['20D_High']) else float('nan'),
                '52W_High': round(latest['52W_High'], 2) if pd.notna(latest['52W_High']) else float('nan'),
                'Near_52W_High': near_52w_high,
                'Setup': setup,
                'Volume (M)': round(latest['Volume'] / 1e6, 1),
                'Avg_Vol_20D (M)': round(latest['Avg_Vol_20D'] / 1e6, 1) if pd.notna(latest['Avg_Vol_20D']) else float('nan'),
                'Vol_Spike': vol_spike,
                'RSI': round(df['RSI'].iloc[-1], 2) if pd.notna(df['RSI'].iloc[-1]) else float('nan')
            })
        except Exception:
            continue

    df_all = pd.DataFrame(results)

    avg_sector_perf = {}
    for sector, returns_list in sector_returns_collector.items():
        valid_returns = [r for r in returns_list if pd.notna(r)]
        if valid_returns:
            avg_sector_perf[sector] = sum(valid_returns) / len(valid_returns)

    return df_all, avg_sector_perf

with st.spinner("🔍 Screening in progress... this may take a few minutes"):
    current_day_iso = datetime.today().strftime('%Y-%m-%d')
    df_all, sector_perf_avg = run_screening_process(current_day_iso)

if df_all.empty:
    st.warning("No stock data could be processed. The NIFTY 200 list might be empty, or there were issues downloading data.")
    st.stop()

st.markdown(f"Data last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Sidebar filters
st.sidebar.header("Filter Stocks")
top_sectors_data = sorted(sector_perf_avg.items(), key=lambda x: x[1], reverse=True)
top_sector_names = [s[0] for s in top_sectors_data][:5]

selected_sectors = st.sidebar.multiselect(
    "Select Sectors",
    options=sorted(list(set(df_all['Sector']))),
    default=top_sector_names
)

setup_options = ['Breakout', 'Retest']
selected_setups = st.sidebar.multiselect("Select Setup Type(s)", options=setup_options, default=setup_options)

volume_spike_filter = st.sidebar.checkbox("Show only Volume Spike Stocks", value=False)

# Filtering data based on sidebar
df_filtered = df_all[
    (df_all['Sector'].isin(selected_sectors)) &
    (df_all['Setup'].isin(selected_setups))
]

if volume_spike_filter:
    df_filtered = df_filtered[df_filtered['Vol_Spike'] == True]

df_filtered = df_filtered.sort_values(by=['Vol_Spike', 'Return_1M'], ascending=[False, False])

st.markdown("### 🏆 Top Performing Sectors (1W Avg Return)")
if not sector_perf_avg:
    st.warning("No sector performance data available.")
else:
    cols = st.columns(min(len(top_sectors_data), 5))
    for i, (name, perf) in enumerate(top_sectors_data[:5]):
        cols[i].metric(label=name, value=f"{perf:.2f} %")

st.markdown("### 📈 Filtered Stock Setups")

# Styling DataFrame: color and icons for Setup and Volume Spike
def color_setup(val):
    if val == "Breakout":
        color = 'background-color: #a6d96a;'  # greenish
        icon = "📈"
    elif val == "Retest":
        color = 'background-color: #fdae61;'  # orange
        icon = "🔁"
    else:
        color = ''
        icon = ''
    return f'{color} {icon} {val}'

def color_vol_spike(val):
    icon = "🔥" if val else ""
    color = 'background-color: #fee08b;' if val else ''
    return f'{color} {icon}'

if df_filtered.empty:
    st.info("No stocks found matching the selected filters.")
else:
    display_cols = ['Ticker', 'Sector', 'Price', 'Return_1D', 'Return_1W', 'Return_1M', 
                    'Setup', 'Vol_Spike', 'Near_52W_High', 'Volume (M)', 'RSI']

    df_display = df_filtered[display_cols].copy()

    # Apply icons and colors using pandas Styler
    def highlight_setup(row):
        return ['' if col not in ['Setup'] else
                ('background-color: #a6d96a;' if row[col] == 'Breakout' else
                 'background-color: #fdae61;' if row[col] == 'Retest' else '')
                for col in df_display.columns]

    def highlight_vol_spike(row):
        return ['background-color: #fee08b;' if col == 'Vol_Spike' and row[col] else '' for col in df_display.columns]

    styler = df_display.style.format({
        'Return_1D': "{:.2f} %",
        'Return_1W': "{:.2f} %",
        'Return_1M': "{:.2f} %",
        'Price': "{:.2f}",
        'RSI': "{:.2f}"
    })

    # Add icons in Setup and Vol_Spike columns
    def icon_formatter(val, col):
        if col == 'Setup':
            if val == "Breakout":
                return f"📈 {val}"
            elif val == "Retest":
                return f"🔁 {val}"
            else:
                return val
        elif col == 'Vol_Spike':
            return "🔥" if val else ""
        elif col == 'Near_52W_High':
            return "⭐" if val else ""
        return val

    for col in ['Setup', 'Vol_Spike', 'Near_52W_High']:
        styler = styler.format({col: lambda x, col=col: icon_formatter(x, col)})

    # Apply row-wise coloring for Setup and Vol_Spike
    styler = styler.apply(highlight_setup, axis=1)
    styler = styler.apply(highlight_vol_spike, axis=1)

    st.dataframe(styler, use_container_width=True)

    # CSV Download button
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"nifty200_filtered_{datetime.today().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )

# Additional RSI Distribution chart
st.markdown("### 📊 RSI Distribution for Filtered Stocks")
if df_filtered.empty:
    st.info("No data available for RSI distribution.")
else:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df_filtered['RSI'].dropna(), bins=20, kde=True, color='purple', ax=ax)
    ax.axvline(70, color='red', linestyle='--', label='Overbought (70)')
    ax.axvline(30, color='green', linestyle='--', label='Oversold (30)')
    ax.set_xlabel("RSI")
    ax.set_ylabel("Count")
    ax.set_title("RSI Histogram of Filtered Stocks")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")
st.markdown("Disclaimer: This is an informational tool and not financial advice. Always do your own research before investing.")
