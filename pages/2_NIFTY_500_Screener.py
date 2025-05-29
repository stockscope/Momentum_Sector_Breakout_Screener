import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Ensure Matplotlib's minus sign is rendered correctly
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide")
st.title("📊 Momentum Sector Breakout Screener (NIFTY 500)")
st.markdown("Identifies breakout or retest setups in top-performing sectors based on trend, volume, and returns.")

with st.expander("🧠 **Screening Criteria Used**", expanded=True):
    st.markdown("""
    - **Universe**: NIFTY 500 stocks
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

# Caching the main data processing function
@st.cache_data(ttl=timedelta(hours=1)) # Cache data for 1 hour
def run_screening_process(today_iso_format):
    # Load NIFTY 500 stock list
    csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty500list.csv"
    df_nifty500 = pd.read_csv(csv_url)
    df_nifty500.columns = df_nifty500.columns.str.strip() # Clean column names
    # Remove dummy entries and strip symbol whitespace
    df_nifty500 = df_nifty500[~df_nifty500['Symbol'].str.contains("DUMMY", na=False)]
    df_nifty500['Symbol'] = df_nifty500['Symbol'].str.strip()
    # Fill missing industries with 'Unknown'
    df_nifty500['Industry'] = df_nifty500['Industry'].fillna('Unknown')
    df_nifty500['Ticker'] = df_nifty500['Symbol'] + ".NS" # Append .NS for NSE tickers

    tickers = df_nifty500['Ticker'].tolist()
    sector_map = dict(zip(df_nifty500['Ticker'], df_nifty500['Industry']))

    if not tickers:
        return pd.DataFrame(), {} # Return empty if no tickers

    # Download historical price data
    end_date = datetime.fromisoformat(today_iso_format)
    start_date = end_date - timedelta(days=400) # Increased buffer for rolling window calculations
    
    stock_data = yf.download(tickers, start=start_date, end=end_date, interval='1d', 
                             group_by='ticker', auto_adjust=False, progress=False)

    results = []
    sector_returns_collector = {}

    # Analyze each ticker
    for ticker in tickers:
        try:
            if ticker not in stock_data or not isinstance(stock_data[ticker], pd.DataFrame) or stock_data[ticker].empty:
                continue
            
            df = stock_data[ticker].copy()
            # Drop rows if essential data points are missing
            df.dropna(subset=['Adj Close', 'High', 'Low', 'Open', 'Volume'], inplace=True)

            if len(df) < 22: # Need at least 22 days for 1-month return and 20-day indicators
                continue

            # Calculate technical indicators
            df['50EMA'] = df['Adj Close'].ewm(span=50, adjust=False).mean()
            df['20D_High'] = df['High'].rolling(window=20).max()
            df['52W_High'] = df['High'].rolling(window=252).max()
            df['Avg_Vol_20D'] = df['Volume'].rolling(window=20).mean()

            # Ensure latest row indicators are not NaN (can happen with sparse data even if len(df) is sufficient)
            if df[['50EMA', '20D_High', '52W_High', 'Avg_Vol_20D']].iloc[-1].isnull().any():
                continue

            latest = df.iloc[-1]
            # Ensure sufficient historical data points for return calculations
            # len(df) >= 22 implies indices -1, -2, -6, -22 are valid
            prev_day = df.iloc[-2]
            prev_week = df.iloc[-6] 
            month_ago = df.iloc[-22]

            # Calculate returns, handle potential division by zero
            if prev_day['Adj Close'] == 0 or prev_week['Adj Close'] == 0 or month_ago['Adj Close'] == 0:
                continue # Skip if any denominator is zero
            
            return_1d = ((latest['Adj Close'] - prev_day['Adj Close']) / prev_day['Adj Close']) * 100
            return_1w = ((latest['Adj Close'] - prev_week['Adj Close']) / prev_week['Adj Close']) * 100
            return_1m = ((latest['Adj Close'] - month_ago['Adj Close']) / month_ago['Adj Close']) * 100

            sector = sector_map.get(ticker, 'Unknown')
            if pd.notna(return_1w):
                 sector_returns_collector.setdefault(sector, []).append(return_1w)

            # Check for volume spike
            vol_spike = False
            if pd.notna(latest['Avg_Vol_20D']) and latest['Avg_Vol_20D'] > 0:
                vol_spike = latest['Volume'] > 1.5 * latest['Avg_Vol_20D']
            elif pd.notna(latest['Volume']): # If avg volume is 0 or NaN, check if current volume > 0
                 vol_spike = latest['Volume'] > 0


            # Check proximity to 52-week high
            near_52w_high = False
            if pd.notna(latest['52W_High']) and latest['52W_High'] > 0:
                near_52w_high = latest['Adj Close'] >= 0.95 * latest['52W_High']
            
            # Identify setup type
            setup = ""
            if pd.notna(latest['20D_High']) and latest['Adj Close'] >= 0.98 * latest['20D_High']:
                setup = "Breakout"
            elif pd.notna(latest['50EMA']) and \
                 latest['Adj Close'] >= latest['50EMA'] and \
                 latest['Adj Close'] <= 1.03 * latest['50EMA']:
                setup = "Retest"

            # Append results if essential data is present
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
                'Volume (M)': round(latest['Volume'] / 1e6, 1), # Volume in millions
                'Avg_Vol_20D (M)': round(latest['Avg_Vol_20D'] / 1e6, 1) if pd.notna(latest['Avg_Vol_20D']) else float('nan'),
                'Vol_Spike': vol_spike
            })
        except KeyError as e: # Handles issues like column not found in df
            # st.sidebar.write(f"KeyError for {ticker}: {e}") # Optional: log errors to sidebar for debugging
            continue
        except Exception as e: # Catch any other exception during processing for a ticker
            # st.sidebar.write(f"Error processing {ticker}: {e}") # Optional: log errors
            continue
    
    df_all = pd.DataFrame(results)

    # Calculate average sector performance robustly
    avg_sector_perf = {}
    for sector, returns_list in sector_returns_collector.items():
        valid_returns = [r for r in returns_list if pd.notna(r)]
        if valid_returns:
            avg_sector_perf[sector] = sum(valid_returns) / len(valid_returns)
            
    return df_all, avg_sector_perf

# Wrap the screening process with spinner
with st.spinner("🔍 Screening in progress... this may take a few minutes"):
    current_day_iso = datetime.today().strftime('%Y-%m-%d')
    df_all, sector_perf_avg = run_screening_process(current_day_iso)

    if df_all.empty:
        st.warning("No stock data could be processed. The NIFTY 500 list might be empty, or there were issues downloading data.")
        st.stop()

st.markdown(f"Data last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Sector ranking and display
st.markdown("### 🏆 Top Performing Sectors (1W Avg Return)")
if not sector_perf_avg:
    st.warning("No sector performance data available.")
    top_sector_names = []
    top_sectors_data = []
else:
    top_sectors_data = sorted(sector_perf_avg.items(), key=lambda x: x[1], reverse=True)[:5]
    top_sector_names = [s[0] for s in top_sectors_data]

if top_sectors_data:
    cols = st.columns(len(top_sectors_data))
    for i, (name, perf) in enumerate(top_sectors_data):
        cols[i].metric(label=name, value=f"{perf:.2f} %")
else:
    st.info("Not enough sector data to display top performing sectors.")


# Filter stocks based on criteria
df_filtered = df_all[(df_all['Sector'].isin(top_sector_names)) & 
                     (df_all['Setup'].isin(['Breakout', 'Retest']))]

# Sort results: Volume Spike (True first), then by 1M Return (descending)
df_filtered = df_filtered.sort_values(by=['Vol_Spike', 'Return_1M'], ascending=[False, False])

st.markdown("### 📈 Top Stock Setups in Leading Sectors")
if df_filtered.empty:
    st.info("No stocks found matching all screening criteria.")
else:
    # Display relevant columns, including 1D Return and Near_52W_High as per description
    display_cols = ['Ticker', 'Sector', 'Price', 'Return_1D', 'Return_1W', 'Return_1M', 
                    'Setup', 'Vol_Spike', 'Near_52W_High', 'Volume (M)']
    st.dataframe(df_filtered[display_cols].head(20), use_container_width=True, hide_index=True)

st.markdown("### 📊 Return Distribution of Selected Stocks (Top Sectors & Setups)")
if df_filtered.empty:
    st.info("No stocks available to plot return distribution.")
else:
    fig, ax = plt.subplots(figsize=(12, 6)) # Increased height slightly
    sns.histplot(data=df_filtered, x='Return_1W', kde=True, color='dodgerblue', label='1W Return', ax=ax, bins=15) # Adjusted color and bins
    sns.histplot(data=df_filtered, x='Return_1M', kde=True, color='mediumseagreen', label='1M Return', ax=ax, bins=15) # Adjusted color and bins
    
    plt.legend()
    plt.title("Histogram of Weekly & Monthly Returns for Selected Stocks", fontsize=15)
    plt.xlabel("Return (%)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    sns.despine() # Remove top and right spines for a cleaner look
    st.pyplot(fig)

st.markdown("---")
st.markdown("Disclaimer: This is an informational tool and not financial advice. Always do your own research before investing.")
