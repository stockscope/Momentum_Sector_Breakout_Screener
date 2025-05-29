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
    - **Top Sectors Selection**: Defaults to the top 5 performing sectors based on average **1-week return** (user can select other sectors).
    - **Setup Detection (Stricter)**:
        - 📈 **Breakout**: Close ≥ **99%** of 20-day high (previously 98%)
        - 🔁 **Retest**: Close ≥ 50 EMA and ≤ **102%** of 50 EMA (previously 103%)
    - **Filters**:
        - Volume Spike: Volume > 1.5× average 20-day volume (used for sorting, True values at top)
        - Proximity to 52W High is highlighted (shown as a column, Close ≥ 95% of 52W High)
    - **Displayed Metrics**:
        - Price, 1D/1W/1M Returns, Volume, Setup Type, Near 52W High, Volume Spike, RSI
    """)

# --- Refactored Data Fetching and Processing Functions ---

@st.cache_data(ttl=timedelta(days=1))
def load_nifty200_list_and_map():
    csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty200list.csv"
    try:
        df_nifty200 = pd.read_csv(csv_url)
    except Exception as e:
        st.error(f"Error loading NIFTY 200 stock list from CSV: {e}")
        return [], {}, {} # Return empty structures on error
        
    df_nifty200.columns = df_nifty200.columns.str.strip()
    df_nifty200 = df_nifty200[~df_nifty200['Symbol'].str.contains("DUMMY", na=False)]
    df_nifty200['Symbol'] = df_nifty200['Symbol'].str.strip()
    df_nifty200['Industry'] = df_nifty200['Industry'].fillna('Unknown')
    df_nifty200['Ticker'] = df_nifty200['Symbol'] + ".NS"

    tickers = df_nifty200['Ticker'].tolist()
    sector_map = dict(zip(df_nifty200['Ticker'], df_nifty200['Industry']))
    return tickers, sector_map, df_nifty200 # Return df_nifty200 for potential future use

@st.cache_data(ttl=timedelta(hours=1))
def fetch_stock_data_from_yfinance(tickers_tuple, start_date_str, end_date_str):
    tickers_list = list(tickers_tuple)
    if not tickers_list:
        return {}
        
    stock_data_downloaded = yf.download(
        tickers_list, 
        start=start_date_str, 
        end=end_date_str, 
        interval='1d', 
        group_by='ticker', 
        auto_adjust=False, 
        progress=False,
        timeout=30 # Added timeout
    )
    
    stock_data_processed = {}
    if stock_data_downloaded.empty:
        return {}

    if isinstance(stock_data_downloaded.columns, pd.MultiIndex):
        for ticker in tickers_list:
            try:
                # Check if data for the ticker is actually present and is a DataFrame
                if ticker in stock_data_downloaded and isinstance(stock_data_downloaded[ticker], pd.DataFrame):
                    stock_data_processed[ticker] = stock_data_downloaded[ticker]
            except KeyError: # Should be caught by 'ticker in stock_data_downloaded'
                pass 
    elif len(tickers_list) == 1 and isinstance(stock_data_downloaded, pd.DataFrame): # Single ticker downloaded
        stock_data_processed[tickers_list[0]] = stock_data_downloaded
        
    return stock_data_processed

@st.cache_data(ttl=timedelta(hours=1))
def analyze_stocks_and_sectors(downloaded_stock_data, tickers_tuple, sector_map_dict, current_day_iso_str_for_analysis):
    results = []
    sector_returns_collector = {}
    tickers_list = list(tickers_tuple)

    for ticker in tickers_list:
        try:
            if ticker not in downloaded_stock_data or downloaded_stock_data[ticker].empty:
                continue
            
            df = downloaded_stock_data[ticker].copy()
            df.dropna(subset=['Adj Close', 'High', 'Low', 'Open', 'Volume'], inplace=True)
            if len(df) < 22: # Minimum for 20-day calcs and 1-month return
                continue

            # Indicators
            df['50EMA'] = df['Adj Close'].ewm(span=50, adjust=False).mean()
            df['20D_High'] = df['High'].rolling(window=20, min_periods=1).max()
            df['52W_High'] = df['High'].rolling(window=252, min_periods=1).max()
            df['Avg_Vol_20D'] = df['Volume'].rolling(window=20, min_periods=1).mean()

            # RSI Calculation
            delta = df['Adj Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            
            rs_val = np.nan # Default RSI to NaN
            # Check if latest loss is available and not NaN
            if pd.notna(loss.iloc[-1]):
                if loss.iloc[-1] == 0: # Avoid division by zero
                    # If gain is also 0 or NaN, RSI is typically 50 or undefined.
                    # If gain > 0, RSI is 100.
                    if pd.notna(gain.iloc[-1]) and gain.iloc[-1] > 0:
                        rs_val = np.inf
                    else: # gain is 0 or NaN
                        # A common convention is RSI = 50 if avg_gain = avg_loss = 0, but yfinance behavior might vary
                        # For simplicity, if loss is 0 and gain is 0 or NaN, we might leave rs_val as NaN or set to 0 for RSI calculation.
                        # If both are zero, it means no price change in 14 periods for RSI calc.
                        # Let's default to 0 for RS if both are 0 or gain is not positive.
                        rs_val = 0 # This would lead to RSI 50 if rs = 0 if gain is also 0
                        if pd.notna(gain.iloc[-1]) and gain.iloc[-1] == 0 and loss.iloc[-1] == 0:
                             pass # rs_val remains 0, implies RSI = 100 / (1+0) = 100, which is wrong.
                                  # RSI should be 50 if no net change. Or neutral.
                                  # If gain and loss are both 0, rs is undefined. Some use RS=1 -> RSI=50.
                                  # Let's stick to the formula: if loss is 0 and gain is 0, rs becomes 0 (by gain/loss logic) -> RSI = 100.
                                  # If loss is 0 and gain is >0, rs becomes inf -> RSI = 100.
                                  # This seems fine. The critical part is loss.iloc[-1] != 0.
                                  # If gain.iloc[-1] / loss.iloc[-1]
                                  # if loss.iloc[-1] == 0 and gain.iloc[-1] == 0 : RSI can be considered 50 or remain NaN.
                                  # Let's ensure we handle division by zero robustly for RS.
                                  # RSI is 100 - (100 / (1 + RS))
                                  # if loss is 0, gain is 0 -> RS = 0/0 (NaN). RSI -> NaN.
                                  # if loss is 0, gain > 0 -> RS = inf. RSI -> 100.
                                  # Let's use try-except for division
                                  pass # rs_val remains NaN if gain is also 0 or NaN
                        
                else: # loss.iloc[-1] is not 0 and not NaN
                    if pd.notna(gain.iloc[-1]):
                         rs_val = gain.iloc[-1] / loss.iloc[-1]


            latest_rsi = np.nan
            if pd.notna(rs_val):
                if rs_val == np.inf : latest_rsi = 100.0
                else: latest_rsi = 100 - (100 / (1 + rs_val))
            df['RSI'] = latest_rsi # Assign to the last row for collection

            if df[['50EMA', '20D_High', 'Avg_Vol_20D']].iloc[-1].isnull().any(): # RSI can be NaN
                continue

            latest = df.iloc[-1]
            
            # Ensure enough historical data for returns
            prev_day_adj_close = df['Adj Close'].iloc[-2] if len(df) >= 2 else np.nan
            prev_week_adj_close = df['Adj Close'].iloc[-6] if len(df) >= 6 else np.nan # 5 trading days ago
            month_ago_adj_close = df['Adj Close'].iloc[-22] if len(df) >= 22 else np.nan # Approx 21 trading days ago

            return_1d = ((latest['Adj Close'] - prev_day_adj_close) / prev_day_adj_close) * 100 if pd.notna(prev_day_adj_close) and prev_day_adj_close != 0 else np.nan
            return_1w = ((latest['Adj Close'] - prev_week_adj_close) / prev_week_adj_close) * 100 if pd.notna(prev_week_adj_close) and prev_week_adj_close != 0 else np.nan
            return_1m = ((latest['Adj Close'] - month_ago_adj_close) / month_ago_adj_close) * 100 if pd.notna(month_ago_adj_close) and month_ago_adj_close != 0 else np.nan
            
            sector = sector_map_dict.get(ticker, 'Unknown')
            if pd.notna(return_1w):
                sector_returns_collector.setdefault(sector, []).append(return_1w)

            vol_spike = False
            if pd.notna(latest['Avg_Vol_20D']) and latest['Avg_Vol_20D'] > 0:
                vol_spike = latest['Volume'] > 1.5 * latest['Avg_Vol_20D']
            elif pd.notna(latest['Volume']) and latest['Volume'] > 0 : # Avg_Vol_20D is 0 or NaN, but Volume exists
                vol_spike = True 

            near_52w_high = False
            if pd.notna(latest['52W_High']) and latest['52W_High'] > 0:
                near_52w_high = latest['Adj Close'] >= 0.95 * latest['52W_High']

            setup = ""
            # Stricter criteria
            if pd.notna(latest['20D_High']) and latest['Adj Close'] >= 0.99 * latest['20D_High']: # Changed from 0.98
                setup = "Breakout"
            elif pd.notna(latest['50EMA']) and latest['Adj Close'] >= latest['50EMA'] and latest['Adj Close'] <= 1.02 * latest['50EMA']: # Changed from 1.03
                setup = "Retest"
            
            if pd.isna(latest['Adj Close']) or pd.isna(latest['Volume']) or setup == "":
                 continue

            results.append({
                'Ticker': ticker, 'Sector': sector,
                'Price': latest['Adj Close'], # Keep as float, round later
                'Return_1D': return_1d, 'Return_1W': return_1w, 'Return_1M': return_1m,
                '50EMA': latest['50EMA'], '20D_High': latest['20D_High'], '52W_High': latest['52W_High'],
                'Near_52W_High': near_52w_high, 'Setup': setup,
                'Volume (M)': latest['Volume'] / 1e6 if pd.notna(latest['Volume']) else np.nan,
                'Avg_Vol_20D (M)': latest['Avg_Vol_20D'] / 1e6 if pd.notna(latest['Avg_Vol_20D']) else np.nan,
                'Vol_Spike': vol_spike,
                'RSI': latest['RSI'] # Already calculated and stored in df
            })
        except Exception as e:
            # st.sidebar.warning(f"Skipping {ticker} due to error: {str(e)[:50]}") # For debugging
            continue

    df_all = pd.DataFrame(results)
    if not df_all.empty:
        # Drop rows where essential data for display is missing or no setup identified
        df_all.dropna(subset=['Price', 'Ticker', 'Setup'], inplace=True)
        df_all = df_all[df_all['Setup'] != ""]

    avg_sector_perf = {}
    for sector, returns_list in sector_returns_collector.items():
        valid_returns = [r for r in returns_list if pd.notna(r)]
        if valid_returns:
            avg_sector_perf[sector] = np.mean(valid_returns)

    return df_all, avg_sector_perf

# --- Main App Logic ---
current_day_iso = datetime.today().strftime('%Y-%m-%d')
# For yfinance, end date is inclusive for string dates.
# Use current_day_iso for end_date to get data up to 'today' if available.
fetch_end_date = current_day_iso
fetch_start_date = (datetime.today() - timedelta(days=400)).strftime('%Y-%m-%d')

with st.spinner("📜 Loading NIFTY 200 list..."):
    tickers, sector_map, _ = load_nifty200_list_and_map()

if not tickers:
    st.error("NIFTY 200 stock list could not be loaded. Cannot proceed.")
    st.stop()

with st.spinner("📥 Fetching market data from yfinance... (this may take a few minutes)"):
    # Pass tuples for cache hashing
    downloaded_stock_data = fetch_stock_data_from_yfinance(tuple(tickers), fetch_start_date, fetch_end_date)

if not downloaded_stock_data:
    st.warning("No stock data could be downloaded from yfinance. Results might be incomplete or empty.")
    # Don't stop yet, analyze_stocks_and_sectors should handle empty dict gracefully
    # and df_all_results will be empty, triggering the later warning.

with st.spinner("⚙️ Processing data and identifying setups..."):
    df_all_results, sector_perf_avg_results = analyze_stocks_and_sectors(
        downloaded_stock_data, tuple(tickers), sector_map, current_day_iso
    )

if df_all_results.empty:
    st.warning("No stock data could be processed or no stocks met the screening criteria. This could be due to data download issues, no stocks matching the (now stricter) setup conditions, or market holidays.")
    st.stop()

st.markdown(f"Data processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Market data up to {current_day_iso} or latest available)")

# Sidebar filters
st.sidebar.header("Filter Stocks")
sorted_sector_perf = sorted(sector_perf_avg_results.items(), key=lambda x: x[1], reverse=True)
all_available_sectors = sorted(list(df_all_results['Sector'].unique()))
top_5_sector_names = [s[0] for s in sorted_sector_perf[:5] if s[0] in all_available_sectors]

selected_sectors = st.sidebar.multiselect(
    "Select Sectors (defaults to top 5 performing)",
    options=all_available_sectors,
    default=top_5_sector_names
)

setup_options = sorted(list(df_all_results['Setup'].unique())) # Dynamic from results
selected_setups = st.sidebar.multiselect("Select Setup Type(s)", options=setup_options, default=setup_options)

volume_spike_filter = st.sidebar.checkbox("Show only Volume Spike Stocks", value=False)
near_52w_high_filter = st.sidebar.checkbox("Show only Near 52W High Stocks", value=False)

# Filtering data
df_filtered = df_all_results.copy()
if selected_sectors:
    df_filtered = df_filtered[df_filtered['Sector'].isin(selected_sectors)]
if selected_setups:
    df_filtered = df_filtered[df_filtered['Setup'].isin(selected_setups)]
if volume_spike_filter:
    df_filtered = df_filtered[df_filtered['Vol_Spike'] == True]
if near_52w_high_filter:
    df_filtered = df_filtered[df_filtered['Near_52W_High'] == True]

df_filtered = df_filtered.sort_values(by=['Vol_Spike', 'Return_1M'], ascending=[False, False])

st.markdown("### 🏆 Top Performing Sectors (1W Avg Return)")
if not sector_perf_avg_results:
    st.warning("No sector performance data available.")
else:
    cols_to_display = min(len(sorted_sector_perf), 5)
    if cols_to_display > 0:
        cols = st.columns(cols_to_display)
        for i, (name, perf) in enumerate(sorted_sector_perf[:cols_to_display]):
            cols[i].metric(label=name, value=f"{perf:.2f} %")
    else:
        st.info("Not enough sector data to display top performers.")


st.markdown(f"### 📈 Filtered Stock Setups ({len(df_filtered)} stocks found)")

if df_filtered.empty:
    st.info("No stocks found matching the selected filters.")
else:
    # Columns for display, Ticker will be index
    display_cols_order = ['Sector', 'Price', 'Return_1D', 'Return_1W', 'Return_1M', 
                          'Setup', 'Vol_Spike', 'Near_52W_High', 'Volume (M)', 'RSI']
    
    # Ensure all display_cols exist in df_filtered to prevent KeyError
    cols_to_show_in_df = [col for col in display_cols_order if col in df_filtered.columns]
    
    # Set Ticker as index for sticky behavior and select columns
    df_display = df_filtered.set_index('Ticker')[cols_to_show_in_df].copy()


    # Apply icons and colors using pandas Styler
    def highlight_setup_and_vol_spike(row_series): # row_series is a Series (a single row)
        # Create a Series of empty strings with the same index as the row
        styles = pd.Series([''] * len(row_series), index=row_series.index)
        if 'Setup' in row_series.index:
            if row_series['Setup'] == "Breakout":
                styles.loc['Setup'] = 'background-color: #a6d96a;' # greenish
            elif row_series['Setup'] == "Retest":
                styles.loc['Setup'] = 'background-color: #fdae61;' # orange
        
        if 'Vol_Spike' in row_series.index and row_series['Vol_Spike']: # Check boolean directly
            styles.loc['Vol_Spike'] = 'background-color: #fee08b;' # yellowish
        return styles

    styler = df_display.style.format({
        'Price': "{:.2f}",
        'Return_1D': "{:.2f} %",
        'Return_1W': "{:.2f} %",
        'Return_1M': "{:.2f} %",
        'Volume (M)': "{:.1f}",
        'RSI': "{:.2f}"
    }, na_rep="-")

    # Icon formatter only for 'Setup' column
    def setup_icon_formatter(val):
        if val == "Breakout": return f"📈 {val}"
        elif val == "Retest": return f"🔁 {val}"
        return val
    
    if 'Setup' in df_display.columns:
        styler = styler.format({'Setup': setup_icon_formatter})

    # Apply row-wise coloring
    styler = styler.apply(highlight_setup_and_vol_spike, axis=1)
    
    # Adjust height dynamically based on number of rows
    df_height = min((len(df_display) + 1) * 35 + 3, 600) # Max height 600px
    st.dataframe(styler, use_container_width=True, height=df_height)

    csv = df_filtered.to_csv(index=True).encode('utf-8') # index=True because Ticker is now index
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"nifty200_momentum_filtered_{datetime.today().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )

# RSI Distribution chart
st.markdown("### 📊 RSI Distribution for Filtered Stocks")
if df_filtered.empty or 'RSI' not in df_filtered.columns or df_filtered['RSI'].dropna().empty:
    st.info("No data available for RSI distribution (no stocks filtered or RSI data missing).")
else:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df_filtered['RSI'].dropna(), bins=20, kde=True, color='purple', ax=ax)
    ax.axvline(70, color='red', linestyle='--', linewidth=1, label='Overbought (70)')
    ax.axvline(30, color='green', linestyle='--', linewidth=1, label='Oversold (30)')
    ax.set_xlabel("RSI")
    ax.set_ylabel("Number of Stocks")
    ax.set_title("RSI Histogram of Filtered Stocks")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")
st.markdown("Disclaimer: This is an informational tool and not financial advice. Always do your own research before investing.")
