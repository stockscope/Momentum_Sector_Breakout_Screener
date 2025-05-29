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
    - **Setup Detection**:
        - 📈 **Breakout**: Close ≥ 98% of 20-day high
        - 🔁 **Retest**: Close ≥ 50 EMA and ≤ 103% of 50 EMA
    - **Filters**:
        - Volume Spike: Volume > 1.5× average 20-day volume (used for sorting, True values at top)
        - Proximity to 52W High is highlighted (shown as a column, Close ≥ 95% of 52W High)
    - **Displayed Metrics**:
        - Price, 1D/1W/1M Returns, Volume, Setup Type, Near 52W High, Volume Spike, RSI
    """)

@st.cache_data(ttl=timedelta(hours=1))
def run_screening_process(today_iso_format):
    csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty200list.csv"
    try:
        df_nifty200 = pd.read_csv(csv_url)
    except Exception as e:
        st.error(f"Error loading NIFTY 200 stock list from CSV: {e}")
        return pd.DataFrame(), {}
        
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
    # To ensure data for 'today_iso_format' is included if available, yfinance 'end' is inclusive for dates.
    # For calculations requiring previous days (e.g. returns), a sufficient history is needed.
    start_date = end_date - timedelta(days=400) # Sufficient for 252-day rolling window

    # Download data
    # Using a placeholder for stock_data in case of download failure
    stock_data_dict = {}
    try:
        data_downloaded = yf.download(tickers, start=start_date, end=end_date, interval='1d', group_by='ticker', auto_adjust=False, progress=False)
        if isinstance(data_downloaded.columns, pd.MultiIndex): # Check if data was downloaded for multiple tickers
             for ticker in tickers:
                try:
                    # Access individual ticker data, handling cases where some tickers might fail
                    # or return only a Series if only one column was downloaded (e.g. if only 'Adj Close')
                    # Check if ticker is in columns first level
                    if ticker in data_downloaded.columns.get_level_values(0):
                        stock_data_dict[ticker] = data_downloaded[ticker]
                except KeyError: # Ticker might not be in downloaded data if it failed
                    pass
        elif not data_downloaded.empty: # Single ticker downloaded
            # This case should not happen if 'tickers' list has more than one item and group_by='ticker'
            # but as a fallback for a single valid ticker in the list:
            if len(tickers) == 1:
                 stock_data_dict[tickers[0]] = data_downloaded

    except Exception as e:
        st.warning(f"Could not download all stock data from yfinance: {e}. Results might be incomplete.")


    results = []
    sector_returns_collector = {}

    for ticker in tickers:
        try:
            if ticker not in stock_data_dict or not isinstance(stock_data_dict[ticker], pd.DataFrame) or stock_data_dict[ticker].empty:
                # st.write(f"No data or insufficient data for {ticker}") # For debugging
                continue
            
            df = stock_data_dict[ticker].copy()
            df.dropna(subset=['Adj Close', 'High', 'Low', 'Open', 'Volume'], inplace=True) # Essential columns
            
            # Ensure enough data for lookbacks (22 for 1-month, 252 for 52-week high)
            # The rolling functions will produce NaNs if not enough data, handled later.
            if len(df) < 22: # Min for 1-month return and 20-day avg vol/high
                continue

            # Indicators
            df['50EMA'] = df['Adj Close'].ewm(span=50, adjust=False).mean()
            df['20D_High'] = df['High'].rolling(window=20, min_periods=1).max() # min_periods=1 to get value even if less than 20 days
            df['52W_High'] = df['High'].rolling(window=252, min_periods=1).max()
            df['Avg_Vol_20D'] = df['Volume'].rolling(window=20, min_periods=1).mean()

            # RSI Calculation
            delta = df['Adj Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            
            # Avoid division by zero for rs
            if loss.iloc[-1] == 0:
                rs = np.inf if gain.iloc[-1] > 0 else 0 # Or handle as per specific RSI definition for zero loss
            else:
                rs = gain.iloc[-1] / loss.iloc[-1]
            
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Ensure latest values of calculated indicators are not NaN
            # Some indicators like 52W_High might be NaN if stock history is short
            # Allow NaNs for 52W_High, but other critical ones must be present
            if df[['50EMA', '20D_High', 'Avg_Vol_20D']].iloc[-1].isnull().any():
                continue
            if pd.isna(df['RSI'].iloc[-1]): # RSI can be NaN if gain and loss are both 0 for 14 periods
                continue


            latest = df.iloc[-1]
            # Ensure enough historical data for returns
            if len(df) < 2: prev_day_adj_close = np.nan
            else: prev_day_adj_close = df['Adj Close'].iloc[-2]

            if len(df) < 6: prev_week_adj_close = np.nan
            else: prev_week_adj_close = df['Adj Close'].iloc[-6] # 5 trading days ago

            if len(df) < 22: month_ago_adj_close = np.nan
            else: month_ago_adj_close = df['Adj Close'].iloc[-22] # Approx 21 trading days ago


            return_1d = ((latest['Adj Close'] - prev_day_adj_close) / prev_day_adj_close) * 100 if pd.notna(prev_day_adj_close) and prev_day_adj_close != 0 else np.nan
            return_1w = ((latest['Adj Close'] - prev_week_adj_close) / prev_week_adj_close) * 100 if pd.notna(prev_week_adj_close) and prev_week_adj_close != 0 else np.nan
            return_1m = ((latest['Adj Close'] - month_ago_adj_close) / month_ago_adj_close) * 100 if pd.notna(month_ago_adj_close) and month_ago_adj_close != 0 else np.nan
            
            sector = sector_map.get(ticker, 'Unknown')
            if pd.notna(return_1w):
                sector_returns_collector.setdefault(sector, []).append(return_1w)

            vol_spike = False
            if pd.notna(latest['Avg_Vol_20D']) and latest['Avg_Vol_20D'] > 0:
                vol_spike = latest['Volume'] > 1.5 * latest['Avg_Vol_20D']
            elif pd.notna(latest['Volume']) and latest['Volume'] > 0: # Fallback if Avg_Vol_20D is 0 but not NaN
                vol_spike = True 


            near_52w_high = False
            if pd.notna(latest['52W_High']) and latest['52W_High'] > 0: # Check 52W_High is not NaN
                near_52w_high = latest['Adj Close'] >= 0.95 * latest['52W_High']

            setup = ""
            # Ensure 20D_High and 50EMA are not NaN for setup detection
            if pd.notna(latest['20D_High']) and latest['Adj Close'] >= 0.98 * latest['20D_High']:
                setup = "Breakout"
            elif pd.notna(latest['50EMA']) and latest['Adj Close'] >= latest['50EMA'] and latest['Adj Close'] <= 1.03 * latest['50EMA']:
                setup = "Retest"
            
            # Skip if essential display values are NaN (Price, Volume, Setup must be meaningful)
            if pd.isna(latest['Adj Close']) or pd.isna(latest['Volume']) or setup == "":
                 continue

            results.append({
                'Ticker': ticker,
                'Sector': sector,
                'Price': round(latest['Adj Close'], 2) if pd.notna(latest['Adj Close']) else np.nan,
                'Return_1D': round(return_1d, 2) if pd.notna(return_1d) else np.nan,
                'Return_1W': round(return_1w, 2) if pd.notna(return_1w) else np.nan,
                'Return_1M': round(return_1m, 2) if pd.notna(return_1m) else np.nan,
                '50EMA': round(latest['50EMA'], 2) if pd.notna(latest['50EMA']) else np.nan,
                '20D_High': round(latest['20D_High'], 2) if pd.notna(latest['20D_High']) else np.nan,
                '52W_High': round(latest['52W_High'], 2) if pd.notna(latest['52W_High']) else np.nan, # Can be NaN
                'Near_52W_High': near_52w_high, # Boolean
                'Setup': setup, # String, can be "" if no setup
                'Volume (M)': round(latest['Volume'] / 1e6, 1) if pd.notna(latest['Volume']) else np.nan,
                'Avg_Vol_20D (M)': round(latest['Avg_Vol_20D'] / 1e6, 1) if pd.notna(latest['Avg_Vol_20D']) else np.nan,
                'Vol_Spike': vol_spike, # Boolean
                'RSI': round(latest['RSI'], 2) if pd.notna(latest['RSI']) else np.nan
            })
        except Exception as e:
            # st.warning(f"Error processing {ticker}: {e}") # For debugging
            continue

    df_all = pd.DataFrame(results)
    # Drop rows where setup is not identified, or essential data for display is missing
    if not df_all.empty:
        df_all.dropna(subset=['Price', 'Setup', 'Ticker'], inplace=True) # Price, Ticker are essential
        df_all = df_all[df_all['Setup'] != ""] # Only keep rows with a detected setup

    avg_sector_perf = {}
    for sector, returns_list in sector_returns_collector.items():
        valid_returns = [r for r in returns_list if pd.notna(r)]
        if valid_returns:
            avg_sector_perf[sector] = np.mean(valid_returns) # Using numpy.mean for robustness

    return df_all, avg_sector_perf

with st.spinner("🔍 Screening in progress... this may take a few minutes"):
    # Use current date for fetching data. yfinance end parameter is inclusive.
    current_day_iso = datetime.today().strftime('%Y-%m-%d')
    df_all_results, sector_perf_avg_results = run_screening_process(current_day_iso)

if df_all_results.empty:
    st.warning("No stock data could be processed or no stocks met the initial criteria. The NIFTY 200 list might be empty, there were issues downloading data, or no setups were found.")
    st.stop()

st.markdown(f"Data processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Data from yfinance up to market close of {current_day_iso} or latest available)")

# Sidebar filters
st.sidebar.header("Filter Stocks")

# Sort sectors by performance for the multiselect options and default
sorted_sector_perf = sorted(sector_perf_avg_results.items(), key=lambda x: x[1], reverse=True)
all_available_sectors = sorted(list(df_all_results['Sector'].unique())) # All sectors present in results

top_5_sector_names = [s[0] for s in sorted_sector_perf[:5]]

selected_sectors = st.sidebar.multiselect(
    "Select Sectors (defaults to top 5 performing)",
    options=all_available_sectors, # Show all sectors from results
    default=top_5_sector_names # Default to top 5
)

setup_options = ['Breakout', 'Retest'] # Assuming these are the only non-empty setup types
selected_setups = st.sidebar.multiselect("Select Setup Type(s)", options=setup_options, default=setup_options)

volume_spike_filter = st.sidebar.checkbox("Show only Volume Spike Stocks", value=False)
near_52w_high_filter = st.sidebar.checkbox("Show only Near 52W High Stocks", value=False)

# Filtering data based on sidebar
df_filtered = df_all_results.copy() # Start with all valid results

if selected_sectors: # Apply sector filter only if any sector is selected
    df_filtered = df_filtered[df_filtered['Sector'].isin(selected_sectors)]
if selected_setups: # Apply setup filter only if any setup type is selected
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
    # Display top 5 or fewer if less than 5 sectors have data
    cols = st.columns(min(len(sorted_sector_perf), 5))
    for i, (name, perf) in enumerate(sorted_sector_perf[:5]):
        cols[i].metric(label=name, value=f"{perf:.2f} %")

st.markdown(f"### 📈 Filtered Stock Setups ({len(df_filtered)} stocks found)")

if df_filtered.empty:
    st.info("No stocks found matching the selected filters.")
else:
    display_cols = ['Ticker', 'Sector', 'Price', 'Return_1D', 'Return_1W', 'Return_1M', 
                    'Setup', 'Vol_Spike', 'Near_52W_High', 'Volume (M)', 'RSI']
    
    # Ensure all display_cols exist in df_filtered to prevent KeyError
    actual_display_cols = [col for col in display_cols if col in df_filtered.columns]
    df_display = df_filtered[actual_display_cols].copy()

    # Apply icons and colors using pandas Styler
    def highlight_setup_and_vol_spike(row):
        styles = [''] * len(row)
        if 'Setup' in row.index:
            if row['Setup'] == "Breakout":
                styles[row.index.get_loc('Setup')] = 'background-color: #a6d96a;' # greenish
            elif row['Setup'] == "Retest":
                styles[row.index.get_loc('Setup')] = 'background-color: #fdae61;' # orange
        
        if 'Vol_Spike' in row.index and row['Vol_Spike']:
            styles[row.index.get_loc('Vol_Spike')] = 'background-color: #fee08b;' # yellowish
        return styles

    styler = df_display.style.format({
        'Return_1D': "{:.2f} %",
        'Return_1W': "{:.2f} %",
        'Return_1M': "{:.2f} %",
        'Price': "{:.2f}",
        'RSI': "{:.2f}",
        'Volume (M)': "{:.1f}"
    }, na_rep="-") # Represent NaN as "-"

    # Add icons in Setup and Vol_Spike columns
    def icon_formatter(val, col_name):
        if col_name == 'Setup':
            if val == "Breakout": return f"📈 {val}"
            elif val == "Retest": return f"🔁 {val}"
            return val
        elif col_name == 'Vol_Spike':
            return "🔥 Yes" if val else "No"
        elif col_name == 'Near_52W_High':
            return "⭐ Yes" if val else "No"
        return val

    formatted_cols = {}
    if 'Setup' in df_display.columns:
        formatted_cols['Setup'] = lambda x: icon_formatter(x, 'Setup')
    if 'Vol_Spike' in df_display.columns:
        formatted_cols['Vol_Spike'] = lambda x: icon_formatter(x, 'Vol_Spike')
    if 'Near_52W_High' in df_display.columns:
         formatted_cols['Near_52W_High'] = lambda x: icon_formatter(x, 'Near_52W_High')
    
    if formatted_cols:
        styler = styler.format(formatted_cols)

    # Apply row-wise coloring for Setup and Vol_Spike
    styler = styler.apply(highlight_setup_and_vol_spike, axis=1)

    st.dataframe(styler, use_container_width=True, height=(len(df_display) + 1) * 35 + 3)


    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"nifty200_momentum_filtered_{datetime.today().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )

# Additional RSI Distribution chart
st.markdown("### 📊 RSI Distribution for Filtered Stocks")
if df_filtered.empty or 'RSI' not in df_filtered.columns or df_filtered['RSI'].dropna().empty:
    st.info("No data available for RSI distribution (no stocks filtered or RSI data missing).")
else:
    fig, ax = plt.subplots(figsize=(10, 4)) # Reduced height slightly
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
