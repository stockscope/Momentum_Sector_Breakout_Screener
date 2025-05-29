import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np

# Ensure Matplotlib's minus sign is rendered correctly
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide")
st.title("📊 Momentum Sector Breakout Screener (NIFTY 500)")
st.markdown("Identifies breakout or retest setups in top-performing sectors based on trend, volume, and returns.")

with st.expander("🧠 **Screening Criteria Used**", expanded=True):
    st.markdown("""
    - **Universe**: NIFTY 500 stocks
    - **Top Sectors**: Stocks shown are from the **top 5 performing sectors** based on average **1-week return**.
    - **Setup Detection (Stricter & New)**:
        - 🌟 **Breakout 52w**: Close ≥ **99%** of 52-week High (New, takes precedence)
        - 📈 **Breakout**: Close ≥ **99%** of 20-day High
        - 🔁 **Retest**: Close ≥ 50 EMA and ≤ **102%** of 50 EMA
    - **Filters Applied Programmatically**:
        - Stocks must belong to one of the top 5 performing sectors.
    - **Displayed Information (Not User Filters)**:
        - Volume Spike: Volume > 1.5× average 20-day volume (used for sorting, True values at top)
        - Proximity to 52W High is highlighted (shown as a column, Close ≥ 95% of 52W High)
    - **Displayed Metrics**:
        - Price, 1D/1W/1M Returns, Volume (M), RSI (all numeric to 2 decimal places), Setup Type, Near 52W High, Volume Spike
    """)

# --- Refactored Data Fetching and Processing Functions ---

@st.cache_data(ttl=timedelta(days=1), show_spinner=False)
def load_nifty_list_and_map(list_name="nifty500"):
    if list_name == "nifty500":
        csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty500list.csv"
    elif list_name == "nifty200":
        csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty200list.csv"
    else:
        st.error(f"Unknown stock list: {list_name}")
        return [], {}, {}
        
    try:
        df_nifty = pd.read_csv(csv_url)
    except Exception as e:
        st.error(f"Error loading NIFTY {list_name.upper()} stock list from CSV: {e}")
        return [], {}, {}
        
    df_nifty.columns = df_nifty.columns.str.strip()
    df_nifty = df_nifty[~df_nifty['Symbol'].str.contains("DUMMY", na=False)]
    df_nifty['Symbol'] = df_nifty['Symbol'].str.strip()
    df_nifty['Industry'] = df_nifty['Industry'].fillna('Unknown')
    df_nifty['Ticker'] = df_nifty['Symbol'] + ".NS"

    tickers = df_nifty['Ticker'].tolist()
    sector_map = dict(zip(df_nifty['Ticker'], df_nifty['Industry']))
    return tickers, sector_map, df_nifty

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
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
        timeout=60 
    )
    
    stock_data_processed = {}
    if stock_data_downloaded.empty:
        return {}

    if isinstance(stock_data_downloaded.columns, pd.MultiIndex):
        for ticker in tickers_list:
            try:
                if ticker in stock_data_downloaded and isinstance(stock_data_downloaded[ticker], pd.DataFrame):
                    stock_data_processed[ticker] = stock_data_downloaded[ticker]
            except KeyError:
                pass 
    elif len(tickers_list) == 1 and isinstance(stock_data_downloaded, pd.DataFrame):
        stock_data_processed[tickers_list[0]] = stock_data_downloaded
        
    return stock_data_processed

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
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
            if len(df) < 22: 
                continue

            df['50EMA'] = df['Adj Close'].ewm(span=50, adjust=False).mean()
            df['20D_High'] = df['High'].rolling(window=20, min_periods=1).max()
            df['52W_High'] = df['High'].rolling(window=252, min_periods=1).max() 
            df['Avg_Vol_20D'] = df['Volume'].rolling(window=20, min_periods=1).mean()

            delta = df['Adj Close'].diff()
            avg_gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean().iloc[-1]
            avg_loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean().iloc[-1]

            rs_val = np.nan
            latest_rsi = np.nan

            if pd.notna(avg_gain) and pd.notna(avg_loss):
                if avg_loss == 0:
                    if avg_gain > 0: rs_val = np.inf
                else: rs_val = avg_gain / avg_loss
            
            if pd.notna(rs_val):
                if rs_val == np.inf: latest_rsi = 100.0
                else: latest_rsi = 100 - (100 / (1 + rs_val))
            
            if 'RSI' not in df.columns: df['RSI'] = np.nan 
            df.loc[df.index[-1], 'RSI'] = latest_rsi

            if df[['50EMA', '20D_High', 'Avg_Vol_20D']].iloc[-1].isnull().any():
                continue

            latest = df.iloc[-1]
            
            prev_day_adj_close = df['Adj Close'].iloc[-2] if len(df) >= 2 else np.nan
            prev_week_adj_close = df['Adj Close'].iloc[-6] if len(df) >= 6 else np.nan
            month_ago_adj_close = df['Adj Close'].iloc[-22] if len(df) >= 22 else np.nan

            return_1d = ((latest['Adj Close'] - prev_day_adj_close) / prev_day_adj_close) * 100 if pd.notna(prev_day_adj_close) and prev_day_adj_close != 0 else np.nan
            return_1w = ((latest['Adj Close'] - prev_week_adj_close) / prev_week_adj_close) * 100 if pd.notna(prev_week_adj_close) and prev_week_adj_close != 0 else np.nan
            return_1m = ((latest['Adj Close'] - month_ago_adj_close) / month_ago_adj_close) * 100 if pd.notna(month_ago_adj_close) and month_ago_adj_close != 0 else np.nan
            
            sector = sector_map_dict.get(ticker, 'Unknown')
            if pd.notna(return_1w):
                sector_returns_collector.setdefault(sector, []).append(return_1w)

            vol_spike = False
            if pd.notna(latest['Avg_Vol_20D']) and latest['Avg_Vol_20D'] > 0:
                vol_spike = latest['Volume'] > 1.5 * latest['Avg_Vol_20D']
            elif pd.notna(latest['Volume']) and latest['Volume'] > 0 :
                vol_spike = True 

            near_52w_high_info = False
            if pd.notna(latest['52W_High']) and latest['52W_High'] > 0:
                near_52w_high_info = latest['Adj Close'] >= 0.95 * latest['52W_High']

            setup = ""
            if pd.notna(latest['52W_High']) and latest['52W_High'] > 0 and latest['Adj Close'] >= 0.99 * latest['52W_High']:
                setup = "Breakout 52w"
            elif pd.notna(latest['20D_High']) and latest['Adj Close'] >= 0.99 * latest['20D_High']:
                setup = "Breakout"
            elif pd.notna(latest['50EMA']) and latest['Adj Close'] >= latest['50EMA'] and latest['Adj Close'] <= 1.02 * latest['50EMA']:
                setup = "Retest"
            
            if pd.isna(latest['Adj Close']) or pd.isna(latest['Volume']) or setup == "":
                 continue

            results.append({
                'Ticker': ticker, 'Sector': sector,
                'Price': round(latest['Adj Close'], 2) if pd.notna(latest['Adj Close']) else np.nan,
                'Return_1D': round(return_1d, 2) if pd.notna(return_1d) else np.nan,
                'Return_1W': round(return_1w, 2) if pd.notna(return_1w) else np.nan,
                'Return_1M': round(return_1m, 2) if pd.notna(return_1m) else np.nan,
                '50EMA': round(latest['50EMA'], 2) if pd.notna(latest['50EMA']) else np.nan,
                '20D_High': round(latest['20D_High'], 2) if pd.notna(latest['20D_High']) else np.nan,
                '52W_High': round(latest['52W_High'], 2) if pd.notna(latest['52W_High']) else np.nan,
                'Near_52W_High': near_52w_high_info,
                'Setup': setup,
                'Volume (M)': round(latest['Volume'] / 1e6, 2) if pd.notna(latest['Volume']) else np.nan,
                'Avg_Vol_20D (M)': round(latest['Avg_Vol_20D'] / 1e6, 2) if pd.notna(latest['Avg_Vol_20D']) else np.nan,
                'Vol_Spike': vol_spike,
                'RSI': round(latest['RSI'], 2) if pd.notna(latest['RSI']) else np.nan
            })
        except Exception: # Catch all exceptions for a single ticker to continue with others
            # For debugging, you might want to log the ticker and the specific error:
            # import traceback
            # st.sidebar.warning(f"Error processing {ticker}: {traceback.format_exc()}")
            continue

    df_all = pd.DataFrame(results)
    if not df_all.empty:
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
fetch_end_date = current_day_iso
fetch_start_date = (datetime.today() - timedelta(days=400)).strftime('%Y-%m-%d')

with st.spinner("📜 Loading NIFTY 500 list..."):
    tickers, sector_map, _ = load_nifty_list_and_map(list_name="nifty500")

if not tickers:
    st.error("NIFTY 500 stock list could not be loaded. Cannot proceed.")
    st.stop()

with st.spinner("📥 Fetching market data from yfinance... (this may take a few minutes for NIFTY 500)"):
    downloaded_stock_data = fetch_stock_data_from_yfinance(tuple(tickers), fetch_start_date, fetch_end_date)

if not downloaded_stock_data: # If yf.download returns an empty dict or fails significantly
    st.warning("No stock data could be downloaded from yfinance. Results might be incomplete or empty.")
    # Allow to proceed, analyze_stocks_and_sectors will handle empty downloaded_stock_data

with st.spinner("⚙️ Processing data and identifying setups..."):
    df_all_results, sector_perf_avg_results = analyze_stocks_and_sectors(
        downloaded_stock_data, tuple(tickers), sector_map, current_day_iso
    )

if df_all_results.empty:
    st.warning("No stock data could be processed or no stocks met the screening criteria. This could be due to data download issues, no stocks matching the setup conditions, or market holidays.")
    st.stop()

st.markdown(f"Data processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Market data up to {current_day_iso} or latest available)")

sorted_sector_perf = sorted(sector_perf_avg_results.items(), key=lambda x: x[1], reverse=True)
top_5_sector_names = [s[0] for s in sorted_sector_perf[:5]]

df_filtered = df_all_results[df_all_results['Sector'].isin(top_5_sector_names)]
setup_order = ["Breakout 52w", "Breakout", "Retest"]
df_filtered['Setup_Sort'] = pd.Categorical(df_filtered['Setup'], categories=setup_order, ordered=True)
df_filtered = df_filtered.sort_values(by=['Setup_Sort', 'Vol_Spike', 'Return_1M'], ascending=[True, False, False])
df_filtered = df_filtered.drop(columns=['Setup_Sort'])

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

st.markdown(f"### 📈 Stock Setups from Top Performing Sectors ({len(df_filtered)} stocks found)")

if df_filtered.empty:
    st.info("No stocks found from the top performing sectors matching the setup criteria.")
else:
    display_cols_order = ['Sector', 'Price', 'Return_1D', 'Return_1W', 'Return_1M', 
                          'Setup', 'Vol_Spike', 'Near_52W_High', 'Volume (M)', 'RSI']
    
    cols_to_show_in_df = [col for col in display_cols_order if col in df_filtered.columns]
    df_display = df_filtered.set_index('Ticker')[cols_to_show_in_df].copy()

    def highlight_setup_and_vol_spike(row_series):
        styles = pd.Series([''] * len(row_series), index=row_series.index)
        if 'Setup' in row_series.index:
            if row_series['Setup'] == "Breakout 52w":
                styles.loc['Setup'] = 'background-color: #ffeda0;' 
            elif row_series['Setup'] == "Breakout":
                styles.loc['Setup'] = 'background-color: #a6d96a;'
            elif row_series['Setup'] == "Retest":
                styles.loc['Setup'] = 'background-color: #fdae61;'
        
        if 'Vol_Spike' in row_series.index and row_series['Vol_Spike']:
            styles.loc['Vol_Spike'] = 'background-color: #fee08b;'
        return styles

    styler = df_display.style.format({
        'Price': "{:.2f}",
        'Return_1D': "{:.2f}%", # No space before %
        'Return_1W': "{:.2f}%", # No space before %
        'Return_1M': "{:.2f}%", # No space before %
        'Volume (M)': "{:.2f}",
        'RSI': "{:.2f}"
    }, na_rep="-", precision=2) # Global precision hint for any other floats

    def setup_icon_formatter(val):
        if val == "Breakout 52w": return f"🌟 {val}"
        elif val == "Breakout": return f"📈 {val}"
        elif val == "Retest": return f"🔁 {val}"
        return val
    
    if 'Setup' in df_display.columns:
        styler = styler.format({'Setup': setup_icon_formatter})

    styler = styler.apply(highlight_setup_and_vol_spike, axis=1)
    
    df_height = min((len(df_display) + 1) * 35 + 3, 600) 
    st.dataframe(styler, use_container_width=True, height=df_height)

    csv = df_filtered.to_csv(index=False).encode('utf-8') 
    st.download_button(
        label="📥 Download Screened Data as CSV",
        data=csv,
        file_name=f"nifty500_top_sectors_screened_{datetime.today().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )

st.markdown("### 📊 RSI Distribution for Screened Stocks")
if df_filtered.empty or 'RSI' not in df_filtered.columns or df_filtered['RSI'].dropna().empty:
    st.info("No data available for RSI distribution (no stocks screened or RSI data missing).")
else:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df_filtered['RSI'].dropna(), bins=20, kde=True, color='purple', ax=ax)
    ax.axvline(70, color='red', linestyle='--', linewidth=1, label='Overbought (70)')
    ax.axvline(30, color='green', linestyle='--', linewidth=1, label='Oversold (30)')
    ax.set_xlabel("RSI")
    ax.set_ylabel("Number of Stocks")
    ax.set_title("RSI Histogram of Screened Stocks")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")
st.markdown("Disclaimer: This is an informational tool and not financial advice. Always do your own research before investing.")
