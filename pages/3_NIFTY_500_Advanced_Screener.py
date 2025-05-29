import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import io

matplotlib.rcParams['axes.unicode_minus'] = False

# --- DataFrame to Excel ---
def to_excel(df):
    output = io.BytesIO()
    df_to_export = df.copy()
    if isinstance(df, pd.io.formats.style.Styler):
        df_to_export = df.data
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_to_export.to_excel(writer, index=False, sheet_name='Screenings')
    processed_data = output.getvalue()
    return processed_data

st.set_page_config(layout="wide")
# Updated Title
st.title("📊 3_NIFTY_500_Core_Sector_Momentum_Screener")
st.markdown("Identifies Breakout, Retest, or Pullback setups in the top 5 performing NIFTY 500 sectors.")

with st.expander("🧠 **Screening Criteria Used (Simplified)**", expanded=True):
    st.markdown("""
    - **Universe**: NIFTY 500 stocks
    - **Top Sectors**: Focus on stocks from the top 5 performing sectors (based on average 1-week return).
    - **Setup Detection (Prioritized Order)**:
        - 📈 **Breakout**: Close ≥ 98% of 20-Day High **OR** Close ≥ 98% of 52-Week High.
        - 🔁 **Retest**: Price within ±3% of 50 EMA (i.e., Close between 97% and 103% of 50 EMA).
        - ↩️ **Pullback**: Close within ±3% of 20 EMA (i.e., Close between 97% and 103% of 20 EMA), and not a Breakout or Retest.
    - **Key Filters Applied**:
        - Stock must belong to one of the top 5 performing sectors.
        - **Basic EMA Trend**: Close > 50 EMA.
        - **Relative Volume (RVOL)**: Current Volume / 20-Day Avg Volume > 1.0 (volume is at least average).
    - **Sorting**: Primarily by Volume Spike (Current Volume > 1.5 × 20-Day Avg Volume; displayed as True/False), then by 1M Return.
    - **Display & Features**:
        - Results table color-coded by setup type. All numerical values rounded to 2 decimal places.
        - Data export to Excel enabled.
    - **Displayed Metrics**:
        - Ticker, Sector, Price, Setup Type, Returns (1D, 1W, 1M), RVOL, Volume Spike (True/False), Near 52W High, Volume (M).
    """)

@st.cache_data(ttl=timedelta(hours=1), show_spinner="Fetching and processing NIFTY 500 data...")
def run_simplified_screening_process(today_iso_format_utc):
    csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty500list.csv"
    df_nifty500 = pd.read_csv(csv_url)
    df_nifty500.columns = df_nifty500.columns.str.strip()
    df_nifty500 = df_nifty500[~df_nifty500['Symbol'].str.contains("DUMMY", na=False)]
    df_nifty500['Symbol'] = df_nifty500['Symbol'].str.strip()
    df_nifty500['Industry'] = df_nifty500['Industry'].fillna('Unknown')
    df_nifty500['Ticker'] = df_nifty500['Symbol'] + ".NS"

    tickers = df_nifty500['Ticker'].tolist()
    sector_map = dict(zip(df_nifty500['Ticker'], df_nifty500['Industry']))

    if not tickers:
        return pd.DataFrame(), {}

    end_date_dt = datetime.fromisoformat(today_iso_format_utc.replace('Z', '+00:00'))
    start_date_dt = end_date_dt - timedelta(days=300)

    stock_data = yf.download(tickers, start=start_date_dt, end=end_date_dt, interval='1d',
                             group_by='ticker', auto_adjust=False, progress=False, timeout=60)

    results = []
    sector_returns_collector = {}
    min_data_length = 60

    for ticker in tickers:
        try:
            if ticker not in stock_data or not isinstance(stock_data[ticker], pd.DataFrame) or stock_data[ticker].empty:
                continue

            df = stock_data[ticker].copy()
            df.dropna(subset=['Adj Close', 'High', 'Low', 'Open', 'Volume'], inplace=True)

            if len(df) < min_data_length:
                continue

            df['20EMA'] = df['Adj Close'].ewm(span=20, adjust=False).mean()
            df['50EMA'] = df['Adj Close'].ewm(span=50, adjust=False).mean()
            df['20D_High'] = df['High'].rolling(window=20).max()
            df['52W_High'] = df['High'].rolling(window=252, min_periods=20).max()
            df['Avg_Vol_20D'] = df['Volume'].rolling(window=20).mean()

            required_cols_check = ['20EMA', '50EMA', '20D_High', 'Avg_Vol_20D']
            if df[required_cols_check].iloc[-1].isnull().any():
                continue

            latest = df.iloc[-1]
            
            prev_day = df.iloc[-2] if len(df) >= 2 else latest
            prev_week = df.iloc[-6] if len(df) >= 6 else latest
            month_ago = df.iloc[-22] if len(df) >= 22 else latest

            if any(val == 0 or pd.isna(val) for val in [prev_day['Adj Close'], prev_week['Adj Close'], month_ago['Adj Close'], latest['Adj Close']]):
                continue
            
            return_1d = ((latest['Adj Close'] - prev_day['Adj Close']) / prev_day['Adj Close']) * 100 if prev_day['Adj Close'] != 0 else 0
            return_1w = ((latest['Adj Close'] - prev_week['Adj Close']) / prev_week['Adj Close']) * 100 if prev_week['Adj Close'] != 0 else 0
            return_1m = ((latest['Adj Close'] - month_ago['Adj Close']) / month_ago['Adj Close']) * 100 if month_ago['Adj Close'] != 0 else 0
            
            sector = sector_map.get(ticker, 'Unknown')
            if pd.notna(return_1w):
                sector_returns_collector.setdefault(sector, []).append(return_1w)

            rvol = 0.0
            if pd.notna(latest['Avg_Vol_20D']) and latest['Avg_Vol_20D'] > 0:
                rvol = latest['Volume'] / latest['Avg_Vol_20D']
            vol_spike = bool(rvol > 1.5) # Ensure it's a Python boolean

            ema_trend_ok = latest['Adj Close'] > latest['50EMA']
            
            setup = ""
            is_breakout = (latest['Adj Close'] >= 0.98 * latest['20D_High']) or \
                          (pd.notna(latest['52W_High']) and latest['Adj Close'] >= 0.98 * latest['52W_High'])
            
            is_retest_50ema = (latest['Adj Close'] >= 0.97 * latest['50EMA']) and \
                              (latest['Adj Close'] <= 1.03 * latest['50EMA'])
            
            is_pullback_20ema = (latest['Adj Close'] >= 0.97 * latest['20EMA']) and \
                                (latest['Adj Close'] <= 1.03 * latest['20EMA'])

            if is_breakout:
                setup = "Breakout"
            elif is_retest_50ema:
                setup = "Retest"
            elif is_pullback_20ema:
                setup = "Pullback"

            near_52w_high = bool(pd.notna(latest['52W_High']) and latest['Adj Close'] >= 0.95 * latest['52W_High']) # Python boolean

            results.append({
                'Ticker': ticker, 'Sector': sector, 
                'Price': round(latest['Adj Close'], 2),
                'Return_1D': round(return_1d, 2), 
                'Return_1W': round(return_1w, 2), 
                'Return_1M': round(return_1m, 2),
                'Setup': setup,
                'RVOL': round(rvol, 2),
                'Volume (M)': round(latest['Volume'] / 1e6, 2), # Changed to 2 decimal places
                'Vol_Spike': vol_spike, # Already a boolean
                'Near_52W_High': near_52w_high, # Already a boolean
                '50EMA': round(latest['50EMA'], 2), 
                '20D_High': round(latest['20D_High'], 2),
                '52W_High': round(latest['52W_High'], 2) if pd.notna(latest['52W_High']) else 'N/A',
                '_filter_rvol_ok': (rvol > 1.0),
                '_filter_ema_trend_ok': ema_trend_ok
            })
        except Exception as e:
            # st.sidebar.warning(f"Error processing {ticker}: {str(e)}")
            continue

    df_all = pd.DataFrame(results)
    avg_sector_perf = {
        k: sum(v)/len(v) for k, v in sector_returns_collector.items() if v and len(v) > 0
    }
    return df_all, avg_sector_perf

# --- Main App Logic ---
current_day_iso_utc = datetime.now(timezone.utc).isoformat()
df_all_results, sector_performance_avg = run_simplified_screening_process(current_day_iso_utc)

st.markdown(f"Data last fetched (UTC): `{current_day_iso_utc}`. Displayed at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}`")

if df_all_results.empty:
    st.warning("No stock data could be processed or no stocks met initial data requirements. Check NIFTY 500 list or data download.")
    st.stop()

st.markdown("### 🏆 Top Performing Sectors (1W Avg Return)")
top_5_sector_names = []
if sector_performance_avg:
    top_sectors_list = sorted(sector_performance_avg.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5_sector_names = [s[0] for s in top_sectors_list]
    if top_sectors_list:
        cols = st.columns(len(top_sectors_list))
        for i, (name, perf) in enumerate(top_sectors_list):
            cols[i].metric(label=name, value=f"{perf:.2f} %") # Sector perf already 2 decimals
    else:
        st.info("Not enough data to rank sectors.")
else:
    st.warning("No sector performance data available.")

df_filtered = df_all_results[
    (df_all_results['Sector'].isin(top_5_sector_names)) &
    (df_all_results['Setup'].isin(['Breakout', 'Retest', 'Pullback'])) &
    (df_all_results['_filter_rvol_ok']) &
    (df_all_results['_filter_ema_trend_ok'])
]
df_filtered = df_filtered.sort_values(by=['Vol_Spike', 'Return_1M'], ascending=[False, False])

st.markdown("### 📈 Filtered Stock Setups in Leading Sectors")

def highlight_setup_cell(val):
    if val == 'Breakout': return 'background-color: lightgreen; color: black;'
    elif val == 'Retest': return 'background-color: lightblue; color: black;'
    elif val == 'Pullback': return 'background-color: orange; color: black;'
    return ''

if df_filtered.empty:
    st.info("No stocks found matching all screening criteria. Try again later or check data sources if persistent.")
else:
    display_cols = ['Ticker', 'Sector', 'Price', 'Setup', 'Return_1D', 'Return_1W', 'Return_1M',
                    'RVOL', 'Vol_Spike', 'Near_52W_High', 'Volume (M)']
    df_display = df_filtered[display_cols].head(30).copy()
    
    # Format numerical columns to 2 decimal places for display where appropriate
    # Note: Price, Returns, RVOL, Volume (M) are already rounded during DataFrame creation.
    # This is more for ensuring the Styler object respects it if needed, but usually it does.
    float_cols_to_format = ['Price', 'Return_1D', 'Return_1W', 'Return_1M', 'RVOL', 'Volume (M)']
    format_dict = {col: '{:.2f}' for col in float_cols_to_format}
    
    df_styled = df_display.style.applymap(highlight_setup_cell, subset=['Setup'])\
                              .format(format_dict, na_rep='N/A') # Apply number formatting

    st.dataframe(df_styled, use_container_width=True, hide_index=True)

    excel_export_df = df_filtered[display_cols].copy()
    excel_data = to_excel(excel_export_df) # to_excel uses the already rounded df
    st.download_button(
        label="📥 Download Filtered Results as Excel",
        data=excel_data,
        file_name=f"core_momentum_screener_results_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.markdown("### 📊 Return Distribution of Filtered Stocks")
if df_filtered.empty:
    st.info("No stocks available to plot return distribution.")
else:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=df_filtered, x='Return_1W', kde=True, color='dodgerblue', label='1W Return', ax=ax, bins=15)
    sns.histplot(data=df_filtered, x='Return_1M', kde=True, color='mediumseagreen', label='1M Return', ax=ax, bins=15)
    plt.legend(); plt.title("Histogram of Weekly & Monthly Returns (Filtered Stocks)", fontsize=15)
    plt.xlabel("Return (%)", fontsize=12); plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7); sns.despine(); st.pyplot(fig)

st.markdown("---")
st.markdown("Disclaimer: This is an informational tool and not financial advice. Always do your own research. Ensure `openpyxl` is installed for Excel export.")
