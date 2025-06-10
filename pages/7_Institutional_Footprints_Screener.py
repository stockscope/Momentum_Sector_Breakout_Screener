# pages/10_Institutional_Footprints_Screener.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import io

st.set_page_config(
    page_title="Institutional Footprints Screener - StockScopePro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Institutional Footprints Screener")
st.markdown("Identifies stocks showing signs that *might* indicate institutional interest or accumulation, based on volume and price action.")
st.warning("Note: These are indirect indicators. True institutional activity is not publicly disclosed in real-time.")


# --- User Selections (Sidebar) ---
st.sidebar.header("Screener Settings")
index_choice_inst = st.sidebar.selectbox(
    "Select Index Universe:", 
    ["NIFTY 500", "NIFTY 200"], 
    index=0, 
    key="inst_index_choice"
)
list_to_load_inst = "nifty500" if index_choice_inst == "NIFTY 500" else "nifty200"

vol_spike_factor_inst = st.sidebar.slider(
    "Volume Spike Factor (Current Vol / AvgVol20D):", 
    1.5, 10.0, 2.5, 0.1, 
    key="inst_vol_spike_factor",
    help="Current volume must be at least this many times its 20-day average."
)
min_market_cap_cr_inst = st.sidebar.number_input(
    "Minimum Market Cap (Crores):", 
    min_value=0, value=1000, step=100, 
    key="inst_min_mcap",
    help="Filters for companies of a certain size, often preferred by institutions."
)
min_price_inst = st.sidebar.number_input(
    "Minimum Stock Price (₹):",
    min_value=0.0, value=50.0, step=1.0,
    key="inst_min_price",
    help="Filters out very low-priced stocks."
)
obv_sma_period_inst = st.sidebar.slider(
    "OBV Smoothing Period (SMA of OBV):",
    5, 50, 20, 1,
    key="inst_obv_sma",
    help="On-Balance Volume should be above its own Simple Moving Average of this period."
)
# Optional: Filter for price action on volume spike day
close_near_high_pct_inst = st.sidebar.slider(
    "On Spike Day: Close within X% of Day's High (0 to disable):",
    0.0, 10.0, 2.0, 0.1, format="%.1f%%",
    key="inst_close_high_pct",
    help="0% means close must be the high. 2% means close is within top 2% of day's range from high."
)


with st.expander("📜 **Screening Logic Explained**", expanded=True):
    st.markdown(f"""
    This screener looks for stocks exhibiting characteristics often associated with institutional buying or accumulation:

    1.  **Significant Volume Spike:**
        - Current day's trading volume is at least **{vol_spike_factor_inst} times** its 20-day average volume.
    2.  **Bullish Price Action on Volume Spike Day** (if Close near High % > 0):
        - The stock closes near the day's high (within **{close_near_high_pct_inst}%** of the high). This suggests buying pressure absorbed selling.
        - *Alternatively, one could check if the price change on the spike day was positive.*
    3.  **Accumulation Indicated by On-Balance Volume (OBV):**
        - The current OBV is greater than its **{obv_sma_period_inst}-day Simple Moving Average**, suggesting buying pressure is outweighing selling pressure over time.
    4.  **Minimum Size & Price:**
        - Market Capitalization is at least **₹{min_market_cap_cr_inst} Crores**.
        - Current closing price is at least **₹{min_price_inst}**.
    
    *These are indirect signs. Institutional buying can be subtle or occur in dark pools. This screener highlights potential candidates for further research.*
    """)

# --- Helper Functions ---
@st.cache_data(ttl=timedelta(days=1), show_spinner=False)
def load_nifty_list_inst(list_name): # Renamed
    # ... (same as load_nifty_list from previous script) ...
    if list_name == "nifty500": csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty500list.csv"
    elif list_name == "nifty200": csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty200list.csv"
    else: st.error(f"Unknown list: {list_name}"); return []
    try:
        df = pd.read_csv(csv_url); df.columns=df.columns.str.strip()
        df = df[~df['Symbol'].str.contains("DUMMY",na=False)]; df['Symbol']=df['Symbol'].str.strip()
        df['Ticker'] = df['Symbol'] + ".NS"; return df['Ticker'].tolist()
    except Exception as e: st.error(f"Error loading {list_name}: {e}"); return []

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def fetch_stock_data_inst(tickers_tuple, days_history=100): # Renamed
    tickers_list = list(tickers_tuple); end_dt=datetime.today()
    # Need enough for OBV SMA and AvgVol20D (e.g., 50 days + buffer)
    start_dt=end_dt-timedelta(days=days_history + 30) 
    if not tickers_list: return {}
    try:
        end_dt_fetch = (end_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        data=yf.download(tickers_list, start=start_dt.strftime('%Y-%m-%d'),end=end_dt_fetch,
                           interval='1d',group_by='ticker',auto_adjust=False,progress=False,timeout=120)
        processed={}; 
        if data.empty: return {}
        actual_end_ts = pd.Timestamp(end_dt.strftime('%Y-%m-%d'))
        if isinstance(data.columns,pd.MultiIndex):
            for t in tickers_list:
                if t in data and isinstance(data[t],pd.DataFrame) and not data[t].empty:
                    df_t=data[t].copy(); df_t.index=pd.to_datetime(df_t.index)
                    df_t = df_t[df_t.index <= actual_end_ts]
                    if not df_t.empty: processed[t]=df_t.ffill(limit=2).bfill(limit=2)
        elif len(tickers_list)==1 and isinstance(data,pd.DataFrame) and not data.empty:
             df_s=data.copy(); df_s.index=pd.to_datetime(df_s.index)
             df_s = df_s[df_s.index <= actual_end_ts]
             if not df_s.empty: processed[tickers_list[0]]=df_s.ffill(limit=2).bfill(limit=2)
        return processed
    except Exception as e: st.sidebar.warning(f"Inst. Data DL Error: {str(e)[:50]}"); return {}

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def get_stock_info_inst(ticker_str): # Renamed
    try:
        stock = yf.Ticker(ticker_str)
        info = stock.info
        # Focus on marketCap, maybe institutionalOwnership if desired (though lagging)
        return {
            'marketCap': info.get('marketCap'),
            'industry': info.get('industry', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            # 'heldPercentInstitutions': info.get('heldPercentInstitutions') # Lagging quarterly data
        }
    except Exception:
        return {}

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def run_institutional_footprints_screener(
    tickers_list_tuple, 
    vol_spike_factor, 
    min_mcap_cr, 
    min_price,
    obv_sma_period,
    close_high_pct # If 0, this condition is skipped
):
    screened_stocks_data = []
    # History needed for 20-day Avg Vol and OBV_SMA
    days_data = max(60, obv_sma_period + 20) 
    hist_data_batch = fetch_stock_data_inst(tickers_list_tuple, days_history=days_data) 

    for ticker in tickers_list_tuple:
        try:
            # 1. Get Stock Info (for Market Cap)
            info = get_stock_info_inst(ticker)
            market_cap = info.get('marketCap')
            if market_cap is None or (market_cap / 1e7) < min_mcap_cr: # Convert to Crores
                continue

            # 2. Get Historical Data
            if ticker not in hist_data_batch or hist_data_batch[ticker].empty:
                continue
            df = hist_data_batch[ticker].copy()
            df.dropna(subset=['Adj Close', 'High', 'Low', 'Volume'], inplace=True)

            if len(df) < max(21, obv_sma_period + 1): # Min for AvgVol20D and OBV SMA
                continue

            # --- Calculate Indicators ---
            latest = df.iloc[-1]
            latest_close = latest['Adj Close']
            latest_high = latest['High']
            latest_low = latest['Low']
            latest_volume = latest['Volume']

            # Minimum Price Filter
            if latest_close < min_price:
                continue

            df['AvgVol20D'] = df['Volume'].rolling(window=20, min_periods=15).mean()
            avg_vol_20d = df['AvgVol20D'].iloc[-1]
            if pd.isna(avg_vol_20d) or avg_vol_20d == 0: continue

            # Condition 1: Volume Spike
            if latest_volume < vol_spike_factor * avg_vol_20d:
                continue
            
            # Condition 2: Price Action on Volume Spike Day (Optional)
            if close_high_pct > 0:
                if latest_high == latest_low: # Avoid division by zero if H=L
                    if latest_close != latest_high: # Only fails if close isn't also H=L
                        continue
                elif ((latest_high - latest_close) / (latest_high - latest_low)) * 100 > close_high_pct:
                    continue # Close is too far from high

            # Condition 3: On-Balance Volume (OBV)
            price_diff = df['Adj Close'].diff()
            obv = np.where(price_diff > 0, df['Volume'], 
                  np.where(price_diff < 0, -df['Volume'], 0)).cumsum()
            df['OBV'] = obv
            df[f'OBV_SMA{obv_sma_period}'] = df['OBV'].rolling(window=obv_sma_period, min_periods=max(1, obv_sma_period-5)).mean()
            
            latest_obv = df['OBV'].iloc[-1]
            latest_obv_sma = df[f'OBV_SMA{obv_sma_period}'].iloc[-1]

            if pd.isna(latest_obv) or pd.isna(latest_obv_sma) or latest_obv < latest_obv_sma:
                continue
            
            screened_stocks_data.append({
                'Ticker': ticker,
                'Price': round(latest_close, 2),
                'Industry': info.get('industry', 'N/A'),
                'MCap (Cr)': round(market_cap / 1e7, 2),
                'Volume (M)': round(latest_volume / 1e6, 2),
                'AvgVol20D (M)': round(avg_vol_20d / 1e6, 2),
                'Vol vs Avg': round(latest_volume / avg_vol_20d, 1) if avg_vol_20d > 0 else 0,
                'OBV > SMA': 'Yes'
            })
        except Exception:
            # import traceback
            # st.sidebar.text(f"Error processing {ticker} for Inst: {traceback.format_exc()[:100]}")
            continue
    return pd.DataFrame(screened_stocks_data)

# --- Main App Logic ---
if 'inst_screener_results' not in st.session_state:
    st.session_state.inst_screener_results = pd.DataFrame()

if st.sidebar.button(f"🚀 Run Institutional Footprints Screener on {index_choice_inst}", type="primary", key="run_inst_btn"):
    with st.spinner(f"📜 Loading {index_choice_inst} list..."):
        tickers_to_scan_inst = load_nifty_list_inst(list_name=list_to_load_inst)

    if not tickers_to_scan_inst: 
        st.error(f"Failed to load {index_choice_inst} stock list.")
    else:
        with st.spinner(f"🔎 Screening {len(tickers_to_scan_inst)} stocks from {index_choice_inst}..."):
            df_results_inst = run_institutional_footprints_screener(
                tuple(tickers_to_scan_inst),
                vol_spike_factor_inst,
                min_market_cap_cr_inst,
                min_price_inst,
                obv_sma_period_inst,
                close_near_high_pct_inst
            )
            st.session_state.inst_screener_results = df_results_inst
            st.session_state.last_run_inst_index = index_choice_inst
    st.rerun()

if not st.session_state.inst_screener_results.empty:
    last_run_idx_inst = st.session_state.get('last_run_inst_index', "N/A")
    st.markdown(f"---")
    st.subheader(f"📊 Potential Institutional Activity ({last_run_idx_inst}) - Found: {len(st.session_state.inst_screener_results)}")
    
    df_display_inst = st.session_state.inst_screener_results.set_index('Ticker')
    
    cols_inst_display_order = ['Industry', 'Price', 'MCap (Cr)', 'Volume (M)', 'AvgVol20D (M)', 'Vol vs Avg', 'OBV > SMA']
    actual_cols_inst_display = [col for col in cols_inst_display_order if col in df_display_inst.columns]
    df_display_final_inst = df_display_inst[actual_cols_inst_display]

    styler_inst = df_display_final_inst.style.format({
        "Price": "{:.2f}", "MCap (Cr)": "{:,.0f}", # Comma for MCap
        "Volume (M)": "{:.2f}", "AvgVol20D (M)": "{:.2f}",
        "Vol vs Avg": "{:.1f}x"
    },na_rep="-")
    
    df_height_inst = min((len(df_display_final_inst) + 1) * 35 + 3, 600) 
    st.dataframe(styler_inst, use_container_width=True, height=df_height_inst)

    csv_inst = st.session_state.inst_screener_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Institutional Footprints Data ({last_run_idx_inst})", data=csv_inst,
        file_name=f"institutional_footprints_{last_run_idx_inst.replace(' ', '_')}_{datetime.today().strftime('%Y%m%d')}.csv",
        mime='text/csv')
elif st.session_state.get('last_run_inst_index'): 
    st.info("No stocks matched all the institutional footprints criteria for the last run.")
else: 
    st.info("Select parameters in the sidebar and click 'Run Institutional Footprints Screener'.")

st.markdown("---")
st.markdown("Disclaimer: This tool provides indirect indicators and does not guarantee institutional activity. For informational purposes only. Always do your own research (DYOR).")
