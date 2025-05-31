import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import io

st.set_page_config(
    page_title="Advanced Pullback Screener - StockScopePro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚙️ Advanced Pullback Screener")
st.markdown("Identifies stocks near yearly highs, pulling back to a key EMA zone, with volume and trend confirmations.")

# --- User Selections (Sidebar) ---
st.sidebar.header("Screener Settings")
index_choice_adv_pullback = st.sidebar.selectbox(
    "Select Index Universe:", 
    ["NIFTY 500", "NIFTY 200"], 
    index=0, 
    key="adv_pullback_index_choice"
)
list_to_load_adv_pullback = "nifty500" if index_choice_adv_pullback == "NIFTY 500" else "nifty200"

# Pullback EMA Zone - We'll check against all three (7, 14, 20)
ema_pullback_zone_proximity_pct = st.sidebar.slider(
    "Max % Above Slowest EMA (20) for Pullback Zone:", 
    0.1, 5.0, 1.5, 0.1, # Close can be up to 1.5% above the 20EMA
    format="%.1f%%",
    help="Stock's Close should be within this % above the 20-EMA if it's the highest of the 7,14,20 EMA zone."
)

prior_strength_lookback = st.sidebar.slider(
    "Prior Strength Lookback (Days):", 5, 20, 10, 1, key="adv_prior_lookback",
    help="Days ago to check if price was significantly above the EMA zone."
)
prior_strength_pct_above = st.sidebar.slider(
    "Prior Strength % Above 20-EMA:", 1.0, 15.0, 5.0, 0.1, key="adv_prior_pct", format="%.1f%%",
    help="Price must have been at least this % above the 20-EMA during the lookback period."
)


with st.expander("📜 **Screening Logic Explained**", expanded=True):
    st.markdown(f"""
    This screener identifies stocks based on the following combined criteria:

    **1. Price & Liquidity Basics:**
        - Current Daily Close > ₹100.
        - Current Daily Volume > 100,000 shares.

    **2. Near Yearly High:**
        - Current Daily Close is at least 75% of the max high over the last 250 trading days.

    **3. Pullback to Short-Term EMA Zone (7, 14, 20 EMA):**
        - The stock's current Low must have touched or gone below the **fastest EMA (7-EMA)** in the last 1-2 days.
        - The stock's current Close must be above the **fastest EMA (7-EMA)**.
        - The stock's current Close must be no more than {ema_pullback_zone_proximity_pct}% above the **slowest EMA (20-EMA)** of the zone.
        - This defines a tight pullback into the support zone formed by these EMAs.

    **4. Recent Upward Momentum (Avoiding Prolonged Consolidation):**
        - Within the last {prior_strength_lookback} trading days, the stock's Close was at least {prior_strength_pct_above}% *above* its 20-EMA, indicating it wasn't just hovering around the EMAs for a long time.
    
    **5. Daily Trend Confirmation:**
        - Current Daily Close > 20-Day EMA (this ensures it's finding support at/above it).

    **6. Weekly Indicators:**
        - Weekly RSI(14) < 85 (not extremely overbought on a weekly chart).
        - Current Week's High > Last Week's High (weekly uptrend continuation).
        - Last Week's Total Volume > 250,000 shares.
    """)

# --- Helper Functions --- (load_nifty_list_pullback can be reused, fetch_stock_data too)
@st.cache_data(ttl=timedelta(days=1), show_spinner=False)
def load_nifty_list_adv_pullback(list_name):
    if list_name == "nifty500": csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty500list.csv"
    elif list_name == "nifty200": csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty200list.csv"
    else: st.error(f"Unknown list: {list_name}"); return []
    try:
        df = pd.read_csv(csv_url); df.columns=df.columns.str.strip()
        df = df[~df['Symbol'].str.contains("DUMMY",na=False)]; df['Symbol']=df['Symbol'].str.strip()
        df['Ticker'] = df['Symbol'] + ".NS"; return df['Ticker'].tolist()
    except Exception as e: st.error(f"Error loading {list_name}: {e}"); return []

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def fetch_stock_data_adv_pullback(tickers_tuple, days_history=300):
    tickers_list = list(tickers_tuple); end_dt=datetime.today()
    start_dt=end_dt-timedelta(days=days_history+70) # Buffer for weekly + EMAs
    if not tickers_list: return {}
    try:
        data=yf.download(tickers_list, start=start_dt.strftime('%Y-%m-%d'),end=end_dt.strftime('%Y-%m-%d'),
                           interval='1d',group_by='ticker',auto_adjust=False,progress=False,timeout=120)
        processed={};_ = [processed.update({t:data[t].copy().set_index(pd.to_datetime(data[t].index))} if t in data and isinstance(data[t],pd.DataFrame) and not data[t].empty else None) for t in tickers_list] if isinstance(data.columns,pd.MultiIndex) else (processed.update({tickers_list[0]:data.copy().set_index(pd.to_datetime(data.index))}) if len(tickers_list)==1 and isinstance(data,pd.DataFrame) and not data.empty else None)
        return processed
    except Exception as e: st.sidebar.warning(f"DL Error: {str(e)[:50]}"); return {}


def calculate_rsi_adv(series, period=14): # Renamed to avoid conflict if in utils
    delta = series.diff(1)
    if delta.empty or len(delta) < period : return pd.Series(index=series.index, dtype=float)
    gain = delta.where(delta > 0, 0.0).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan) 
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[(gain > 0) & (loss == 0)] = 100.0
    rsi[(gain == 0) & (loss == 0)] = 50.0
    rsi[gain.isna() | loss.isna()] = np.nan
    return rsi

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def run_advanced_pullback_screener(
    tickers_list_tuple, 
    pb_zone_prox_pct, 
    prior_str_lookback, 
    prior_str_pct_abv
):
    screened_stocks_data = []
    # Need enough history for 250-day high, EMAs, weekly data
    hist_data_batch = fetch_stock_data_adv_pullback(tickers_list_tuple, days_history=300) 

    for ticker in tickers_list_tuple:
        try:
            if ticker not in hist_data_batch or hist_data_batch[ticker].empty: continue
            df_daily = hist_data_batch[ticker].copy()
            df_daily.dropna(subset=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)

            if len(df_daily) < 255: continue # Min for 250-day high + a few days for EMAs/lookbacks

            # --- Daily Calculations & Filters ---
            latest = df_daily.iloc[-1]
            latest_close = latest['Adj Close']
            latest_low = latest['Low']
            latest_volume = latest['Volume']

            if latest_close <= 100: continue
            if latest_volume <= 100000: continue

            max_250d_high = df_daily['High'].rolling(window=250, min_periods=200).max().iloc[-1]
            if pd.isna(max_250d_high) or latest_close < (0.75 * max_250d_high): continue

            df_daily['EMA7'] = df_daily['Adj Close'].ewm(span=7, adjust=False, min_periods=6).mean()
            df_daily['EMA14'] = df_daily['Adj Close'].ewm(span=14, adjust=False, min_periods=13).mean()
            df_daily['EMA20'] = df_daily['Adj Close'].ewm(span=20, adjust=False, min_periods=19).mean()
            
            ema7 = latest['EMA7']; ema14 = latest['EMA14']; ema20 = latest['EMA20']
            if pd.isna(ema7) or pd.isna(ema14) or pd.isna(ema20): continue

            # Daily Close > 20 EMA (Condition 5)
            if latest_close <= ema20: continue
            
            # Pullback Zone Condition
            # Low touched fastest EMA (7) in last 1-2 days
            touched_ema7 = (latest_low <= ema7) or \
                           (len(df_daily) >= 2 and df_daily['Low'].iloc[-2] <= df_daily['EMA7'].iloc[-2] if pd.notna(df_daily['EMA7'].iloc[-2]) else False)
            
            # Close is above fastest EMA (7)
            close_above_ema7 = latest_close > ema7 # Or >= depending on strictness

            # Close is within proximity of slowest EMA (20)
            close_near_ema20_zone = latest_close <= ema20 * (1 + pb_zone_prox_pct / 100.0)
            
            if not (touched_ema7 and close_above_ema7 and close_near_ema20_zone): continue

            # Prior Strength Condition (relative to 20 EMA for this example)
            if len(df_daily) < prior_str_lookback + 1: continue
            price_at_lookback = df_daily['Adj Close'].iloc[-(prior_str_lookback + 1)]
            ema20_at_lookback = df_daily['EMA20'].iloc[-(prior_str_lookback + 1)]
            if pd.isna(price_at_lookback) or pd.isna(ema20_at_lookback): continue
            if not (price_at_lookback > ema20_at_lookback * (1 + prior_str_pct_abv / 100.0)): continue

            # --- Weekly Conditions ---
            if len(df_daily) < 70: continue # Enough for ~14 weekly periods
            df_weekly = df_daily.resample('W-FRI').agg(
                {'Open': 'first', 'High': 'max', 'Low': 'min', 'Adj Close': 'last', 'Volume': 'sum'}
            )
            df_weekly.dropna(inplace=True)
            if len(df_weekly) < 15: continue # For weekly RSI

            df_weekly['RSI14'] = calculate_rsi_adv(df_weekly['Adj Close'], period=14)
            if df_weekly['RSI14'].empty: continue
            weekly_rsi = df_weekly['RSI14'].iloc[-1] 
            if pd.isna(weekly_rsi) or weekly_rsi >= 85: continue
            
            if len(df_weekly) < 2: continue 
            current_week_high = df_weekly['High'].iloc[-1] 
            last_completed_week_high = df_weekly['High'].iloc[-2]
            if pd.isna(current_week_high) or pd.isna(last_completed_week_high) or current_week_high <= last_completed_week_high: continue

            last_completed_week_volume = df_weekly['Volume'].iloc[-2]
            if pd.isna(last_completed_week_volume) or last_completed_week_volume <= 250000: continue
            
            screened_stocks_data.append({
                'Ticker': ticker,
                'Price': round(latest_close, 2),
                'EMA7': round(ema7, 2), 'EMA14': round(ema14, 2), 'EMA20': round(ema20, 2),
                'D_Vol': int(latest_volume),
                'W_RSI14': round(weekly_rsi, 2) if pd.notna(weekly_rsi) else np.nan,
                'W_High_Curr': round(current_week_high,2),
                'W_Vol_PrevWk': int(last_completed_week_volume),
                'Yearly_High': round(max_250d_high,2)
            })
        except Exception: continue # Skip ticker on any error
            
    return pd.DataFrame(screened_stocks_data)

# --- Main App Logic ---
if 'adv_pullback_results' not in st.session_state:
    st.session_state.adv_pullback_results = pd.DataFrame()

if st.sidebar.button(f"🚀 Run Advanced Pullback Screener on {index_choice_adv_pullback}", type="primary", key="run_adv_pullback_btn"):
    with st.spinner(f"📜 Loading {index_choice_adv_pullback} list..."):
        tickers_to_scan_adv = load_nifty_list_adv_pullback(list_name=list_to_load_adv_pullback)

    if not tickers_to_scan_adv: 
        st.error(f"Failed to load {index_choice_adv_pullback} stock list.")
    else:
        with st.spinner(f"🔎 Screening {len(tickers_to_scan_adv)} stocks from {index_choice_adv_pullback}... This is complex and may take time."):
            df_results_adv = run_advanced_pullback_screener(
                tuple(tickers_to_scan_adv),
                ema_pullback_zone_proximity_pct,
                prior_strength_lookback,
                prior_strength_pct_above
            )
            st.session_state.adv_pullback_results = df_results_adv
            st.session_state.last_run_adv_pullback_index = index_choice_adv_pullback
    st.rerun()

# Display results from session state
if not st.session_state.adv_pullback_results.empty:
    last_run_idx_adv = st.session_state.get('last_run_adv_pullback_index', "N/A")
    st.markdown(f"---")
    st.subheader(f"📊 Advanced Pullback Candidates for {last_run_idx_adv} - Found: {len(st.session_state.adv_pullback_results)}")
    
    df_display_adv_pullback = st.session_state.adv_pullback_results.set_index('Ticker')
    
    cols_adv_display_order = ['Price', 'EMA7', 'EMA14', 'EMA20', 'Yearly_High', 
                              'D_Vol', 'W_High_Curr', 'W_RSI14', 'W_Vol_PrevWk']
    actual_cols_adv_display = [col for col in cols_adv_display_order if col in df_display_adv_pullback.columns]
    df_display_final_adv = df_display_adv_pullback[actual_cols_adv_display]

    formats_adv = {col: "{:.2f}" for col in ['Price', 'EMA7', 'EMA14', 'EMA20', 'Yearly_High', 'W_High_Curr', 'W_RSI14']}
    styler_adv_pullback = df_display_final_adv.style.format(formats_adv, na_rep="-")
    
    df_height_adv_pullback = min((len(df_display_final_adv) + 1) * 35 + 3, 600) 
    st.dataframe(styler_adv_pullback, use_container_width=True, height=df_height_adv_pullback)

    csv_adv_pullback = st.session_state.adv_pullback_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Advanced Pullback Data ({last_run_idx_adv})", data=csv_adv_pullback,
        file_name=f"adv_pullback_{last_run_idx_adv.replace(' ', '_')}_{datetime.today().strftime('%Y%m%d')}.csv",
        mime='text/csv')
elif st.session_state.get('last_run_adv_pullback_index'): # If run but no results
    st.info("No stocks matched all the advanced pullback screening criteria for the last run.")
else: 
    st.info("Select parameters in the sidebar and click 'Run Advanced Pullback Screener'.")

st.markdown("---")
st.markdown("Disclaimer: This tool is for informational purposes only. Always do your own research (DYOR).")
