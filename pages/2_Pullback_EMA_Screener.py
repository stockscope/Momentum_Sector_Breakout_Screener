import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import io

st.set_page_config(
    page_title="Pullback to EMA Screener - StockScopePro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔁 Pullback to EMA Screener")
st.markdown("Identifies stocks potentially pulling back to key short-term Exponential Moving Averages (EMAs) within an uptrend.")

# --- User Selections ---
st.sidebar.header("Screener Settings")
index_choice_pullback = st.sidebar.selectbox(
    "Select Index Universe:", 
    ["NIFTY 500", "NIFTY 200"], 
    index=0, 
    key="pullback_index_choice"
)
list_to_load_pullback = "nifty500" if index_choice_pullback == "NIFTY 500" else "nifty200"

ema_options = [7, 14, 20]
selected_ema = st.sidebar.selectbox(
    "Pullback to EMA Period:", 
    ema_options, 
    index=ema_options.index(20), # Default to 20 EMA
    key="pullback_ema_period"
)

pullback_proximity_pct = st.sidebar.slider(
    f"Max % Distance from {selected_ema}-EMA for Pullback:", 
    0.1, 3.0, 1.0, 0.1, 
    format="%.1f%%",
    help=f"Stock's Close should be within this percentage above the {selected_ema}-EMA. Low of day might dip below."
)

lookback_for_prior_strength = st.sidebar.slider(
    "Days Ago to Check for Price > EMA (Strength):", 
    3, 15, 7, 1,
    help="How many days ago the price should have been clearly above the EMA before the pullback."
)
min_pct_above_ema_prior = st.sidebar.slider(
    f"% Above {selected_ema}-EMA for Prior Strength:",
    1.0, 10.0, 3.0, 0.1,
    format="%.1f%%",
    help=f"Price must have been at least this % above {selected_ema}-EMA {lookback_for_prior_strength} days ago."
)


with st.expander("📜 **Screening Logic Explained**", expanded=True):
    st.markdown(f"""
    This screener looks for stocks that:
    1.  Are in a general **medium-term uptrend** (Current Close > 50-day Simple Moving Average).
    2.  Have recently **pulled back towards the {selected_ema}-day EMA**:
        - The Low of the current or previous day was at or below the {selected_ema}-EMA.
        - The current Close is at or slightly above the {selected_ema}-EMA (within {pullback_proximity_pct}%).
    3.  Showed **prior strength** before the pullback:
        - About {lookback_for_prior_strength} trading days ago, the Close was at least {min_pct_above_ema_prior}% *above* the {selected_ema}-EMA. This helps filter out stocks that have been consolidating around the EMA for a long time.
    4.  Have a **minimum daily volume** (Average 20-day Volume > 50,000).
    5.  Have a **minimum closing price** (> ₹20).
    """)

# --- Helper Functions ---
@st.cache_data(ttl=timedelta(days=1), show_spinner=False)
def load_nifty_list_pullback(list_name): # Renamed to avoid conflict
    # ... (same as load_nifty_list from previous script) ...
    if list_name == "nifty500": csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty500list.csv"
    elif list_name == "nifty200": csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty200list.csv"
    else: st.error(f"Unknown stock list: {list_name}"); return []
    try:
        df_nifty = pd.read_csv(csv_url)
        df_nifty.columns = df_nifty.columns.str.strip()
        df_nifty = df_nifty[~df_nifty['Symbol'].str.contains("DUMMY", na=False)]
        df_nifty['Symbol'] = df_nifty['Symbol'].str.strip()
        df_nifty['Ticker'] = df_nifty['Symbol'] + ".NS"
        return df_nifty['Ticker'].tolist()
    except Exception as e: st.error(f"Error loading {list_name.upper()} list: {e}"); return []

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def fetch_stock_data_pullback(tickers_tuple, days_history=100): # Renamed
    # ... (same as fetch_stock_data_for_screener, ensure enough history for 50SMA + EMA + lookbacks) ...
    tickers_list = list(tickers_tuple)
    if not tickers_list: return {}
    end_date_dt = datetime.today()
    start_date_dt = end_date_dt - timedelta(days=days_history + 30) # Buffer for EMAs
    try:
        data = yf.download(tickers_list, start=start_date_dt.strftime('%Y-%m-%d'), 
                           end=end_date_dt.strftime('%Y-%m-%d'),
                           interval='1d', group_by='ticker', auto_adjust=False, progress=False, timeout=120)
        stock_data_processed = {}
        if data.empty: return {}
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in tickers_list:
                try:
                    if ticker in data and isinstance(data[ticker], pd.DataFrame) and not data[ticker].empty:
                        df_ticker = data[ticker].copy()
                        df_ticker.index = pd.to_datetime(df_ticker.index)
                        stock_data_processed[ticker] = df_ticker
                except KeyError: pass
        elif len(tickers_list) == 1 and isinstance(data, pd.DataFrame) and not data.empty:
             df_single = data.copy(); df_single.index = pd.to_datetime(df_single.index)
             stock_data_processed[tickers_list[0]] = df_single
        return stock_data_processed
    except Exception as e: st.sidebar.warning(f"Data download error: {str(e)[:100]}"); return {}

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def run_pullback_screener(
    tickers_list_tuple, 
    ema_period, 
    proximity_pct, 
    strength_lookback_days, 
    strength_min_pct_above_ema
):
    screened_stocks_data = []
    # Need enough history for 50SMA and the EMA period + strength_lookback_days
    days_of_data_needed = max(60, ema_period + strength_lookback_days + 20) # Buffer
    hist_data_batch = fetch_stock_data_pullback(tickers_list_tuple, days_history=days_of_data_needed) 

    for ticker in tickers_list_tuple:
        try:
            if ticker not in hist_data_batch or hist_data_batch[ticker].empty:
                continue
            
            df = hist_data_batch[ticker].copy()
            df.dropna(subset=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)

            if len(df) < days_of_data_needed - 10: # Ensure sufficient data after dropna
                continue

            # Calculate EMAs and SMA
            ema_col_name = f'EMA{ema_period}'
            df[ema_col_name] = df['Adj Close'].ewm(span=ema_period, adjust=False, min_periods=ema_period-1).mean()
            df['SMA50'] = df['Adj Close'].rolling(window=50, min_periods=40).mean()
            df['AvgVol20'] = df['Volume'].rolling(window=20, min_periods=15).mean()


            # Get latest values
            if len(df) < 2: continue # Need at least 2 days for current and previous
            latest = df.iloc[-1]
            prev_day = df.iloc[-2]

            latest_close = latest['Adj Close']
            latest_low = latest['Low']
            current_ema = latest[ema_col_name]
            current_sma50 = latest['SMA50']
            avg_vol_20 = latest['AvgVol20']

            if pd.isna(latest_close) or pd.isna(current_ema) or pd.isna(current_sma50) or pd.isna(avg_vol_20):
                continue
            
            # --- Apply Filters ---
            # 1. Minimum Price & Volume
            if latest_close <= 20: continue
            if avg_vol_20 <= 50000: continue

            # 2. General Uptrend: Price > 50 SMA
            if latest_close <= current_sma50: continue

            # 3. Pullback Condition:
            #    - Low of current or previous day touched or went below EMA
            #    - Close is currently at or slightly above EMA (within proximity_pct)
            touched_ema = (latest_low <= current_ema) or \
                          (prev_day['Low'] <= df[ema_col_name].iloc[-2] if len(df[ema_col_name]) >=2 and pd.notna(df[ema_col_name].iloc[-2]) else False)
            
            close_near_ema = (latest_close >= current_ema) and \
                             (latest_close <= current_ema * (1 + proximity_pct / 100.0))
            
            if not (touched_ema and close_near_ema):
                continue

            # 4. Prior Strength Condition (to avoid long consolidation)
            if len(df) < strength_lookback_days + 1: continue # Need enough data for lookback
            
            price_at_lookback = df['Adj Close'].iloc[-(strength_lookback_days + 1)] # +1 because iloc is 0-based from end
            ema_at_lookback = df[ema_col_name].iloc[-(strength_lookback_days + 1)]
            
            if pd.isna(price_at_lookback) or pd.isna(ema_at_lookback): continue

            if not (price_at_lookback > ema_at_lookback * (1 + strength_min_pct_above_ema / 100.0)):
                continue
            
            # If all conditions met
            screened_stocks_data.append({
                'Ticker': ticker,
                'Price': round(latest_close, 2),
                f'{ema_period}EMA': round(current_ema, 2),
                'SMA50': round(current_sma50, 2),
                'Dist_EMA(%)': round(((latest_close - current_ema) / current_ema) * 100, 2) if current_ema > 0 else 0,
                'AvgVol20(K)': round(avg_vol_20 / 1000, 1) # Avg Vol in thousands
            })

        except Exception:
            # import traceback # For debugging
            # st.sidebar.text(f"Error processing {ticker} for pullback: {traceback.format_exc()[:100]}")
            continue
            
    return pd.DataFrame(screened_stocks_data)

# --- Main App Logic ---
if 'pullback_screened_results' not in st.session_state: # Initialize if not present
    st.session_state.pullback_screened_results = pd.DataFrame()

if st.sidebar.button(f"🚀 Run Pullback Screener on {index_choice_pullback}", type="primary", key="run_pullback_btn"):
    with st.spinner(f"📜 Loading {index_choice_pullback} list..."):
        tickers_to_scan = load_nifty_list_pullback(list_name=list_to_load_pullback)

    if not tickers_to_scan: 
        st.error(f"Failed to load {index_choice_pullback} stock list.")
    else:
        with st.spinner(f"🔎 Screening {len(tickers_to_scan)} stocks from {index_choice_pullback}... This may take a minute."):
            df_results = run_pullback_screener(
                tuple(tickers_to_scan),
                selected_ema,
                pullback_proximity_pct,
                lookback_for_prior_strength,
                min_pct_above_ema_prior
            )
            st.session_state.pullback_screened_results = df_results # Store in session state
            st.session_state.last_run_pullback_index = index_choice_pullback # Store which index was run
            st.session_state.last_run_pullback_ema = selected_ema
    st.rerun() # Rerun to display results outside button scope

# Display results from session state
if not st.session_state.pullback_screened_results.empty:
    last_run_idx = st.session_state.get('last_run_pullback_index', "N/A")
    last_run_ema = st.session_state.get('last_run_pullback_ema', "N/A")
    st.markdown(f"---")
    st.subheader(f"📊 Pullback Candidates for {last_run_idx} (EMA {last_run_ema}) - Found: {len(st.session_state.pullback_screened_results)}")
    
    df_display_pullback = st.session_state.pullback_screened_results.set_index('Ticker')
    
    # Basic styling
    styler_pullback = df_display_pullback.style.format("{:.2f}", na_rep="-", subset=pd.IndexSlice[:, df_display_pullback.select_dtypes(include=np.number).columns])
    
    df_height_pullback = min((len(df_display_pullback) + 1) * 35 + 3, 600) 
    st.dataframe(styler_pullback, use_container_width=True, height=df_height_pullback)

    csv_pullback = st.session_state.pullback_screened_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Pullback Data ({last_run_idx} EMA{last_run_ema})", data=csv_pullback,
        file_name=f"pullback_ema{last_run_ema}_{last_run_idx.replace(' ', '_')}_{datetime.today().strftime('%Y%m%d')}.csv",
        mime='text/csv')
elif st.session_state.get('last_run_pullback_index'): # If run but no results
    st.info("No stocks matched all the pullback screening criteria for the last run.")
else: # Initial state before any run
    st.info("Select parameters in the sidebar and click 'Run Pullback Screener'.")


st.markdown("---")
st.markdown("Disclaimer: This tool is for informational purposes only and not financial advice. Always do your own research (DYOR).")
