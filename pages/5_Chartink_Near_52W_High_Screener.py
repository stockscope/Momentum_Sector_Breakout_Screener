import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import io

# (Keep your matplotlib and seaborn imports if you plan to add charts later)
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib
# matplotlib.rcParams['axes.unicode_minus'] = False


st.set_page_config(
    page_title="Chartink Style Screener - StockScopePro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚙️ Chartink Style Screener: Near 52-Week High")
st.markdown("Replicates a Chartink screener focusing on stocks near their yearly highs with specific conditions.")

with st.expander("📜 **Screening Conditions (Based on Chartink Logic)**", expanded=True):
    st.markdown("""
    This screener attempts to replicate the following logic:
    1.  **Price Proximity to Yearly High:** `Daily Close >= 0.75 * Max High of last 250 days`
    2.  **Minimum Price:** `Daily Close > 20`
    3.  **Recent Daily Liquidity:** `Yesterday's Volume > 70,000`
    4.  **Short-term Trend:** `Daily Close > 20-Day EMA`
    5.  **Weekly RSI (Not Extreme):** `Weekly RSI(14) < 85`
    6.  **Weekly Up-Move:** `Current Week's High > Last Week's High`
    7.  **Recent Weekly Liquidity:** `Last Week's Volume > 200,000`
    """)

# --- Helper Functions --- (Can be moved to a common utils.py if used across many screeners)
@st.cache_data(ttl=timedelta(days=1), show_spinner=False)
def load_nifty_list(list_name="nifty500"): # Default to NIFTY 500, can be changed
    # Determine which list to load based on list_name
    if list_name == "nifty500":
        csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty500list.csv"
    elif list_name == "nifty200":
        csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty200list.csv"
    else: # Fallback or error for unknown list
        st.error(f"Unknown stock list specified: {list_name}")
        return []
        
    try:
        df_nifty = pd.read_csv(csv_url)
        df_nifty.columns = df_nifty.columns.str.strip()
        df_nifty = df_nifty[~df_nifty['Symbol'].str.contains("DUMMY", na=False)]
        df_nifty['Symbol'] = df_nifty['Symbol'].str.strip()
        # Industry might not be needed for this specific screener logic but good to have
        df_nifty['Industry'] = df_nifty['Industry'].fillna('Unknown') 
        df_nifty['Ticker'] = df_nifty['Symbol'] + ".NS"
        return df_nifty['Ticker'].tolist()
    except Exception as e:
        st.error(f"Error loading NIFTY {list_name.upper()} list: {e}")
        return []

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def fetch_stock_data_for_screener(tickers_tuple, days_history=300): # ~250 for yearly high + buffer for weekly calcs
    tickers_list = list(tickers_tuple)
    if not tickers_list: return {}
    
    end_date_dt = datetime.today()
    # Ensure start_date covers enough for 250 trading days + weekly lookbacks
    # Weekly resampling needs a bit more past data to form complete previous weeks.
    start_date_dt = end_date_dt - timedelta(days=days_history + 60) # Add extra buffer for weekly needs

    try:
        data = yf.download(tickers_list, start=start_date_dt.strftime('%Y-%m-%d'), 
                           end=end_date_dt.strftime('%Y-%m-%d'), # yf end is exclusive for datetime, inclusive for string
                           interval='1d', group_by='ticker', auto_adjust=False, progress=False, timeout=90)
        stock_data_processed = {}
        if data.empty: return {}

        if isinstance(data.columns, pd.MultiIndex):
            for ticker in tickers_list:
                try:
                    if ticker in data and isinstance(data[ticker], pd.DataFrame) and not data[ticker].empty:
                        df_ticker = data[ticker].copy()
                        # Ensure correct datetime index for resampling
                        df_ticker.index = pd.to_datetime(df_ticker.index)
                        stock_data_processed[ticker] = df_ticker
                except KeyError: pass
        elif len(tickers_list) == 1 and isinstance(data, pd.DataFrame) and not data.empty:
             df_single = data.copy()
             df_single.index = pd.to_datetime(df_single.index)
             stock_data_processed[tickers_list[0]] = df_single
        return stock_data_processed
    except Exception as e:
        st.sidebar.warning(f"Data download error in Chartink screener: {str(e)[:100]}")
        return {}

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0.0).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=1).mean()
    
    rs = gain / loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Handle cases: if loss is 0, RSI is 100 if gain > 0, else 50 (neutral)
    rsi[loss == 0] = np.where(gain[loss == 0] > 0, 100.0, 50.0)
    return rsi


@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def run_chartink_style_screener(tickers_list_tuple):
    screened_stocks_data = []
    # Fetch enough history for 250-day max and weekly calculations
    # 250 trading days is about 365 calendar days. Add buffer for weekly.
    hist_data_batch = fetch_stock_data_for_screener(tickers_list_tuple, days_history=300) 

    for ticker in tickers_list_tuple:
        try:
            if ticker not in hist_data_batch or hist_data_batch[ticker].empty:
                continue
            
            df_daily = hist_data_batch[ticker].copy()
            df_daily.dropna(subset=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)

            if len(df_daily) < 250: # Need at least 250 days for Max(250, High)
                continue

            # --- Daily Conditions ---
            # 1. Price Proximity to Yearly High
            max_250d_high = df_daily['High'].rolling(window=250, min_periods=200).max().iloc[-1] # Min 200 days for a decent 250d high
            latest_close = df_daily['Adj Close'].iloc[-1]
            if pd.isna(max_250d_high) or latest_close < (0.75 * max_250d_high):
                continue

            # 2. Minimum Price
            if latest_close <= 20:
                continue

            # 3. Recent Daily Liquidity (Yesterday's Volume)
            if len(df_daily) < 2: continue # Need at least 2 days for yesterday
            yesterdays_volume = df_daily['Volume'].iloc[-2]
            if pd.isna(yesterdays_volume) or yesterdays_volume <= 70000:
                continue
                
            # 4. Short-term Trend: Close > 20 EMA
            df_daily['EMA20'] = df_daily['Adj Close'].ewm(span=20, adjust=False).mean()
            ema20_daily = df_daily['EMA20'].iloc[-1]
            if pd.isna(ema20_daily) or latest_close <= ema20_daily:
                continue

            # --- Weekly Conditions ---
            # Resample daily data to weekly. 'W' or 'W-FRI' for Friday-ended weeks.
            # 'W-MON', 'W-TUE' etc. can also be used. Default 'W' is Sunday-ended.
            # For India, often 'W-FRI' makes sense. Let's use default 'W' for simplicity, ensure enough data.
            if len(df_daily) < 100: # Arbitrary check for enough data for weekly resampling (at least ~20 weeks)
                continue

            df_weekly = df_daily.resample('W').agg(
                {'Open': 'first', 'High': 'max', 'Low': 'min', 'Adj Close': 'last', 'Volume': 'sum'}
            )
            df_weekly.dropna(inplace=True) # Drop weeks with NaNs (e.g., start of data if not full week)

            if len(df_weekly) < 15: # Need at least 15 weeks for 14-period weekly RSI + prev week data
                continue

            # 5. Weekly RSI < 85
            df_weekly['RSI14'] = calculate_rsi(df_weekly['Adj Close'], period=14)
            weekly_rsi = df_weekly['RSI14'].iloc[-1] # Current (incomplete) week's RSI
            if pd.isna(weekly_rsi) or weekly_rsi >= 85:
                continue
            
            # 6. Current Week's High > Last Week's High
            # Ensure we have at least two weeks of data in df_weekly
            if len(df_weekly) < 2: continue
            current_week_high = df_weekly['High'].iloc[-1]
            last_week_high = df_weekly['High'].iloc[-2]
            if pd.isna(current_week_high) or pd.isna(last_week_high) or current_week_high <= last_week_high:
                continue

            # 7. Last Week's Volume > 200,000
            last_week_volume = df_weekly['Volume'].iloc[-2] # Volume of the last completed week
            if pd.isna(last_week_volume) or last_week_volume <= 200000:
                continue

            # If all conditions pass
            screened_stocks_data.append({
                'Ticker': ticker,
                'Price': round(latest_close, 2),
                'D_EMA20': round(ema20_daily,2),
                'D_Vol_Prev': int(yesterdays_volume),
                'W_RSI14': round(weekly_rsi, 2),
                'W_High': round(current_week_high,2),
                'W_Vol_Prev': int(last_week_volume),
                'Max250D_H': round(max_250d_high,2)
            })
        except Exception:
            # import traceback # For debugging
            # st.sidebar.text(f"Error processing {ticker}: {traceback.format_exc()[:100]}")
            continue
            
    return pd.DataFrame(screened_stocks_data)

# --- Main App Logic ---
# Allow user to select NIFTY 200 or NIFTY 500
index_choice = st.selectbox("Select Index Universe:", ["NIFTY 500", "NIFTY 200"], index=0)
list_to_load = "nifty500" if index_choice == "NIFTY 500" else "nifty200"

with st.spinner(f"📜 Loading {index_choice} list..."):
    tickers_list = load_nifty_list(list_name=list_to_load)

if not tickers_list: 
    st.error(f"Failed to load {index_choice} stock list. Screener cannot run.")
    st.stop()

# Run screener button
if st.button(f"🚀 Run Screener on {index_choice}", type="primary"):
    with st.spinner(f"🔎 Screening {len(tickers_list)} stocks from {index_choice}... This may take a few minutes."):
        df_screened_results = run_chartink_style_screener(tuple(tickers_list))

    st.markdown(f"---")
    st.subheader(f"📊 Screened Stocks ({len(df_screened_results)} found)")

    if df_screened_results.empty:
        st.info("No stocks matched all the screening criteria.")
    else:
        # Define display order
        cols_display_order = ['Price', 'Max250D_H', 'D_EMA20', 'D_Vol_Prev', 
                              'W_High', 'W_RSI14', 'W_Vol_Prev']
        
        df_display = df_screened_results.set_index('Ticker')
        
        # Ensure only existing columns are selected
        actual_cols_to_display = [col for col in cols_display_order if col in df_display.columns]
        df_display = df_display[actual_cols_to_display]

        styler = df_display.style.format("{:.2f}", na_rep="-") # General formatting for floats
        
        df_height = min((len(df_display) + 1) * 35 + 3, 600) 
        st.dataframe(styler, use_container_width=True, height=df_height)

        csv = df_screened_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Screened Data as CSV", data=csv,
            file_name=f"chartink_style_screener_{index_choice.replace(' ', '_')}_{datetime.today().strftime('%Y%m%d')}.csv",
            mime='text/csv')
else:
    st.info(f"Click the 'Run Screener' button to start scanning the {index_choice} stocks.")


st.markdown("---")
st.markdown("Disclaimer: This screener attempts to replicate logic from a Chartink setup. Data from Yahoo Finance. For informational purposes only, not financial advice. DYOR.")
