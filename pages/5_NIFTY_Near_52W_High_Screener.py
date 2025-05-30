import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import io

# Optional: Keep these if you plan to add charts later
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib
# matplotlib.rcParams['axes.unicode_minus'] = False


st.set_page_config(
    page_title="Near Yearly Highs Screener - StockScopePro", # Updated
    layout="wide",
    initial_sidebar_state="expanded"
)

# Updated Title and Markdown
st.title("📈 Stocks Near Yearly Highs with Momentum & Liquidity")
st.markdown("Identifies stocks trading close to their 52-week highs, supported by positive momentum and liquidity indicators.")

# Updated Expander
with st.expander("📜 **Key Screening Filters Applied**", expanded=True):
    st.markdown("""
    This screener identifies stocks based on the following combined criteria:

    **Price Level & Proximity to Peak:**
    1.  **Near Yearly High:** The current closing price is at least 75% of the highest price recorded over the last 250 trading days (approximately one year). This helps identify stocks that have maintained strength or recovered significantly.
    2.  **Minimum Closing Price:** The stock's current closing price must be greater than ₹20, filtering out very low-priced stocks.

    **Liquidity Indicators:**
    3.  **Sufficient Daily Volume:** Yesterday's trading volume was greater than 70,000 shares, indicating reasonable daily trading activity.
    4.  **Sufficient Weekly Volume:** The previous week's total trading volume was greater than 200,000 shares, suggesting consistent investor interest over the week.

    **Trend & Momentum Signals:**
    5.  **Short-Term Uptrend (Daily):** The current closing price is above its 20-day Exponential Moving Average (EMA), a common sign of positive short-term momentum.
    6.  **Weekly Price Advancement:** The highest price reached in the current week (to date) surpasses the highest price of the entire previous week, indicating upward progression on a weekly timeframe.
    7.  **Weekly Momentum Check (RSI):** The 14-period Relative Strength Index, calculated on weekly data, is below 85. This ensures the stock isn't in an extremely overbought condition on its weekly chart, potentially leaving room for further upside.
    """)

# --- Helper Functions ---
@st.cache_data(ttl=timedelta(days=1), show_spinner=False)
def load_nifty_list(list_name="nifty500"):
    if list_name == "nifty500":
        csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty500list.csv"
    elif list_name == "nifty200":
        csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty200list.csv"
    else:
        st.error(f"Unknown stock list specified: {list_name}")
        return []
    try:
        df_nifty = pd.read_csv(csv_url)
        df_nifty.columns = df_nifty.columns.str.strip()
        df_nifty = df_nifty[~df_nifty['Symbol'].str.contains("DUMMY", na=False)]
        df_nifty['Symbol'] = df_nifty['Symbol'].str.strip()
        df_nifty['Industry'] = df_nifty['Industry'].fillna('Unknown') 
        df_nifty['Ticker'] = df_nifty['Symbol'] + ".NS"
        return df_nifty['Ticker'].tolist()
    except Exception as e:
        st.error(f"Error loading NIFTY {list_name.upper()} list: {e}")
        return []

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def fetch_stock_data_for_screener(tickers_tuple, days_history=300):
    tickers_list = list(tickers_tuple)
    if not tickers_list: return {}
    end_date_dt = datetime.today()
    start_date_dt = end_date_dt - timedelta(days=days_history + 70) # Increased buffer slightly for weekly data start
    try:
        data = yf.download(tickers_list, start=start_date_dt.strftime('%Y-%m-%d'), 
                           end=end_date_dt.strftime('%Y-%m-%d'),
                           interval='1d', group_by='ticker', auto_adjust=False, progress=False, timeout=120) # Increased timeout
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
             df_single = data.copy()
             df_single.index = pd.to_datetime(df_single.index)
             stock_data_processed[tickers_list[0]] = df_single
        return stock_data_processed
    except Exception as e:
        st.sidebar.warning(f"Data download error: {str(e)[:100]}")
        return {}

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    if delta.empty: return pd.Series(index=series.index, dtype=float) # Handle empty delta

    gain = delta.where(delta > 0, 0.0).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=1).mean()
    
    # To avoid division by zero if loss is consistently zero
    rs = gain / loss.replace(0, np.nan) # Replace 0 loss with NaN for division, then handle
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Refined handling for RSI edge cases (loss is zero)
    # If gain > 0 and loss is 0, RSI should be 100
    rsi[ (gain > 0) & (loss == 0) ] = 100.0
    # If gain is 0 and loss is 0 (no price change over period), RSI is often considered 50 (neutral)
    rsi[ (gain == 0) & (loss == 0) ] = 50.0
    # If gain is NaN (e.g. not enough data for mean), RSI will be NaN
    rsi[gain.isna() | loss.isna()] = np.nan
    return rsi

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def run_proximity_screener(tickers_list_tuple): # Renamed function
    screened_stocks_data = []
    hist_data_batch = fetch_stock_data_for_screener(tickers_list_tuple, days_history=300) 

    for ticker in tickers_list_tuple:
        try:
            if ticker not in hist_data_batch or hist_data_batch[ticker].empty:
                continue
            
            df_daily = hist_data_batch[ticker].copy()
            # Use 'Close' for EMA, 'Adj Close' for price checks if preferred, be consistent
            # For this screener, let's use 'Adj Close' for price levels and EMA on 'Adj Close'
            df_daily.dropna(subset=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)

            if len(df_daily) < 250: 
                continue

            max_250d_high = df_daily['High'].rolling(window=250, min_periods=200).max().iloc[-1]
            latest_close = df_daily['Adj Close'].iloc[-1]
            if pd.isna(max_250d_high) or latest_close < (0.75 * max_250d_high):
                continue
            if latest_close <= 20:
                continue
            if len(df_daily) < 2: continue
            yesterdays_volume = df_daily['Volume'].iloc[-2]
            if pd.isna(yesterdays_volume) or yesterdays_volume <= 70000:
                continue
            df_daily['EMA20'] = df_daily['Adj Close'].ewm(span=20, adjust=False).mean()
            ema20_daily = df_daily['EMA20'].iloc[-1]
            if pd.isna(ema20_daily) or latest_close <= ema20_daily:
                continue

            if len(df_daily) < 70: # Need ~14 weeks * 5 days for weekly RSI, plus some buffer
                continue
            df_weekly = df_daily.resample('W-FRI').agg( # Using Friday-ended weeks
                {'Open': 'first', 'High': 'max', 'Low': 'min', 'Adj Close': 'last', 'Volume': 'sum'}
            )
            df_weekly.dropna(inplace=True)
            if len(df_weekly) < 15: 
                continue

            df_weekly['RSI14'] = calculate_rsi(df_weekly['Adj Close'], period=14)
            # For weekly RSI, use the last *completed* week if available, or current if it's the only one.
            # If current week is very early, its RSI might be volatile.
            # Let's assume Chartink uses the latest available weekly RSI value.
            weekly_rsi = df_weekly['RSI14'].iloc[-1] 
            if pd.isna(weekly_rsi) or weekly_rsi >= 85:
                continue
            
            if len(df_weekly) < 2: continue # Need current and previous week
            current_week_high = df_weekly['High'].iloc[-1] # High of the current (possibly incomplete) week
            last_completed_week_high = df_weekly['High'].iloc[-2] # High of the last fully completed week
            if pd.isna(current_week_high) or pd.isna(last_completed_week_high) or current_week_high <= last_completed_week_high:
                continue

            last_completed_week_volume = df_weekly['Volume'].iloc[-2]
            if pd.isna(last_completed_week_volume) or last_completed_week_volume <= 200000:
                continue

            screened_stocks_data.append({
                'Ticker': ticker,
                'Price': round(latest_close, 2),
                'D_EMA20': round(ema20_daily,2),
                'D_Vol_Yest': int(yesterdays_volume), # Renamed for clarity
                'W_RSI14': round(weekly_rsi, 2) if pd.notna(weekly_rsi) else np.nan,
                'W_High_Curr': round(current_week_high,2), # Renamed
                'W_Vol_PrevWk': int(last_completed_week_volume), # Renamed
                'Yearly_High': round(max_250d_high,2) # Renamed
            })
        except Exception:
            continue
            
    return pd.DataFrame(screened_stocks_data)

# --- Main App Logic ---
index_choice = st.selectbox("Select Index Universe:", ["NIFTY 500", "NIFTY 200"], index=0)
list_to_load = "nifty500" if index_choice == "NIFTY 500" else "nifty200"

with st.spinner(f"📜 Loading {index_choice} list..."):
    tickers_list = load_nifty_list(list_name=list_to_load)

if not tickers_list: 
    st.error(f"Failed to load {index_choice} stock list. Screener cannot run.")
    st.stop()

if st.button(f"🚀 Run Screener on {index_choice}", type="primary"):
    with st.spinner(f"🔎 Screening {len(tickers_list)} stocks from {index_choice}... This may take a few minutes."):
        df_screened_results = run_proximity_screener(tuple(tickers_list)) # Called renamed function

    st.markdown(f"---")
    st.subheader(f"📊 Screened Stocks ({len(df_screened_results)} found)")

    if df_screened_results.empty:
        st.info("No stocks matched all the screening criteria.")
    else:
        cols_display_order = ['Price', 'Yearly_High', 'D_EMA20', 'D_Vol_Yest', 
                              'W_High_Curr', 'W_RSI14', 'W_Vol_PrevWk']
        
        df_display = df_screened_results.set_index('Ticker')
        actual_cols_to_display = [col for col in cols_display_order if col in df_display.columns]
        df_display_final = df_display[actual_cols_to_display] # Use a new variable for the final df

        # Formatting dictionary for Styler
        formats = {col: "{:.2f}" for col in ['Price', 'Yearly_High', 'D_EMA20', 'W_High_Curr', 'W_RSI14']}
        # Integer columns (volume) don't need float formatting if already int, but can be left to default
        # If they are floats due to NaNs being introduced then coerced, {:.0f} might be better or Styler handles it.

        styler = df_display_final.style.format(formats, na_rep="-")
        
        df_height = min((len(df_display_final) + 1) * 35 + 3, 600) 
        st.dataframe(styler, use_container_width=True, height=df_height)

        csv = df_screened_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Screened Data as CSV", data=csv,
            file_name=f"near_yearly_high_screener_{index_choice.replace(' ', '_')}_{datetime.today().strftime('%Y%m%d')}.csv",
            mime='text/csv')
else:
    st.info(f"Click the 'Run Screener' button to start scanning the {index_choice} stocks.")

st.markdown("---")
st.markdown("Disclaimer: Data is sourced from Yahoo Finance and may have inaccuracies or delays. This tool is for informational purposes only and not financial advice. Always do your own research (DYOR).")
