import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import io # For capturing df.info()

# Ensure Matplotlib's minus sign is rendered correctly
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide")
st.title("📈 NIFTY 500: Valuation & Uptrend Screener")
st.markdown("Identifies NIFTY 500 stocks with potentially good valuation and signs of an uptrend using fixed criteria.")

# --- Define Fixed Screening Criteria ---
FIXED_MAX_PE = 40.0
FIXED_MAX_PB = 7.0
FIXED_MAX_DE = 2.0
FIXED_MIN_ROE_PERCENT = 10.0
FIXED_MIN_EPS_GROWTH_PERCENT = 0.0
FIXED_PRICE_GT_20EMA = True
FIXED_PRICE_GT_50SMA = True
FIXED_SMA50_GT_SMA200 = False
FIXED_MIN_RSI = 35
FIXED_MAX_RSI = 75
FIXED_APPLY_VOLUME_BUZZ = True
FIXED_VOLUME_BUZZ_FACTOR = 1.1

with st.expander("🧠 **Screening Philosophy & Fixed Criteria Used**", expanded=True):
    st.markdown(f"""
    This screener combines fundamental valuation metrics with technical trend indicators to find potentially undervalued stocks that are showing signs of an upward price movement. The following fixed criteria are used:

    **Fundamental Filters:**
    - **Max P/E Ratio (Trailing):** {FIXED_MAX_PE}
    - **Max P/B Ratio:** {FIXED_MAX_PB}
    - **Max Debt-to-Equity Ratio:** {FIXED_MAX_DE}
    - **Min ROE (Return on Equity):** {FIXED_MIN_ROE_PERCENT}%
    - **Min EPS Growth (Quarterly, YoY):** {FIXED_MIN_EPS_GROWTH_PERCENT}%

    **Technical Trend Filters:**
    - **Price > 20-Day EMA:** {'Yes' if FIXED_PRICE_GT_20EMA else 'No'}
    - **Price > 50-Day SMA:** {'Yes' if FIXED_PRICE_GT_50SMA else 'No'}
    - **50-Day SMA > 200-Day SMA (Golden Cross):** {'Yes' if FIXED_SMA50_GT_SMA200 else 'No'}
    - **RSI (14-day) Range:** {FIXED_MIN_RSI} - {FIXED_MAX_RSI}
    - **Volume Buzz Applied:** {'Yes' if FIXED_APPLY_VOLUME_BUZZ else 'No'}
    - **Volume Buzz Factor (Current Vol / Avg Vol):** {FIXED_VOLUME_BUZZ_FACTOR}x (if applied)
    """)

@st.cache_data(ttl=timedelta(days=1), show_spinner=False)
def load_nifty500_list():
    csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty500list.csv"
    try:
        df_nifty = pd.read_csv(csv_url)
        df_nifty.columns = df_nifty.columns.str.strip()
        df_nifty = df_nifty[~df_nifty['Symbol'].str.contains("DUMMY", na=False)]
        df_nifty['Symbol'] = df_nifty['Symbol'].str.strip()
        df_nifty['Industry'] = df_nifty['Industry'].fillna('Unknown')
        df_nifty['Ticker'] = df_nifty['Symbol'] + ".NS"
        return df_nifty['Ticker'].tolist(), dict(zip(df_nifty['Ticker'], df_nifty['Industry']))
    except Exception as e:
        st.error(f"Error loading NIFTY 500 list: {e}")
        return [], {}

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def fetch_stock_data(tickers_tuple, start_date_str, end_date_str):
    tickers_list = list(tickers_tuple)
    if not tickers_list: return {}
    try:
        data = yf.download(tickers_list, start=start_date_str, end=end_date_str,
                           interval='1d', group_by='ticker', auto_adjust=False, progress=False, timeout=60)
        stock_data_processed = {}
        if data.empty: return {}
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in tickers_list:
                try:
                    if ticker in data and isinstance(data[ticker], pd.DataFrame) and not data[ticker].empty:
                        stock_data_processed[ticker] = data[ticker]
                except KeyError: pass
        elif len(tickers_list) == 1 and isinstance(data, pd.DataFrame) and not data.empty:
             stock_data_processed[tickers_list[0]] = data
        return stock_data_processed
    except Exception:
        return {}

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def get_stock_info(ticker_str):
    try:
        stock = yf.Ticker(ticker_str)
        info = stock.info
        keys_to_extract = [
            'trailingPE', 'forwardPE', 'priceToBook', 'debtToEquity', 
            'returnOnEquity', 'earningsQuarterlyGrowth', 'beta', 
            'marketCap', 'industry', 'sector', 'previousClose', 'volume', 'averageVolume'
        ]
        filtered_info = {k: info.get(k) for k in keys_to_extract} # Keep None if that's what yf returns
        return filtered_info
    except Exception:
        return {}

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def run_screener(tickers_list_tuple, sector_map_dict, 
                 filter_max_pe, filter_max_pb, filter_max_de, filter_min_roe_pct, filter_min_eps_g_pct,
                 filter_price_gt_20ema, filter_price_gt_50sma, filter_sma50_gt_sma200,
                 filter_min_rsi, filter_max_rsi, filter_vol_buzz_factor, filter_apply_vol_buzz):
    screened_stocks_data = []
    end_date_dt = datetime.today()
    start_date_dt = end_date_dt - timedelta(days=365 * 1.5) 
    hist_data_batch = fetch_stock_data(tickers_list_tuple, start_date_dt.strftime('%Y-%m-%d'), end_date_dt.strftime('%Y-%m-%d'))

    for ticker in tickers_list_tuple:
        try:
            info = get_stock_info(ticker)
            pe = info.get('trailingPE')
            pb = info.get('priceToBook')
            de = info.get('debtToEquity')
            roe = info.get('returnOnEquity')
            eps_g = info.get('earningsQuarterlyGrowth')

            # Apply Fundamental Filters - Skip if metric is None OR fails condition
            if pe is not None and (pe > filter_max_pe or pe <= 0): continue
            if pb is not None and (pb > filter_max_pb or pb <= 0): continue
            if de is not None and de > filter_max_de: continue # Note: D/E can be None if not available
            if roe is not None and roe < (filter_min_roe_pct / 100.0): continue
            if eps_g is not None and eps_g < (filter_min_eps_g_pct / 100.0): continue
            
            if ticker not in hist_data_batch or hist_data_batch[ticker].empty: continue
            df = hist_data_batch[ticker].copy()
            df.dropna(subset=['Adj Close', 'Volume'], inplace=True)
            if len(df) < 50: continue
            if filter_sma50_gt_sma200 and len(df) < 200: continue

            latest_close = df['Adj Close'].iloc[-1]
            latest_volume = df['Volume'].iloc[-1]
            df['EMA20'] = df['Adj Close'].ewm(span=20, adjust=False).mean()
            df['SMA50'] = df['Adj Close'].rolling(window=50).mean()
            if filter_sma50_gt_sma200: df['SMA200'] = df['Adj Close'].rolling(window=200).mean()
            df['AvgVol20'] = df['Volume'].rolling(window=20).mean()
            delta = df['Adj Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            # Handle division by zero for rs
            if loss.iloc[-1] == 0: rs_val = np.inf if gain.iloc[-1] > 0 else 0 
            else: rs_val = gain.iloc[-1] / loss.iloc[-1]
            df['RSI'] = 100 - (100 / (1 + rs_val))


            ema20 = df['EMA20'].iloc[-1]
            sma50 = df['SMA50'].iloc[-1]
            sma200 = df['SMA200'].iloc[-1] if filter_sma50_gt_sma200 and 'SMA200' in df.columns else np.nan
            rsi = df['RSI'].iloc[-1]
            avg_vol20 = df['AvgVol20'].iloc[-1]
            
            if pd.isna(ema20) or pd.isna(sma50) or pd.isna(rsi): continue
            if filter_sma50_gt_sma200 and pd.isna(sma200): continue
            if filter_price_gt_20ema and (latest_close <= ema20): continue
            if filter_price_gt_50sma and (latest_close <= sma50): continue
            if filter_sma50_gt_sma200 and (sma50 <= sma200): continue
            if not (filter_min_rsi <= rsi <= filter_max_rsi): continue
            if filter_apply_vol_buzz:
                if pd.isna(avg_vol20) or avg_vol20 == 0: continue
                if latest_volume < (filter_vol_buzz_factor * avg_vol20): continue
            
            screened_stocks_data.append({
                'Ticker': ticker,
                'Industry': sector_map_dict.get(ticker, info.get('industry', 'N/A')),
                'Price': latest_close, 'P/E': pe, 'P/B': pb, 'D/E': de, # Keep original types here
                'ROE (%)': roe * 100 if roe is not None else np.nan, # Convert to %
                'EPS Growth (%)': eps_g * 100 if eps_g is not None else np.nan, # Convert to %
                'RSI': rsi, 'Volume (M)': latest_volume / 1e6,
                'Avg Vol (M)': avg_vol20 / 1e6 if pd.notna(avg_vol20) else np.nan,
                'Market Cap (Cr)': info.get('marketCap', np.nan) / 1e7 if info.get('marketCap') is not None else np.nan
            })
        except Exception: continue
        
    df_result = pd.DataFrame(screened_stocks_data)
    if not df_result.empty:
        numeric_cols = ['Price', 'P/E', 'P/B', 'D/E', 'ROE (%)', 'EPS Growth (%)', 
                        'RSI', 'Volume (M)', 'Avg Vol (M)', 'Market Cap (Cr)']
        for col in numeric_cols:
            if col in df_result.columns:
                df_result[col] = pd.to_numeric(df_result[col], errors='coerce')
                # Rounding will be handled by Styler's precision or specific formats
    return df_result

with st.spinner("📜 Loading NIFTY 500 list..."):
    tickers_list, sector_map = load_nifty500_list()
if not tickers_list: st.error("Failed to load stock list."); st.stop()

with st.spinner(f"🔎 Screening NIFTY 500 stocks... This might take {int(len(tickers_list)*0.3)}-{int(len(tickers_list)*1)} seconds."): # Adjusted time estimate
    df_screened_raw = run_screener( tuple(tickers_list), sector_map,
        FIXED_MAX_PE, FIXED_MAX_PB, FIXED_MAX_DE, FIXED_MIN_ROE_PERCENT, FIXED_MIN_EPS_GROWTH_PERCENT,
        FIXED_PRICE_GT_20EMA, FIXED_PRICE_GT_50SMA, FIXED_SMA50_GT_SMA200,
        FIXED_MIN_RSI, FIXED_MAX_RSI, FIXED_VOLUME_BUZZ_FACTOR, FIXED_APPLY_VOLUME_BUZZ )

st.markdown(f"---")
st.subheader(f"📊 Screened Stocks ({len(df_screened_raw)} found)")

if df_screened_raw.empty:
    st.info("No stocks matched all the fixed screening criteria.")
else:
    # --- Start Debugging Block ---
    # st.write("--- Debug Info ---")
    # st.write("`df_screened_raw` (first 5 rows):")
    # st.dataframe(df_screened_raw.head())
    # st.write("Columns in `df_screened_raw`:", df_screened_raw.columns.tolist())
    # buffer = io.StringIO()
    # df_screened_raw.info(buf=buffer)
    # s = buffer.getvalue()
    # st.text_area("`df_screened_raw.info()`:", s, height=200)
    # --- End Debugging Block ---

    cols_display_order = ['Industry', 'Price', 'P/E', 'P/B', 'D/E', 'ROE (%)', 'EPS Growth (%)', 
                          'RSI', 'Volume (M)', 'Avg Vol (M)', 'Market Cap (Cr)']
    
    if 'Ticker' not in df_screened_raw.columns:
        st.error("Critical Error: 'Ticker' column missing in df_screened_raw results.")
        st.stop()

    try:
        # Ensure df_screened_raw is a DataFrame and not empty before setting index
        if not isinstance(df_screened_raw, pd.DataFrame) or df_screened_raw.empty:
            st.error("`df_screened_raw` is not a valid DataFrame or is empty before setting index.")
            st.stop()
            
        df_display_intermediate = df_screened_raw.set_index('Ticker')
        # st.write("`df_display_intermediate` (first 5 rows, after set_index):")
        # st.dataframe(df_display_intermediate.head())
        # st.write("Columns in `df_display_intermediate`:", df_display_intermediate.columns.tolist())

        actual_cols_to_display = [col for col in cols_display_order if col in df_display_intermediate.columns]
        # st.write("`actual_cols_to_display` (columns selected for final display):", actual_cols_to_display)
        
        if not actual_cols_to_display:
            st.error("Critical Error: No common columns for display. `actual_cols_to_display` is empty.")
            st.stop()
            
        df_display = df_display_intermediate[actual_cols_to_display].copy()
        # st.write("`df_display` (final for styling, first 5 rows):")
        # st.dataframe(df_display.head())

    except Exception as e:
        st.error(f"Error creating df_display: {e}")
        st.write("Details of df_screened_raw when error occurred:")
        st.dataframe(df_screened_raw)
        st.stop()

    if not isinstance(df_display, pd.DataFrame) or df_display.empty :
        st.warning("`df_display` became invalid or empty after processing. Check data and column selection.")
        st.write("Content of `df_screened_raw` that led to this:")
        st.dataframe(df_screened_raw)
        st.stop()

    # --- MORE DEBUGGING RIGHT BEFORE THE ERROR ---
    # st.write("--- State of `df_display` BEFORE `.style` ---")
    # st.write(f"Is `df_display` a DataFrame? {isinstance(df_display, pd.DataFrame)}")
    # st.write(f"Is `df_display` empty? {df_display.empty}")
    # if isinstance(df_display, pd.DataFrame) and not df_display.empty:
    #     st.write(f"Shape of `df_display`: {df_display.shape}")
    #     st.write("`df_display.head()` right before `.style` call:")
    #     st.dataframe(df_display.head())
    #     buffer_display = io.StringIO()
    #     df_display.info(buf=buffer_display)
    #     s_display = buffer_display.getvalue()
    #     st.text_area("`df_display.info()`:", s_display, height=200)
    # else:
    #     st.error("`df_display` is NOT a valid DataFrame or is empty right before .style call!")
    #     st.stop()
    # --- END MORE DEBUGGING ---

    styler = df_display.style.set_na_rep("-").format(precision=2) 
    df_height = min((len(df_display) + 1) * 35 + 3, 700)
    st.dataframe(styler, use_container_width=True, height=df_height)

    csv = df_screened_raw.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Screened Data as CSV", data=csv,
        file_name=f"nifty500_valuation_uptrend_screener_fixed_{datetime.today().strftime('%Y%m%d')}.csv",
        mime='text/csv')

st.markdown("---")
st.markdown("Disclaimer: Financial data by Yahoo Finance may have inaccuracies/delays. Tool for info, not advice. DYOR.")
