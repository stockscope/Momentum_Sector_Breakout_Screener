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
st.title("📈 NIFTY 500: Valuation & Uptrend Screener")
st.markdown("Identifies NIFTY 500 stocks with potentially good valuation and signs of an uptrend.")

with st.expander("🧠 **Screening Philosophy & Criteria**", expanded=True):
    st.markdown("""
    This screener combines fundamental valuation metrics with technical trend indicators to find potentially undervalued stocks that are showing signs of an upward price movement.

    **Fundamental Filters (User Adjustable):**
    - **Max P/E Ratio (Trailing):** Price-to-Earnings ratio. Lower can indicate undervaluation.
    - **Max P/B Ratio:** Price-to-Book ratio. Lower can indicate undervaluation.
    - **Max Debt-to-Equity Ratio:** Measures financial leverage. Lower is generally better.
    - **Min ROE (Return on Equity):** Measures profitability. Higher is generally better.
    - **Min EPS Growth (Quarterly, YoY):** Indicates earnings growth.

    **Technical Trend Filters (User Adjustable):**
    - **Price vs Moving Averages:**
        - Price > 20-Day Exponential Moving Average (EMA)
        - Price > 50-Day Simple Moving Average (SMA)
        - (Optional) 50-Day SMA > 200-Day SMA (Golden Cross - Slower signal)
    - **RSI (14-day):** Relative Strength Index.
        - Min RSI / Max RSI: To filter for stocks in a healthy momentum range (e.g., 40-70).
    - **Volume:**
        - Current Volume > 1.2x of 20-Day Average Volume (Volume Buzz)

    **Note:** Fundamental data from `yfinance` can sometimes be missing or delayed. Technical signals are based on historical price data.
    """)

# --- Helper Functions ---
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
    except Exception as e:
        # st.sidebar.warning(f"Error in yf.download: {str(e)[:100]}")
        return {}


@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def get_stock_info(ticker_str):
    try:
        stock = yf.Ticker(ticker_str)
        info = stock.info
        # Select only a few key metrics to avoid overly large cache objects / too much data
        # Add more as needed, but be mindful of yfinance's data availability
        keys_to_extract = [
            'trailingPE', 'forwardPE', 'priceToBook', 'debtToEquity', 
            'returnOnEquity', 'earningsQuarterlyGrowth', 'beta', 
            'marketCap', 'industry', 'sector', 'previousClose', 'volume', 'averageVolume'
        ]
        filtered_info = {k: info.get(k) for k in keys_to_extract if info.get(k) is not None}
        return filtered_info
    except Exception:
        return {} # Return empty dict on error

# --- Sidebar for User Filters ---
st.sidebar.header("⚙️ Set Screening Filters")

st.sidebar.subheader("Fundamental Filters")
max_pe = st.sidebar.slider("Max P/E Ratio (Trailing)", 0.1, 100.0, 30.0, 0.1, help="Set to 100 to effectively ignore.")
max_pb = st.sidebar.slider("Max P/B Ratio", 0.1, 20.0, 5.0, 0.1, help="Set to 20 to effectively ignore.")
max_de = st.sidebar.slider("Max Debt-to-Equity Ratio", 0.0, 5.0, 1.5, 0.01, help="Set to 5 to effectively ignore.")
min_roe = st.sidebar.slider("Min ROE (%)", -50.0, 100.0, 15.0, 0.1, help="Return on Equity. Set to -50 to ignore.")
min_eps_growth = st.sidebar.slider("Min EPS Growth (Quarterly YoY, %)", -100.0, 200.0, 5.0, 0.1, help="Set to -100 to ignore.")


st.sidebar.subheader("Technical Trend Filters")
price_gt_20ema = st.sidebar.checkbox("Price > 20 EMA", value=True)
price_gt_50sma = st.sidebar.checkbox("Price > 50 SMA", value=True)
sma50_gt_sma200 = st.sidebar.checkbox("50 SMA > 200 SMA (Golden Cross)", value=False) # Slower signal

min_rsi = st.sidebar.slider("Min RSI (14-day)", 0, 100, 40, 1)
max_rsi = st.sidebar.slider("Max RSI (14-day)", 0, 100, 70, 1)
if min_rsi > max_rsi:
    st.sidebar.warning("Min RSI cannot be greater than Max RSI.")
    st.stop()

volume_buzz_factor = st.sidebar.slider("Volume Buzz Factor (Current Vol / Avg Vol)", 1.0, 5.0, 1.2, 0.1, help="E.g., 1.2 means current volume > 1.2x average volume.")
apply_volume_buzz = st.sidebar.checkbox("Apply Volume Buzz Filter", value=True)


# --- Main Screening Logic ---
@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def run_screener(tickers_list_tuple, sector_map_dict, 
                 filter_max_pe, filter_max_pb, filter_max_de, filter_min_roe, filter_min_eps_g,
                 filter_price_gt_20ema, filter_price_gt_50sma, filter_sma50_gt_sma200,
                 filter_min_rsi, filter_max_rsi, filter_vol_buzz_factor, filter_apply_vol_buzz):
    
    screened_stocks = []
    end_date_dt = datetime.today()
    start_date_dt = end_date_dt - timedelta(days=365 * 1.5) # For 200 DMA
    
    # Batch fetch historical data
    hist_data_batch = fetch_stock_data(tickers_list_tuple, start_date_dt.strftime('%Y-%m-%d'), end_date_dt.strftime('%Y-%m-%d'))

    for ticker in tickers_list_tuple:
        try:
            # 1. Fundamental Data
            info = get_stock_info(ticker) # This is cached per ticker

            pe = info.get('trailingPE')
            pb = info.get('priceToBook')
            de = info.get('debtToEquity')
            roe = info.get('returnOnEquity')
            eps_g = info.get('earningsQuarterlyGrowth') # This is often YoY quarterly growth

            # Apply Fundamental Filters (allow None to pass if filter is set to max/min effectively ignoring it)
            if filter_max_pe < 100.0 and (pe is None or pe > filter_max_pe or pe <=0): continue # Ignore negative or zero P/E by default with >0 check
            if filter_max_pb < 20.0 and (pb is None or pb > filter_max_pb or pb <=0): continue
            if filter_max_de < 5.0 and (de is None or de > filter_max_de): continue
            if filter_min_roe > -50.0 and (roe is None or roe < (filter_min_roe / 100.0)): continue # ROE is decimal in yf
            if filter_min_eps_g > -100.0 and (eps_g is None or eps_g < (filter_min_eps_g / 100.0)): continue # EPS Growth is decimal

            # 2. Technical Data
            if ticker not in hist_data_batch or hist_data_batch[ticker].empty:
                continue
            
            df = hist_data_batch[ticker].copy()
            df.dropna(subset=['Adj Close', 'Volume'], inplace=True)
            if len(df) < 200: continue # Need enough data for longest MA

            latest_close = df['Adj Close'].iloc[-1]
            latest_volume = df['Volume'].iloc[-1]

            df['EMA20'] = df['Adj Close'].ewm(span=20, adjust=False).mean()
            df['SMA50'] = df['Adj Close'].rolling(window=50).mean()
            df['SMA200'] = df['Adj Close'].rolling(window=200).mean()
            df['AvgVol20'] = df['Volume'].rolling(window=20).mean()

            # RSI
            delta = df['Adj Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Get latest technical values
            ema20 = df['EMA20'].iloc[-1]
            sma50 = df['SMA50'].iloc[-1]
            sma200 = df['SMA200'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            avg_vol20 = df['AvgVol20'].iloc[-1]
            
            if pd.isna(ema20) or pd.isna(sma50) or pd.isna(rsi): continue # Essential technicals
            if filter_sma50_gt_sma200 and pd.isna(sma200): continue


            # Apply Technical Filters
            if filter_price_gt_20ema and (latest_close <= ema20): continue
            if filter_price_gt_50sma and (latest_close <= sma50): continue
            if filter_sma50_gt_sma200 and (sma50 <= sma200): continue
            
            if not (filter_min_rsi <= rsi <= filter_max_rsi): continue
            
            if filter_apply_vol_buzz:
                if pd.isna(avg_vol20) or avg_vol20 == 0: continue # Cannot calculate buzz
                if latest_volume < (filter_vol_buzz_factor * avg_vol20): continue
            
            screened_stocks.append({
                'Ticker': ticker,
                'Industry': sector_map_dict.get(ticker, info.get('industry', 'N/A')),
                'Price': round(latest_close, 2),
                'P/E': round(pe, 2) if pe is not None else 'N/A',
                'P/B': round(pb, 2) if pb is not None else 'N/A',
                'D/E': round(de, 2) if de is not None else 'N/A',
                'ROE (%)': round(roe * 100, 2) if roe is not None else 'N/A',
                'EPS Growth (%)': round(eps_g * 100, 2) if eps_g is not None else 'N/A',
                'RSI': round(rsi, 2),
                'Volume (M)': round(latest_volume / 1e6, 2),
                'Avg Vol (M)': round(avg_vol20 / 1e6, 2) if pd.notna(avg_vol20) else 'N/A',
                'Market Cap (Cr)': round(info.get('marketCap', 0) / 1e7, 2) if info.get('marketCap') else 'N/A' # Crores
            })

        except Exception as e:
            # st.sidebar.warning(f"Error processing {ticker}: {str(e)[:50]}")
            continue
            
    return pd.DataFrame(screened_stocks)


# --- Load Initial Data ---
with st.spinner("📜 Loading NIFTY 500 list..."):
    tickers_list, sector_map = load_nifty500_list()

if not tickers_list:
    st.error("Failed to load stock list. Screener cannot run.")
    st.stop()

# --- Run Screener on Button Click or Automatically ---
# For simplicity, let's run it automatically when filters change (Streamlit's natural behavior)
# If it's too slow, a button could be added.

with st.spinner(f"🔎 Screening NIFTY 500 stocks... This might take a few minutes for {len(tickers_list)} stocks."):
    df_screened = run_screener(tuple(tickers_list), sector_map, # Pass tuple for caching
                               max_pe, max_pb, max_de, min_roe, min_eps_growth,
                               price_gt_20ema, price_gt_50sma, sma50_gt_sma200,
                               min_rsi, max_rsi, volume_buzz_factor, apply_volume_buzz)

st.markdown(f"---")
st.subheader(f"📊 Screened Stocks ({len(df_screened)} found)")

if df_screened.empty:
    st.info("No stocks matched all the selected criteria. Try adjusting the filters.")
else:
    # Define display order - Ticker will be index
    cols_display_order = ['Industry', 'Price', 'P/E', 'P/B', 'D/E', 'ROE (%)', 'EPS Growth (%)', 
                          'RSI', 'Volume (M)', 'Avg Vol (M)', 'Market Cap (Cr)']
    
    df_display = df_screened.set_index('Ticker')[cols_display_order].copy()

    # Formatting for display
    # Since data is already rounded and has 'N/A', specific Styler formats are less critical here for decimals
    # but useful for alignment or other styling.
    styler = df_display.style.set_na_rep("-") # Handles any remaining NaNs
    
    # Example of conditional formatting (optional)
    # def color_pe(val):
    #     color = 'green' if isinstance(val, (int, float)) and val < 15 else 'black'
    #     return f'color: {color}'
    # styler = styler.applymap(color_pe, subset=['P/E'])

    df_height = min((len(df_display) + 1) * 35 + 3, 700) # Max height 700px
    st.dataframe(styler, use_container_width=True, height=df_height)

    csv = df_screened.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Screened Data as CSV",
        data=csv,
        file_name=f"nifty500_valuation_uptrend_screener_{datetime.today().strftime('%Y%m%d')}.csv",
        mime='text/csv',
    )

st.markdown("---")
st.markdown("Disclaimer: Financial data provided by Yahoo Finance can have inaccuracies or delays. This tool is for informational purposes only and not financial advice. Always do your own research (DYOR).")
