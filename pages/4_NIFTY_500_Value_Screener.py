import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import io 

matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide")
st.title("📈 NIFTY 500: Fair Value & Uptrend Screener (Relaxed Criteria)")
st.markdown("Identifies NIFTY 500 stocks with more lenient valuation and uptrend signals.")

# --- Define Fixed Screening Criteria (FULL RELAXED) ---
# Fundamental (More Relaxed)
FIXED_MAX_PE = 60.0       
FIXED_MAX_PEG = 3.0       
FIXED_MAX_PB = 10.0       
FIXED_MAX_DE = 2.5        
FIXED_MIN_ROE_PERCENT = 5.0 
FIXED_MIN_EPS_GROWTH_PERCENT = -10.0 

# Technical (More Relaxed)
FIXED_PRICE_GT_50SMA = True     
FIXED_PRICE_GT_200SMA = False   # RELAXED 
FIXED_SMA50_GT_SMA200 = False   # RELAXED 
FIXED_MIN_RSI = 30              
FIXED_MAX_RSI = 85              
FIXED_APPLY_VOLUME_BUZZ = True  
FIXED_VOLUME_BUZZ_FACTOR = 1.0  

with st.expander("🧠 **Screening Philosophy & Fixed Criteria Used (Relaxed)**", expanded=True):
    st.markdown(f"""
    This screener uses **relaxed criteria** to identify a broader range of companies that might offer reasonable value and are showing some signs of an uptrend.

    **Fundamental Filters (Relaxed):**
    - **Max P/E Ratio (Trailing):** {FIXED_MAX_PE}
    - **Max PEG Ratio:** {FIXED_MAX_PEG}
    - **Max P/B Ratio:** {FIXED_MAX_PB}
    - **Max Debt-to-Equity Ratio:** {FIXED_MAX_DE}
    - **Min ROE (Return on Equity):** {FIXED_MIN_ROE_PERCENT}%
    - **Min EPS Growth (Quarterly, YoY):** {FIXED_MIN_EPS_GROWTH_PERCENT}%

    **Technical Trend Filters (Relaxed):**
    - **Price > 50-Day SMA:** {'Yes' if FIXED_PRICE_GT_50SMA else 'No'}
    - **Price > 200-Day SMA:** {'Yes' if FIXED_PRICE_GT_200SMA else 'No'}
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
                           interval='1d', group_by='ticker', auto_adjust=False, progress=False, timeout=90)
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
            'trailingPE', 'forwardPE', 'priceToBook', 'debtToEquity', 'pegRatio',
            'returnOnEquity', 'earningsQuarterlyGrowth', 'beta', 
            'marketCap', 'industry', 'sector', 'previousClose', 'volume', 'averageVolume'
        ]
        filtered_info = {k: info.get(k) for k in keys_to_extract} 
        return filtered_info
    except Exception:
        return {}

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def run_screener(tickers_list_tuple, sector_map_dict, 
                 filter_max_pe, filter_max_peg, filter_max_pb, filter_max_de, filter_min_roe_pct, filter_min_eps_g_pct,
                 filter_price_gt_50sma, filter_price_gt_200sma, filter_sma50_gt_sma200,
                 filter_min_rsi, filter_max_rsi, filter_vol_buzz_factor, filter_apply_vol_buzz):
    screened_stocks_data = []
    end_date_dt = datetime.today()
    start_date_dt = end_date_dt - timedelta(days=400) 
    hist_data_batch = fetch_stock_data(tickers_list_tuple, start_date_dt.strftime('%Y-%m-%d'), end_date_dt.strftime('%Y-%m-%d'))

    for ticker in tickers_list_tuple:
        try:
            info = get_stock_info(ticker)
            pe = info.get('trailingPE')
            peg = info.get('pegRatio')
            pb = info.get('priceToBook')
            de = info.get('debtToEquity') 
            roe = info.get('returnOnEquity')
            eps_g = info.get('earningsQuarterlyGrowth')

            if pe is not None and (pe > filter_max_pe or pe <= 0): continue
            if peg is not None and (peg > filter_max_peg or peg <= 0): continue
            if pb is not None and (pb > filter_max_pb or pb <= 0): continue
            if de is not None and de > filter_max_de: continue 
            if roe is not None and roe < (filter_min_roe_pct / 100.0): continue
            if eps_g is not None and eps_g < (filter_min_eps_g_pct / 100.0): continue
            
            if ticker not in hist_data_batch or hist_data_batch[ticker].empty: continue
            df = hist_data_batch[ticker].copy()
            df.dropna(subset=['Adj Close', 'High', 'Volume'], inplace=True)
            
            min_days_for_200ma = 200
            min_days_for_50ma = 50
            min_days_for_52wh = 200 

            if len(df) < min_days_for_50ma: continue
            # Only require 200 days if the specific filters needing 200MA are active
            if (filter_sma50_gt_sma200 or filter_price_gt_200sma) and len(df) < min_days_for_200ma: continue

            latest_close = df['Adj Close'].iloc[-1]
            latest_volume = df['Volume'].iloc[-1]
            
            df['SMA50'] = df['Adj Close'].rolling(window=50, min_periods=min_days_for_50ma-10).mean() 
            df['SMA200'] = df['Adj Close'].rolling(window=200, min_periods=min_days_for_200ma-20).mean() if len(df) >= min_days_for_200ma else np.nan
            df['52W_High_Val'] = df['High'].rolling(window=252, min_periods=min_days_for_52wh-20).max() if len(df) >= min_days_for_52wh else np.nan
            df['AvgVol20'] = df['Volume'].rolling(window=20, min_periods=15).mean()
            
            delta = df['Adj Close'].diff(1)
            gain_series = delta.where(delta > 0, 0.0).rolling(window=14, min_periods=1).mean()
            loss_series = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=1).mean()

            latest_gain = gain_series.iloc[-1]
            latest_loss = loss_series.iloc[-1]
            
            current_rsi = 50.0 
            if pd.notna(latest_gain) and pd.notna(latest_loss):
                if latest_loss == 0:
                    current_rsi = 100.0 if latest_gain > 0 else 50.0 
                else:
                    rs = latest_gain / latest_loss
                    current_rsi = 100.0 - (100.0 / (1.0 + rs))
            elif pd.notna(latest_gain) and latest_gain > 0: 
                 current_rsi = 100.0

            sma50 = df['SMA50'].iloc[-1]
            sma200 = df['SMA200'].iloc[-1] if 'SMA200' in df.columns and pd.notna(df['SMA200'].iloc[-1]) else np.nan
            latest_52w_high = df['52W_High_Val'].iloc[-1] if '52W_High_Val' in df.columns and pd.notna(df['52W_High_Val'].iloc[-1]) else np.nan
            avg_vol20 = df['AvgVol20'].iloc[-1]
            
            # Check essential technicals that are always calculated
            if pd.isna(sma50) or pd.isna(current_rsi) or pd.isna(avg_vol20): continue
            # Check technicals that depend on filters
            if (filter_sma50_gt_sma200 or filter_price_gt_200sma) and pd.isna(sma200): continue

            if filter_price_gt_50sma and (latest_close <= sma50): continue
            if filter_price_gt_200sma and (latest_close <= sma200): continue # Will only apply if sma200 is valid
            if filter_sma50_gt_sma200 and (sma50 <= sma200): continue # Will only apply if sma200 is valid
            if not (filter_min_rsi <= current_rsi <= filter_max_rsi): continue
            if filter_apply_vol_buzz:
                if avg_vol20 == 0: continue 
                if latest_volume < (filter_vol_buzz_factor * avg_vol20): continue
            
            dist_from_52wh = ((latest_close - latest_52w_high) / latest_52w_high) * 100 if pd.notna(latest_52w_high) and latest_52w_high > 0 else np.nan

            screened_stocks_data.append({
                'Ticker': ticker,
                'Industry': sector_map_dict.get(ticker, info.get('industry', 'N/A')),
                'Price': latest_close, 'P/E': pe, 'PEG': peg, 'P/B': pb, 'D/E': de,
                'ROE (%)': roe * 100 if roe is not None else np.nan,
                'EPS Gr. (%)': eps_g * 100 if eps_g is not None else np.nan,
                'RSI': current_rsi, 
                'Dist 52WH (%)': dist_from_52wh,
                'Vol (M)': latest_volume / 1e6,
                'AvgVol (M)': avg_vol20 / 1e6 if pd.notna(avg_vol20) else np.nan,
                'MCap (Cr)': info.get('marketCap', np.nan) / 1e7 if info.get('marketCap') is not None else np.nan
            })
        except Exception: 
            continue
        
    df_result = pd.DataFrame(screened_stocks_data)
    if not df_result.empty:
        numeric_cols = ['Price', 'P/E', 'PEG', 'P/B', 'D/E', 'ROE (%)', 'EPS Gr. (%)', 
                        'RSI', 'Dist 52WH (%)', 'Vol (M)', 'AvgVol (M)', 'MCap (Cr)']
        for col in numeric_cols:
            if col in df_result.columns:
                df_result[col] = pd.to_numeric(df_result[col], errors='coerce')
    return df_result

# --- Main App Logic ---
with st.spinner("📜 Loading NIFTY 500 list..."):
    tickers_list, sector_map = load_nifty500_list()
if not tickers_list: st.error("Failed to load stock list."); st.stop()

with st.spinner(f"🔎 Screening NIFTY 500 stocks with relaxed criteria... This can take several minutes."):
    df_screened_raw = run_screener( 
        tuple(tickers_list), sector_map,
        FIXED_MAX_PE, FIXED_MAX_PEG, FIXED_MAX_PB, FIXED_MAX_DE, FIXED_MIN_ROE_PERCENT, FIXED_MIN_EPS_GROWTH_PERCENT,
        FIXED_PRICE_GT_50SMA, FIXED_PRICE_GT_200SMA, FIXED_SMA50_GT_SMA200,
        FIXED_MIN_RSI, FIXED_MAX_RSI, FIXED_VOLUME_BUZZ_FACTOR, FIXED_APPLY_VOLUME_BUZZ 
    )

st.markdown(f"---")
st.subheader(f"📊 Screened Stocks ({len(df_screened_raw)} found using relaxed criteria)")

if df_screened_raw.empty:
    st.info("No stocks matched even the relaxed screening criteria. Market conditions might be unusual or fundamental data coverage is sparse.")
else:
    cols_display_order = ['Industry', 'Price', 'P/E','PEG', 'P/B', 'D/E', 'ROE (%)', 'EPS Gr. (%)', 
                          'RSI', 'Dist 52WH (%)', 'Vol (M)', 'AvgVol (M)', 'MCap (Cr)']
    
    if 'Ticker' not in df_screened_raw.columns:
        st.error("Critical Error: 'Ticker' column missing in df_screened_raw results.")
        st.stop()

    try:
        if not isinstance(df_screened_raw, pd.DataFrame) or df_screened_raw.empty:
            st.error("`df_screened_raw` is not valid or empty before setting index.")
            st.stop()
        df_display_intermediate = df_screened_raw.set_index('Ticker')
        actual_cols_to_display = [col for col in cols_display_order if col in df_display_intermediate.columns]
        if not actual_cols_to_display:
            st.error("Critical Error: No common columns for display.")
            st.stop()
        df_display = df_display_intermediate[actual_cols_to_display].copy()
    except Exception as e:
        st.error(f"Error creating df_display: {e}")
        st.dataframe(df_screened_raw)
        st.stop()

    if not isinstance(df_display, pd.DataFrame) or df_display.empty :
        st.warning("`df_display` became invalid or empty. Check data.")
        st.dataframe(df_screened_raw)
        st.stop()
    
    # Optional: Debug df_display types before styling (uncomment if issues persist)
    # buffer_display = io.StringIO()
    # df_display.info(buf=buffer_display)
    # s_display = buffer_display.getvalue()
    # st.text_area("`df_display.info()` before styling:", s_display, height=300)

    try:
        float_cols_to_format = {
            'Price': "{:.2f}", 'P/E': "{:.2f}", 'PEG': "{:.2f}", 'P/B': "{:.2f}", 'D/E': "{:.2f}",
            'ROE (%)': "{:.2f}", 'EPS Gr. (%)': "{:.2f}", 'RSI': "{:.2f}",
            'Dist 52WH (%)': "{:.2f}", 
            'Vol (M)': "{:.2f}", 'AvgVol (M)': "{:.2f}", 'MCap (Cr)': "{:.2f}"
        }
        actual_formats = {}
        for col_name, fmt_str in float_cols_to_format.items():
            if col_name in df_display.columns and pd.api.types.is_numeric_dtype(df_display[col_name]):
                actual_formats[col_name] = fmt_str
        
        styler = df_display.style.format(formatter=actual_formats if actual_formats else None, na_rep="-")
        
    except AttributeError as ae: 
        st.error(f"AttributeError during styling: {ae}. `df_display` might be invalid.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during styling: {e}")
        import traceback
        st.text_area("Styling Traceback:", traceback.format_exc(), height=300)
        st.stop()
        
    df_height = min((len(df_display) + 1) * 35 + 3, 700) # Dynamic height
    st.dataframe(styler, use_container_width=True, height=df_height)

    csv = df_screened_raw.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Screened Data as CSV", data=csv,
        file_name=f"nifty500_fair_value_uptrend_relaxed_{datetime.today().strftime('%Y%m%d')}.csv", # Updated filename
        mime='text/csv')

st.markdown("---")
st.markdown("Disclaimer: Financial data by Yahoo Finance may have inaccuracies/delays. Tool for info, not advice. DYOR.")
