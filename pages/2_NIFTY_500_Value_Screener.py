import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import io # For Excel export

# Ensure Matplotlib's minus sign is rendered correctly
matplotlib.rcParams['axes.unicode_minus'] = False

# --- Helper Functions ---
def to_excel(df):
    output = io.BytesIO()
    df_to_export = df.copy()
    if isinstance(df, pd.io.formats.style.Styler):
        df_to_export = df.data
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_to_export.to_excel(writer, index=False, sheet_name='Valuation_Uptrend_Screen')
    processed_data = output.getvalue()
    return processed_data

def calculate_rsi_manual(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan) # Handle division by zero if loss is 0
    
    rsi = 100 - (100 / (1 + rs))
    
    # If gain is positive and loss is zero, RSI should be 100
    rsi.loc[(gain > 0) & (loss == 0)] = 100
    # If gain is zero and loss is zero (or both NaN), RSI can be neutral 50 or NaN
    rsi.loc[(gain == 0) & (loss == 0)] = 50 # Or np.nan
    rsi.loc[gain.isna() | loss.isna()] = np.nan

    return rsi


st.set_page_config(layout="wide")
st.title("📈 NIFTY 500: Valuation & Uptrend Screener")
st.markdown("Identifies NIFTY 500 stocks with potentially good valuation metrics and signs of an uptrend.")

with st.expander("🧠 **Screening Criteria & Approach**", expanded=True):
    st.markdown("""
    - **Universe**: NIFTY 500 stocks.
    - **Data Source**: `yfinance` for price history and fundamental ratios.
    - **Valuation Metrics (Lower is Better - Target < Median)**:
        - `Price to Earnings Ratio (P/E)`: Using `trailingPE` from yfinance.
        - `Price to Book Ratio (P/B)`: Using `priceToBook` from yfinance.
        - `Price to Sales Ratio (P/S)`: Using `priceToSalesTrailing12Months` from yfinance.
        - *Note*: Stocks with P/E, P/B, P/S below the NIFTY 500 median for that metric are favored. Stocks with negative P/E are excluded from P/E filter.
    - **Uptrend Indicators (Higher/Positive is Better)**:
        - `Relative Strength Index (RSI - 14 day)`: Target between 40 and 70 (avoiding overbought/oversold extremes, indicating steady momentum).
        - `50-Day Exponential Moving Average (50 EMA)`: Current Price > 50 EMA.
        - `200-Day Exponential Moving Average (200 EMA)`: Current Price > 200 EMA (long-term uptrend).
        - `50 EMA vs 200 EMA`: 50 EMA > 200 EMA (golden cross indicator).
    - **Growth/Profitability (Higher is Better - Target > Median)**:
        - `Return on Equity (ROE)`: Using `returnOnEquity` from yfinance.
        - `Earnings Per Share Growth (EPS Growth - TTM vs Previous TTM)`: Using `earningsQuarterlyGrowth` as a proxy (yoy quarterly).
        - `Revenue Growth (Quarterly YoY)`: Using `revenueQuarterlyGrowth` from yfinance.
    - **Filtering Logic**:
        1. Fetch fundamental data and price history for all NIFTY 500 stocks.
        2. Calculate medians for P/E, P/B, P/S, ROE, EPS Growth, Revenue Growth across the NIFTY 500.
        3. **Filter Stocks**:
            - P/E < N500 Median P/E (and P/E > 0)
            - P/B < N500 Median P/B (and P/B > 0)
            - P/S < N500 Median P/S (and P/S > 0)
            - ROE > N500 Median ROE
            - Price > 50 EMA AND Price > 200 EMA AND 50 EMA > 200 EMA
            - 40 < RSI < 70
        - *Optional Filters (User Selectable)*:
            - EPS Growth > N500 Median EPS Growth
            - Revenue Growth > N500 Median Revenue Growth
    - **Displayed Metrics**: Key valuation ratios, trend indicators, growth metrics, and price data.
    - **Robustness**: Handles missing data from `yfinance` gracefully.
    """)

# --- Data Fetching and Processing Functions ---

@st.cache_data(ttl=timedelta(days=1), show_spinner="Loading NIFTY 500 stock list...")
def load_nifty_list_and_tickers():
    csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty500list.csv"
    try:
        df_nifty = pd.read_csv(csv_url)
    except Exception as e:
        st.error(f"Error loading NIFTY 500 stock list from CSV: {e}")
        return [], {}
        
    df_nifty.columns = df_nifty.columns.str.strip()
    df_nifty = df_nifty[~df_nifty['Symbol'].str.contains("DUMMY", na=False)]
    df_nifty['Symbol'] = df_nifty['Symbol'].str.strip()
    df_nifty['Ticker'] = df_nifty['Symbol'] + ".NS"
    return df_nifty['Ticker'].tolist(), dict(zip(df_nifty['Ticker'], df_nifty['Industry']))

@st.cache_data(ttl=timedelta(hours=4), show_spinner="Fetching stock data from yfinance...")
def fetch_stock_data_batch(tickers_tuple):
    tickers_list = list(tickers_tuple)
    if not tickers_list:
        return pd.DataFrame(), pd.DataFrame()

    # Fetch historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=450) # For EMAs and RSI
    
    price_data_dict = {}
    info_data_list = []

    # Batch tickers for yf.Ticker to avoid too many individual calls
    # yf.download is generally more efficient for prices
    # yf.Ticker().info is needed for fundamentals

    st.progress(0)
    total_tickers = len(tickers_list)
    
    # --- Price Data ---
    # Using yf.download for prices is generally more robust for many tickers
    historical_data = yf.download(tickers_list, start=start_date, end=end_date, interval='1d', 
                                   group_by='ticker', auto_adjust=False, progress=False, timeout=120)

    for i, ticker_str in enumerate(tickers_list):
        # --- Fundamental Data ---
        try:
            stock_info = yf.Ticker(ticker_str)
            info = stock_info.info
            # Add ticker to info dict for easy merging later
            info['ticker'] = ticker_str 
            info_data_list.append(info)
        except Exception: # Broad exception for yf.Ticker().info issues
            info_data_list.append({'ticker': ticker_str}) # Append at least the ticker

        # --- Process Price Data ---
        if isinstance(historical_data.columns, pd.MultiIndex): # If multiple tickers downloaded
            if ticker_str in historical_data:
                price_data_dict[ticker_str] = historical_data[ticker_str].copy()
        elif len(tickers_list) == 1 and not historical_data.empty : # Single ticker downloaded
             price_data_dict[ticker_str] = historical_data.copy()
        
        st.progress( (i + 1) / total_tickers )


    df_info = pd.DataFrame(info_data_list)
    return price_data_dict, df_info


@st.cache_data(ttl=timedelta(hours=1), show_spinner="Analyzing stocks...")
def analyze_stocks(price_data_dict_param, df_info_param, industry_map_param):
    processed_results = []
    
    # Make copies to avoid modifying cached data if functions are called multiple times
    price_data_dict = {k: v.copy() for k, v in price_data_dict_param.items()}
    df_info = df_info_param.copy()

    # Calculate EMAs and RSI for all stocks
    for ticker, df_price in price_data_dict.items():
        if df_price.empty or len(df_price) < 200: # Need enough data for 200 EMA
            price_data_dict[ticker] = None # Mark for removal or skip
            continue
        df_price.dropna(subset=['Adj Close'], inplace=True)
        if df_price.empty:
            price_data_dict[ticker] = None
            continue

        df_price['50EMA'] = df_price['Adj Close'].ewm(span=50, adjust=False).mean()
        df_price['200EMA'] = df_price['Adj Close'].ewm(span=200, adjust=False).mean()
        df_price['RSI'] = calculate_rsi_manual(df_price['Adj Close'], period=14)
        price_data_dict[ticker] = df_price # Store modified df

    # Select relevant columns from yf.info and rename for clarity
    valuation_cols = {
        'ticker': 'Ticker',
        'shortName': 'Name',
        'trailingPE': 'P/E',
        'priceToBook': 'P/B',
        'priceToSalesTrailing12Months': 'P/S',
        'returnOnEquity': 'ROE',
        'earningsQuarterlyGrowth': 'EPS_Q_YoY_Growth', # Proxy for EPS TTM growth
        'revenueQuarterlyGrowth': 'Revenue_Q_YoY_Growth',
        'marketCap': 'MarketCap',
        'currentPrice': 'CurrentPrice_Info', # From .info, can differ from latest Adj Close
        'regularMarketVolume': 'AvgVolume_Info'
    }
    
    df_fundamentals = df_info[[col for col in valuation_cols.keys() if col in df_info.columns]].copy()
    df_fundamentals.rename(columns=valuation_cols, inplace=True)
    
    # Add industry
    df_fundamentals['Industry'] = df_fundamentals['Ticker'].map(industry_map_param).fillna('Unknown')


    # Calculate Medians for filtering (only on valid, positive values for ratios)
    median_pe = df_fundamentals[df_fundamentals['P/E'] > 0]['P/E'].median()
    median_pb = df_fundamentals[df_fundamentals['P/B'] > 0]['P/B'].median()
    median_ps = df_fundamentals[df_fundamentals['P/S'] > 0]['P/S'].median()
    median_roe = df_fundamentals['ROE'].median() # ROE can be negative
    median_eps_g = df_fundamentals['EPS_Q_YoY_Growth'].median()
    median_rev_g = df_fundamentals['Revenue_Q_YoY_Growth'].median()

    # Store medians for display
    st.session_state.medians = {
        "Median P/E (N500)": median_pe,
        "Median P/B (N500)": median_pb,
        "Median P/S (N500)": median_ps,
        "Median ROE (N500)": median_roe,
        "Median EPS Q YoY Growth (N500)": median_eps_g,
        "Median Revenue Q YoY Growth (N500)": median_rev_g,
    }

    for idx, fund_row in df_fundamentals.iterrows():
        ticker = fund_row['Ticker']
        df_price = price_data_dict.get(ticker)

        if df_price is None or df_price.empty:
            continue
        
        latest_price_row = df_price.iloc[-1]
        current_close = latest_price_row['Adj Close']
        
        # Trend conditions
        price_gt_50ema = current_close > latest_price_row['50EMA'] if pd.notna(latest_price_row['50EMA']) else False
        price_gt_200ema = current_close > latest_price_row['200EMA'] if pd.notna(latest_price_row['200EMA']) else False
        ema50_gt_ema200 = latest_price_row['50EMA'] > latest_price_row['200EMA'] if pd.notna(latest_price_row['50EMA']) and pd.notna(latest_price_row['200EMA']) else False
        rsi_val = latest_price_row['RSI']
        rsi_ok = (40 < rsi_val < 70) if pd.notna(rsi_val) else False

        # Valuation conditions
        pe_val = fund_row.get('P/E')
        pb_val = fund_row.get('P/B')
        ps_val = fund_row.get('P/S')
        roe_val = fund_row.get('ROE')
        eps_g_val = fund_row.get('EPS_Q_YoY_Growth')
        rev_g_val = fund_row.get('Revenue_Q_YoY_Growth')

        pe_ok = (0 < pe_val < median_pe) if pd.notna(pe_val) and pd.notna(median_pe) else False
        pb_ok = (0 < pb_val < median_pb) if pd.notna(pb_val) and pd.notna(median_pb) else False
        ps_ok = (0 < ps_val < median_ps) if pd.notna(ps_val) and pd.notna(median_ps) else False
        roe_ok = (roe_val > median_roe) if pd.notna(roe_val) and pd.notna(median_roe) else False
        
        # Optional growth filters
        eps_g_ok_optional = (eps_g_val > median_eps_g) if pd.notna(eps_g_val) and pd.notna(median_eps_g) else False
        rev_g_ok_optional = (rev_g_val > median_rev_g) if pd.notna(rev_g_val) and pd.notna(median_rev_g) else False

        # Combine data
        data_point = {
            'Ticker': ticker,
            'Name': fund_row.get('Name', 'N/A'),
            'Industry': fund_row.get('Industry', 'N/A'),
            'Price': round(current_close,2) if pd.notna(current_close) else np.nan,
            'P/E': round(pe_val,2) if pd.notna(pe_val) else np.nan,
            'P/B': round(pb_val,2) if pd.notna(pb_val) else np.nan,
            'P/S': round(ps_val,2) if pd.notna(ps_val) else np.nan,
            'ROE': f"{round(roe_val*100,2)}%" if pd.notna(roe_val) else np.nan,
            'EPS_Q_YoY_Growth': f"{round(eps_g_val*100,2)}%" if pd.notna(eps_g_val) else np.nan,
            'Revenue_Q_YoY_Growth': f"{round(rev_g_val*100,2)}%" if pd.notna(rev_g_val) else np.nan,
            'RSI': round(rsi_val,2) if pd.notna(rsi_val) else np.nan,
            'Price_vs_50EMA': 'Above' if price_gt_50ema else 'Below',
            'Price_vs_200EMA': 'Above' if price_gt_200ema else 'Below',
            '50EMA_vs_200EMA': 'Above' if ema50_gt_ema200 else 'Below',
            'MarketCap_Cr': round(fund_row.get('MarketCap', 0) / 10**7, 2) if pd.notna(fund_row.get('MarketCap')) else np.nan,
            # Store filter pass/fail for easier filtering later
            '_pe_ok': pe_ok, '_pb_ok': pb_ok, '_ps_ok': ps_ok, '_roe_ok': roe_ok,
            '_trend_ok': price_gt_50ema and price_gt_200ema and ema50_gt_ema200,
            '_rsi_ok': rsi_ok,
            '_eps_g_ok_opt': eps_g_ok_optional, '_rev_g_ok_opt': rev_g_ok_optional
        }
        processed_results.append(data_point)

    return pd.DataFrame(processed_results)

# --- Main App UI & Logic ---

tickers_n500, industry_map_n500 = load_nifty_list_and_tickers()

if not tickers_n500:
    st.error("NIFTY 500 stock list could not be loaded. Screener cannot run.")
    st.stop()

price_data_historical, info_data_fundamentals = fetch_stock_data_batch(tuple(tickers_n500))

if info_data_fundamentals.empty and not price_data_historical:
    st.error("Failed to fetch critical stock data. Please try again later.")
    st.stop()
elif info_data_fundamentals.empty:
    st.warning("Fundamental data could not be fetched for many stocks. Valuation screening will be limited.")
elif not price_data_historical:
     st.warning("Price history data could not be fetched for many stocks. Trend analysis will be limited.")


df_analyzed = analyze_stocks(price_data_historical, info_data_fundamentals, industry_map_n500)

st.sidebar.header("📊 N500 Median Values")
if 'medians' in st.session_state:
    for key, val in st.session_state.medians.items():
        st.sidebar.metric(label=key, value=f"{val:.2f}" if isinstance(val, (int, float)) and pd.notna(val) else "N/A")
else:
    st.sidebar.info("Median values will be calculated after data processing.")

st.sidebar.header("⚙️ Optional Growth Filters")
filter_eps_growth = st.sidebar.checkbox("EPS Growth > N500 Median", value=True)
filter_revenue_growth = st.sidebar.checkbox("Revenue Growth > N500 Median", value=True)


if df_analyzed.empty:
    st.warning("No stocks could be analyzed. This might be due to data fetching issues or all stocks being filtered out early.")
    st.stop()

# Apply filters
conditions = (
    df_analyzed['_pe_ok']) & \
   (df_analyzed['_pb_ok']) & \
   (df_analyzed['_ps_ok']) & \
   (df_analyzed['_roe_ok']) & \
   (df_analyzed['_trend_ok']) & \
   (df_analyzed['_rsi_ok']
)

if filter_eps_growth:
    conditions &= df_analyzed['_eps_g_ok_opt']
if filter_revenue_growth:
    conditions &= df_analyzed['_rev_g_ok_opt']

df_screened = df_analyzed[conditions].copy()

# Drop internal filter columns before display
cols_to_drop_internal = [col for col in df_screened.columns if col.startswith('_')]
df_screened.drop(columns=cols_to_drop_internal, inplace=True, errors='ignore')


st.markdown(f"### 🌟 Screened Stocks ({len(df_screened)} found)")

if df_screened.empty:
    st.info("No stocks found matching all criteria. Try adjusting optional filters or checking back later.")
else:
    display_cols_order = [
        'Ticker', 'Name', 'Industry', 'Price', 'P/E', 'P/B', 'P/S', 'ROE', 
        'RSI', 'Price_vs_50EMA', '50EMA_vs_200EMA', 'MarketCap_Cr',
        'EPS_Q_YoY_Growth', 'Revenue_Q_YoY_Growth'
    ]
    df_display = df_screened[[col for col in display_cols_order if col in df_screened.columns]]
    
    # Convert MarketCap to float for formatting if it's not already
    if 'MarketCap_Cr' in df_display.columns:
        df_display['MarketCap_Cr'] = pd.to_numeric(df_display['MarketCap_Cr'], errors='coerce')

    styler = df_display.style.format({
        'Price': "{:.2f}", 'P/E': "{:.2f}", 'P/B': "{:.2f}", 'P/S': "{:.2f}",
        'RSI': "{:.2f}", 'MarketCap_Cr': "{:,.2f} Cr"
    }, na_rep="-")
    
    # Highlight good valuation (e.g., P/E < median)
    def highlight_valuation(val, median_val):
        if pd.notna(val) and pd.notna(median_val) and 0 < val < median_val:
            return 'background-color: lightgreen'
        return ''

    if 'medians' in st.session_state:
        styler = styler.applymap(lambda x: highlight_valuation(x, st.session_state.medians.get("Median P/E (N500)")), subset=['P/E'])
        styler = styler.applymap(lambda x: highlight_valuation(x, st.session_state.medians.get("Median P/B (N500)")), subset=['P/B'])
        styler = styler.applymap(lambda x: highlight_valuation(x, st.session_state.medians.get("Median P/S (N500)")), subset=['P/S'])
    
    df_height = min((len(df_display) + 1) * 35 + 3, 700)
    st.dataframe(styler, use_container_width=True, height=df_height)

    excel_data = to_excel(df_display)
    st.download_button(
        label="📥 Download Screened Data as Excel",
        data=excel_data,
        file_name=f"N500_Valuation_Uptrend_Screened_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- Additional Visualizations (Optional) ---
if not df_screened.empty:
    st.markdown("### 📊 Visualizations for Screened Stocks")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### P/E Ratio Distribution")
        fig_pe, ax_pe = plt.subplots(figsize=(8, 4))
        sns.histplot(df_screened['P/E'].dropna(), bins=20, kde=True, color='skyblue', ax=ax_pe)
        if 'medians' in st.session_state and pd.notna(st.session_state.medians.get("Median P/E (N500)")):
            ax_pe.axvline(st.session_state.medians["Median P/E (N500)"], color='red', linestyle='--', label=f'N500 Median P/E ({st.session_state.medians["Median P/E (N500)"]:.2f})')
        ax_pe.set_xlabel("P/E Ratio")
        ax_pe.set_ylabel("Frequency")
        ax_pe.legend()
        st.pyplot(fig_pe)

    with col2:
        st.markdown("#### RSI Distribution")
        fig_rsi, ax_rsi = plt.subplots(figsize=(8, 4))
        sns.histplot(df_screened['RSI'].dropna(), bins=15, kde=True, color='salmon', ax=ax_rsi)
        ax_rsi.axvline(40, color='blue', linestyle='--', label='Target Min (40)')
        ax_rsi.axvline(70, color='blue', linestyle='--', label='Target Max (70)')
        ax_rsi.set_xlabel("RSI")
        ax_rsi.set_ylabel("Frequency")
        ax_rsi.legend()
        st.pyplot(fig_rsi)

st.markdown("---")
st.markdown("Disclaimer: Financial data from `yfinance` can have inaccuracies or be delayed. This tool is for informational purposes only and not financial advice. Always perform your own due diligence.")
