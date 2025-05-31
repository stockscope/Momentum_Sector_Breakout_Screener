# pages/6_Strategy_Backtester.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from stqdm import stqdm 
import io 
import importlib # For dynamic imports
from pathlib import Path # For listing pages

st.set_page_config(
    page_title="Universal Strategy Backtester - StockScopePro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛠️ Universal Strategy Backtester")
st.markdown("Test the historical performance of your defined screener strategies.")

# --- Helper Functions (load_index_list, fetch_historical_data - keep as is) ---
# ... (load_index_list_for_backtest and fetch_historical_data functions) ...
@st.cache_data(ttl=timedelta(days=1), show_spinner=False)
def load_index_list_for_backtest(list_name="nifty500"):
    if list_name == "nifty500": csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty500list.csv"
    elif list_name == "nifty200": csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty200list.csv"
    else: st.error(f"Unknown stock list: {list_name}"); return []
    try:
        df_nifty = pd.read_csv(csv_url); df_nifty.columns = df_nifty.columns.str.strip()
        df_nifty = df_nifty[~df_nifty['Symbol'].str.contains("DUMMY", na=False)]; df_nifty['Symbol'] = df_nifty['Symbol'].str.strip()
        df_nifty['Ticker'] = df_nifty['Symbol'] + ".NS"; return df_nifty['Ticker'].tolist()
    except Exception as e: st.error(f"Error loading {list_name.upper()} list: {e}"); return []

@st.cache_data(ttl=timedelta(hours=4), show_spinner=False)
def fetch_historical_data(tickers_tuple, start_date_str, end_date_str):
    tickers_list = list(tickers_tuple)
    if not tickers_list: return {}
    try:
        # yf.download end_date is exclusive for datetime objects, for strings it's usually inclusive.
        # To be safe and ensure we get data for end_date_str, we can fetch up to end_date_str + 1 day
        # and then slice later if needed, or rely on yf's string date inclusivity.
        # For simplicity, assuming yf.download handles string end_date inclusively for the daily interval.
        data = yf.download(tickers_list, start=start_date_str, end=end_date_str, 
                           interval='1d', group_by='ticker', auto_adjust=False, progress=False, timeout=180) # Increased timeout
        stock_data_processed = {}
        if data.empty: return {}

        # Determine actual start and end Timestamps for precise slicing
        actual_start_ts = pd.Timestamp(start_date_str)
        actual_end_ts = pd.Timestamp(end_date_str)

        if isinstance(data.columns, pd.MultiIndex): # Multiple tickers
            for ticker in tickers_list:
                if ticker in data and isinstance(data[ticker], pd.DataFrame) and not data[ticker].empty:
                    df_ticker = data[ticker].copy()
                    df_ticker.index = pd.to_datetime(df_ticker.index)
                    # Filter data strictly between actual start and end dates AFTER download
                    df_ticker = df_ticker[(df_ticker.index >= actual_start_ts) & (df_ticker.index <= actual_end_ts)]
                    if not df_ticker.empty:
                        df_ticker = df_ticker.ffill(limit=2).bfill(limit=2) # Fill small gaps
                        stock_data_processed[ticker] = df_ticker
        elif len(tickers_list) == 1 and isinstance(data, pd.DataFrame) and not data.empty: # Single ticker
             df_single = data.copy()
             df_single.index = pd.to_datetime(df_single.index)
             df_single = df_single[(df_single.index >= actual_start_ts) & (df_single.index <= actual_end_ts)]
             if not df_single.empty:
                df_single = df_single.ffill(limit=2).bfill(limit=2)
                stock_data_processed[tickers_list[0]] = df_single
        return stock_data_processed
    except Exception as e:
        st.warning(f"Could not download/process all historical data: {str(e)[:150]}")
        return {}


# --- Function to dynamically get available screener modules ---
def get_available_screeners():
    screener_modules = {}
    pages_path = Path("pages")
    if pages_path.is_dir():
        for f in pages_path.iterdir():
            if f.is_file() and f.suffix == '.py' and not f.name.startswith('.') and not f.name == "__init__.py":
                # Exclude self (the backtester page) if it's in pages and named something identifiable
                # For now, assume all .py files in pages (other than __init__) are screeners
                # if f.stem == "6_Strategy_Backtester": continue # Example to exclude self
                
                module_name_for_import = f"pages.{f.stem}" # e.g., pages.1_NIFTY_200_Screener
                
                # Create a user-friendly display name
                display_name_base = f.stem
                parts = display_name_base.split("_", 1)
                if len(parts) > 1 and parts[0].isdigit():
                    display_name_base = parts[1] # Strip leading "X_"
                display_name = display_name_base.replace("_", " ").title()
                
                screener_modules[display_name] = module_name_for_import
    return screener_modules

# --- Universal Backtesting Engine ---
@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def run_universal_backtest(
    all_hist_data_full_range, # Dict: {ticker: DataFrame over full backtest period + buffer}
    tickers_to_test_tuple, 
    bt_start_date_input, 
    bt_end_date_input,   
    initial_capital, 
    target_profit_pct, 
    stop_loss_pct,
    selected_screener_module_path, # e.g., "pages.1_NIFTY_200_Screener"
    screener_signal_function_name, # e.g., "get_entry_signals_for_date"
    max_concurrent_positions, 
    investment_per_trade_pct
):
    tickers_to_test = list(tickers_to_test_tuple)
    cash = initial_capital
    positions = {}
    trades_log = []
    portfolio_history_daily = []

    bt_start_date_ts = pd.Timestamp(bt_start_date_input)
    bt_end_date_ts = pd.Timestamp(bt_end_date_input)
    
    # Initial portfolio value point
    portfolio_history_daily.append({'Date': bt_start_date_ts - pd.Timedelta(days=1), 'Portfolio_Value': initial_capital})

    try:
        screener_module = importlib.import_module(selected_screener_module_path)
        signal_function = getattr(screener_module, screener_signal_function_name)
    except Exception as e:
        st.error(f"Error importing screener module '{selected_screener_module_path}' or finding function '{screener_signal_function_name}': {e}")
        return None # Indicate failure

    # Ensure all dfs have datetime index
    for ticker_key in all_hist_data_full_range:
        if not isinstance(all_hist_data_full_range[ticker_key].index, pd.DatetimeIndex):
            all_hist_data_full_range[ticker_key].index = pd.to_datetime(all_hist_data_full_range[ticker_key].index)

    backtest_date_range = pd.date_range(start=bt_start_date_ts, end=bt_end_date_ts, freq='B')
    if backtest_date_range.empty: 
        st.warning("Backtest date range is empty.")
        return None

    for current_date in stqdm(backtest_date_range, desc=f"Backtesting {selected_screener_module_path.split('.')[-1]}"):
        current_portfolio_mtm_value = 0.0
        tickers_to_sell_today = []

        # --- Exit Logic (same as your current working backtester) ---
        for ticker, pos_data in positions.items():
            if ticker not in all_hist_data_full_range or all_hist_data_full_range[ticker].empty: continue
            current_price = pos_data['entry_price'] 
            if current_date in all_hist_data_full_range[ticker].index and pd.notna(all_hist_data_full_range[ticker].loc[current_date, 'Adj Close']):
                current_price = all_hist_data_full_range[ticker].loc[current_date, 'Adj Close']
            else: 
                price_slice = all_hist_data_full_range[ticker].loc[all_hist_data_full_range[ticker].index <= current_date, 'Adj Close'].dropna()
                if not price_slice.empty: current_price = price_slice.iloc[-1]
            current_portfolio_mtm_value += current_price * pos_data['quantity']
            profit_target_price = pos_data['entry_price'] * (1 + target_profit_pct / 100.0)
            stop_loss_price = pos_data['entry_price'] * (1 - stop_loss_pct / 100.0)
            exit_reason = None
            if current_price >= profit_target_price: exit_reason = "Target Profit"
            elif current_price <= stop_loss_price: exit_reason = "Stop Loss"
            if exit_reason:
                sell_price = current_price 
                cash += sell_price * pos_data['quantity']
                trades_log.append({'Ticker': ticker, 'Entry Date': pos_data['entry_date'], 'Entry Price': pos_data['entry_price'],
                                   'Exit Date': current_date, 'Exit Price': sell_price, 'Quantity': pos_data['quantity'],
                                   'P/L': (sell_price - pos_data['entry_price']) * pos_data['quantity'], 'Reason': exit_reason})
                tickers_to_sell_today.append(ticker)
        for ticker in tickers_to_sell_today:
            if ticker in positions: del positions[ticker]
        # --- End Exit Logic ---

        # --- Entry Logic ---
        if len(positions) < max_concurrent_positions:
            signal_date_dt = current_date - pd.Timedelta(days=1) # Signals based on D-1 data
            if signal_date_dt >= bt_start_date_ts:
                
                # Prepare data for the signal function: dict of {ticker: df_slice_up_to_signal_date}
                data_for_signal_function = {}
                for t in tickers_to_test: 
                    if t in all_hist_data_full_range:
                        # Ensure the slice ends on or before signal_date_dt
                        df_slice = all_hist_data_full_range[t].loc[all_hist_data_full_range[t].index <= signal_date_dt]
                        if not df_slice.empty:
                            data_for_signal_function[t] = df_slice
                
                if data_for_signal_function:
                    try:
                        # Call the dynamically imported screener's signal function
                        # The screener function must be designed to handle this input format.
                        entry_signal_tickers = signal_function(
                            tuple(data_for_signal_function.keys()), # Pass tickers being checked
                            data_for_signal_function,            # Pass data slices
                            signal_date_dt                       # Pass the signal date
                            # If your signal functions need other specific params (e.g. fixed thresholds from their own file),
                            # those would need to be handled, possibly by passing a params dict.
                        )
                    except Exception as e_sig:
                        # st.sidebar.warning(f"Error in signal function {screener_signal_function_name} from {selected_screener_module_path} for {signal_date_dt.date()}: {e_sig}")
                        entry_signal_tickers = []

                    for ticker_to_buy in entry_signal_tickers:
                        if ticker_to_buy in positions or len(positions) >= max_concurrent_positions: continue
                        
                        # Check if data exists for trade execution on current_date
                        if current_date in all_hist_data_full_range[ticker_to_buy].index and \
                           pd.notna(all_hist_data_full_range[ticker_to_buy].loc[current_date, 'Adj Close']) and \
                           pd.notna(all_hist_data_full_range[ticker_to_buy].loc[current_date, 'Volume']):
                            
                            actual_entry_price = all_hist_data_full_range[ticker_to_buy].loc[current_date, 'Adj Close']
                            trade_day_volume = all_hist_data_full_range[ticker_to_buy].loc[current_date, 'Volume']

                            # Apply generic execution day filters (e.g. min price/volume if defined globally for backtester)
                            # For example, from your Advanced Pullback: Price > 100, Vol > 100k
                            # These could be passed as parameters to run_universal_backtest if they vary per strategy being tested
                            passes_execution_filters = True
                            if selected_screener_module_path.endswith("Advanced_Pullback_Screener"): # Example: specific filters for this strategy
                                if actual_entry_price <= 100 or trade_day_volume <= 100000:
                                    passes_execution_filters = False
                            
                            if passes_execution_filters:
                                # --- Position Sizing and Trade Execution (same as before) ---
                                current_value_of_open_positions = sum(
                                    (all_hist_data_full_range[t].loc[current_date, 'Adj Close'] if current_date in all_hist_data_full_range[t].index and pd.notna(all_hist_data_full_range[t].loc[current_date, 'Adj Close']) else pos['entry_price']) * pos['quantity']
                                    for t, pos in positions.items()
                                )
                                current_total_value_for_sizing = cash + current_value_of_open_positions
                                amount_to_invest = current_total_value_for_sizing * (investment_per_trade_pct / 100.0)
                                
                                if cash >= amount_to_invest and amount_to_invest > 0 and actual_entry_price > 0:
                                    quantity = int(amount_to_invest / actual_entry_price)
                                    if quantity > 0:
                                        cost_of_trade = quantity * actual_entry_price 
                                        if cash >= cost_of_trade: # Final cash check
                                            cash -= cost_of_trade
                                            positions[ticker_to_buy] = {'entry_price': actual_entry_price, 'quantity': quantity, 'entry_date': current_date}
        
        # --- EOD Portfolio Value Calculation (same as before) ---
        eod_mtm_value_of_positions = 0
        for ticker, pos_data in positions.items():
            price_to_use_eod = pos_data['entry_price']
            if ticker in all_hist_data_full_range and current_date in all_hist_data_full_range[ticker].index and pd.notna(all_hist_data_full_range[ticker].loc[current_date, 'Adj Close']):
                price_to_use_eod = all_hist_data_full_range[ticker].loc[current_date, 'Adj Close']
            else:
                 price_slice_eod = all_hist_data_full_range[ticker].loc[all_hist_data_full_range[ticker].index <= current_date, 'Adj Close'].dropna()
                 if not price_slice_eod.empty: price_to_use_eod = price_slice_eod.iloc[-1]
            eod_mtm_value_of_positions += price_to_use_eod * pos_data['quantity']
        current_portfolio_value = cash + eod_mtm_value_of_positions
        portfolio_history_daily.append({'Date': current_date, 'Portfolio_Value': current_portfolio_value})

    # --- Final Mark-to-Market & KPI Calculation (same as before) ---
    # ... (This part of the function remains identical to your working version) ...
    final_cash = cash; final_market_value_of_positions = 0
    bt_end_date_actual = backtest_date_range[-1] if not backtest_date_range.empty else bt_end_date_ts
    for ticker, pos_data in positions.items():
        final_price = pos_data['entry_price'] 
        if ticker in all_hist_data_full_range:
            price_slice_final = all_hist_data_full_range[ticker].loc[all_hist_data_full_range[ticker].index <= bt_end_date_actual, 'Adj Close'].dropna()
            if not price_slice_final.empty: final_price = price_slice_final.iloc[-1]
        final_market_value_of_positions += final_price * pos_data['quantity']
        if not any(t['Ticker'] == ticker and t['Reason'] == 'End of Backtest' for t in trades_log):
            trades_log.append({'Ticker': ticker, 'Entry Date': pos_data['entry_date'], 'Entry Price': pos_data['entry_price'],
                               'Exit Date': bt_end_date_actual, 'Exit Price': final_price, 'Quantity': pos_data['quantity'],
                               'P/L': (final_price - pos_data['entry_price']) * pos_data['quantity'], 'Reason': 'End of Backtest'})
    final_portfolio_val = final_cash + final_market_value_of_positions
    if portfolio_history_daily:
        if not portfolio_history_daily or portfolio_history_daily[-1]['Date'] != bt_end_date_actual : portfolio_history_daily.append({'Date': bt_end_date_actual, 'Portfolio_Value': final_portfolio_val})
        else: portfolio_history_daily[-1]['Portfolio_Value'] = final_portfolio_val
    elif initial_capital > 0: portfolio_history_daily.append({'Date': bt_start_date_ts, 'Portfolio_Value': initial_capital})
    df_trades_final = pd.DataFrame(trades_log); df_portfolio_history_final = pd.DataFrame(portfolio_history_daily)
    if not df_portfolio_history_final.empty: df_portfolio_history_final = df_portfolio_history_final.set_index('Date')
    total_return_pct,max_drawdown_pct,num_trades,win_rate_pct = 0,0,0,0
    if not df_portfolio_history_final.empty and initial_capital > 0:
        total_return_pct=(df_portfolio_history_final['Portfolio_Value'].iloc[-1]/initial_capital - 1)*100
        roll_max=df_portfolio_history_final['Portfolio_Value'].cummax()
        daily_drawdown=df_portfolio_history_final['Portfolio_Value']/roll_max - 1.0
        max_drawdown_pct=daily_drawdown.min()*100 if not daily_drawdown.empty and not daily_drawdown.isnull().all() else 0
    if not df_trades_final.empty:
        num_trades=len(df_trades_final)
        if num_trades > 0: win_rate_pct=(len(df_trades_final[df_trades_final['P/L'] > 0])/num_trades*100)
    return {"total_return_pct": total_return_pct, "max_drawdown_pct": max_drawdown_pct,
            "num_trades": num_trades, "win_rate_pct": win_rate_pct,
            "df_portfolio_history": df_portfolio_history_final, "df_trades": df_trades_final}


# --- UI for Backtesting Parameters ---
st.sidebar.header("Backtest Configuration")
index_choice_bt = st.sidebar.selectbox("Index Universe", ["NIFTY 500", "NIFTY 200"], key="bt_idx_choice_sidebar_univ")
bt_list_to_load = "nifty500" if index_choice_bt == "NIFTY 500" else "nifty200"

if 'bt_start_date_univ' not in st.session_state: st.session_state.bt_start_date_univ = datetime.today() - timedelta(days=365*2)
if 'bt_end_date_univ' not in st.session_state: st.session_state.bt_end_date_univ = datetime.today() - timedelta(days=1)
bt_start_date = st.sidebar.date_input("Start Date", value=st.session_state.bt_start_date_univ, key="bt_start_dt_picker_univ")
bt_end_date = st.sidebar.date_input("End Date", value=st.session_state.bt_end_date_univ, max_value=datetime.today() - timedelta(days=1), key="bt_end_dt_picker_univ")
st.session_state.bt_start_date_univ = bt_start_date; st.session_state.bt_end_date_univ = bt_end_date

initial_capital_bt = st.sidebar.number_input("Initial Capital", min_value=10000, value=100000, step=10000, key="bt_cap_input_univ")

available_screener_modules = get_available_screeners()
selected_screener_display_name = st.sidebar.selectbox(
    "Screener Strategy to Backtest",
    options=list(available_screener_modules.keys()), 
    key="bt_screener_select_univ"
)
selected_module_path_for_import = available_screener_modules.get(selected_screener_display_name)


st.sidebar.markdown("--- Exit Strategy (Common for all strategies) ---")
target_profit_pct_bt = st.sidebar.slider("Target Profit (%)", 1, 100, 20, 1, key="bt_tp_sld_univ") 
stop_loss_pct_bt = st.sidebar.slider("Stop Loss (%)", 1, 50, 10, 1, key="bt_sl_sld_univ") 
st.sidebar.markdown("--- Portfolio Allocation (Common) ---")
max_concurrent_positions_bt = st.sidebar.slider("Max Open Positions", 1, 20, 5, 1, key="bt_max_pos_sld_univ")
investment_per_trade_pct_bt = st.sidebar.slider("Investment/Trade (% of Portfolio)", 1, 50, 10, 1, key="bt_inv_pct_sld_univ")

run_backtest_button = st.sidebar.button("🚀 Run Backtest", type="primary", key="bt_run_btn_main_univ")

if run_backtest_button:
    if not bt_start_date or not bt_end_date or bt_start_date >= bt_end_date: st.error("Invalid date range.")
    elif initial_capital_bt <=0: st.error("Initial capital must be > 0.")
    elif not selected_module_path_for_import: st.error("Please select a screener strategy.")
    else:
        with st.spinner(f"Loading {index_choice_bt} list..."): 
            tickers_for_backtest = load_index_list_for_backtest(list_name=bt_list_to_load)
        
        if not tickers_for_backtest: st.error(f"Failed to load tickers for {index_choice_bt}.")
        else:
            # Define historical data range: from buffer before backtest start to backtest end
            # Buffer for indicators like 252 day rolling max
            data_fetch_start_str = (bt_start_date - timedelta(days=400)).strftime('%Y-%m-%d') 
            data_fetch_end_str = bt_end_date.strftime('%Y-%m-%d') # Actual end of user-selected range

            with st.spinner(f"Fetching historical data for {len(tickers_for_backtest)} stocks ({data_fetch_start_str} to {data_fetch_end_str})..."):
                 all_hist_data_bt = fetch_historical_data(
                    tuple(tickers_for_backtest), 
                    data_fetch_start_str, 
                    data_fetch_end_str 
                )

            if not all_hist_data_bt: st.error("Failed to fetch sufficient historical data for backtesting.")
            else:
                # Assume a standard function name for signal generation in each screener module
                SIGNAL_FUNCTION_NAME = "get_entry_signals_for_date" # *** IMPORTANT CONVENTION ***

                with st.spinner(f"Executing backtest using '{selected_screener_display_name}' strategy on {index_choice_bt}..."):
                    backtest_results_run = run_universal_backtest(
                        all_hist_data_bt, 
                        tuple(tickers_for_backtest), 
                        bt_start_date, # This is datetime.date object
                        bt_end_date,   # This is datetime.date object
                        initial_capital_bt, 
                        target_profit_pct_bt, 
                        stop_loss_pct_bt,
                        selected_module_path_for_import, 
                        SIGNAL_FUNCTION_NAME,
                        max_concurrent_positions_bt, 
                        investment_per_trade_pct_bt
                    )
                st.session_state.backtest_results_data_univ = backtest_results_run
                st.session_state.backtest_params_used_univ = {
                    "Screener Strategy": selected_screener_display_name,
                    "Index Universe": index_choice_bt, "Start Date": bt_start_date.strftime("%Y-%m-%d"),
                    "End Date": bt_end_date.strftime("%Y-%m-%d"), "Initial Capital": f"₹{initial_capital_bt:,.0f}",
                    "Target Profit (%)": target_profit_pct_bt, "Stop Loss (%)": stop_loss_pct_bt, 
                    "Max Open Positions": max_concurrent_positions_bt, "Investment/Trade (%)": investment_per_trade_pct_bt
                }
                st.rerun()

if 'backtest_results_data_univ' in st.session_state:
    results = st.session_state.backtest_results_data_univ
    if results:
        st.markdown("---"); st.header("📜 Backtest Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{results.get('total_return_pct', 0):.2f}%")
        col2.metric("Max Drawdown", f"{results.get('max_drawdown_pct', 0):.2f}%")
        col3.metric("Number of Trades", results.get('num_trades', 0))
        col4.metric("Win Rate", f"{results.get('win_rate_pct', 0):.2f}%")
        df_portfolio_history = results.get('df_portfolio_history')
        if df_portfolio_history is not None and not df_portfolio_history.empty:
            st.subheader("Equity Curve"); df_portfolio_history['Portfolio_Value'] = pd.to_numeric(df_portfolio_history['Portfolio_Value'], errors='coerce')
            st.line_chart(df_portfolio_history['Portfolio_Value'])
        df_trades = results.get('df_trades')
        if df_trades is not None and not df_trades.empty:
            st.subheader("Trades Log"); st.dataframe(df_trades.style.format({"Entry Price":"{:.2f}","Exit Price":"{:.2f}","P/L":"{:.2f}","Entry Date":"{:%Y-%m-%d}","Exit Date":"{:%Y-%m-%d}"},na_rep="-"))
        elif results.get('num_trades', 0) == 0 : st.info("No trades were executed.")
        
        if 'backtest_params_used_univ' in st.session_state:
            st.markdown("---"); st.subheader("📋 Backtest Parameters Used")
            params_to_display = st.session_state.backtest_params_used_univ
            param_col1, param_col2 = st.columns(2)
            param_items = list(params_to_display.items()); mid_point = len(param_items)//2 + (len(param_items)%2)
            with param_col1:
                for key, value in param_items[:mid_point]: st.markdown(f"- **{key}:** {value}")
            with param_col2:
                for key, value in param_items[mid_point:]: st.markdown(f"- **{key}:** {value}")
    else: st.error("Backtest failed or returned no results.")
else: st.info("Configure parameters in sidebar & click 'Run Backtest'.")

st.markdown("---")
st.markdown("Disclaimer: Historical backtesting does not guarantee future results. Informational purposes only. DYOR.")
