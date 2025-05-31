# pages/6_Strategy_Backtester.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from stqdm import stqdm 
import io 

st.set_page_config(
    page_title="Strategy Backtester - StockScopePro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛠️ Strategy Backtester")
st.markdown("Test the historical performance of predefined trading setups based on momentum screener logic.")

with st.expander("📜 **Backtested Entry Signal Conditions**", expanded=True):
    st.markdown("""
    This backtester simulates entries based on the following conditions, evaluated on data up to the **day before** a potential trade:

    **For "Breakout 52w" Setup:**
    1.  `Adjusted Close (Signal Day) >= 0.99 * 52-Week High (calculated up to Signal Day)`

    **For "Breakout" Setup:**
    1.  `Adjusted Close (Signal Day) >= 0.99 * 20-Day High (calculated up to Signal Day)`
    2.  *(And not already a "Breakout 52w")*

    **For "Retest" Setup:**
    1.  `Adjusted Close (Signal Day) >= 50-Day EMA (calculated up to Signal Day)`
    2.  `Adjusted Close (Signal Day) <= 1.02 * 50-Day EMA (calculated up to Signal Day)`
    3.  *(And not already a "Breakout 52w" or "Breakout")*
    
    **Common to all Setups for a Valid Signal in Backtest:**
    - Sufficient historical data must be available to calculate the necessary indicators (e.g., at least 252 days for 52W High).
    - The stock must have valid price data on the day the trade is to be executed.

    **Exit Conditions (User Defined in Sidebar):**
    - Target Profit (%)
    - Stop Loss (%)

    **Portfolio Allocation (User Defined in Sidebar):**
    - Max Concurrent Open Positions
    - Investment per Trade (% of current portfolio value)
    """)

# --- Helper Functions ---
@st.cache_data(ttl=timedelta(days=1), show_spinner=False)
def load_index_list_for_backtest(list_name="nifty500"):
    if list_name == "nifty500":
        csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty500list.csv"
    elif list_name == "nifty200":
        csv_url = "https://raw.githubusercontent.com/stockscope/Momentum_Sector_Breakout_Screener/main/ind_nifty200list.csv"
    else:
        st.error(f"Unknown stock list: {list_name}")
        return []
    try:
        df_nifty = pd.read_csv(csv_url)
        df_nifty.columns = df_nifty.columns.str.strip()
        df_nifty = df_nifty[~df_nifty['Symbol'].str.contains("DUMMY", na=False)]
        df_nifty['Symbol'] = df_nifty['Symbol'].str.strip()
        df_nifty['Ticker'] = df_nifty['Symbol'] + ".NS"
        return df_nifty['Ticker'].tolist()
    except Exception as e:
        st.error(f"Error loading {list_name.upper()} list: {e}")
        return []

@st.cache_data(ttl=timedelta(hours=4), show_spinner=False)
def fetch_historical_data(tickers_tuple, start_date_str, end_date_str):
    tickers_list = list(tickers_tuple)
    if not tickers_list: return {}
    try:
        # Ensure end_date for yf.download is inclusive by adding a day if needed by its behavior with string dates
        # yf.download typically treats string end dates as inclusive.
        # For datetime objects, it's often exclusive. Let's stick to string for yf.download.
        # For safety, fetching one extra day and then slicing can be more robust if behavior is uncertain.
        end_date_dt_plus_one = (datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

        data = yf.download(tickers_list, start=start_date_str, end=end_date_dt_plus_one,
                           interval='1d', group_by='ticker', auto_adjust=False, progress=False, timeout=120)
        stock_data_processed = {}
        if data.empty: return {}

        actual_start_ts = pd.Timestamp(start_date_str)
        actual_end_ts = pd.Timestamp(end_date_str) # The user's desired actual end date

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

def calculate_indicators_for_date_slice(df_slice):
    # df_slice is data up to and including the signal date
    if len(df_slice) < 20: return None # Min for 20D_High & some EMAs
    
    indicators = {}
    # Ensure data exists for the last row
    if df_slice.empty: return None
    last_valid_index = df_slice.index[-1] 

    indicators['Adj Close'] = df_slice.loc[last_valid_index, 'Adj Close']
    indicators['High'] = df_slice.loc[last_valid_index, 'High']
    # indicators['Volume'] = df_slice.loc[last_valid_index, 'Volume'] # Not used by these specific simple setups

    # Min periods are crucial to avoid NaNs if df_slice is short but meets overall len requirement
    if len(df_slice) >= 50:
        indicators['50EMA'] = df_slice['Adj Close'].ewm(span=50, adjust=False, min_periods=max(1,50-5)).mean().loc[last_valid_index]
    else: indicators['50EMA'] = np.nan
    
    if len(df_slice) >= 20:
        indicators['20D_High'] = df_slice['High'].rolling(window=20, min_periods=max(1,20-5)).max().loc[last_valid_index]
    else: indicators['20D_High'] = np.nan
    
    if len(df_slice) >= 252:
        indicators['52W_High'] = df_slice['High'].rolling(window=252, min_periods=max(1,252-50)).max().loc[last_valid_index] # Allow more leeway for 52W_High
    else: indicators['52W_High'] = np.nan
    
    return indicators

# --- Backtesting Engine ---
@st.cache_data(ttl=timedelta(hours=1), show_spinner=False) 
def run_backtest(
    all_hist_data, tickers_to_test_tuple, bt_start_date_input, bt_end_date_input,   
    initial_capital, target_profit_pct, stop_loss_pct, setup_to_backtest,
    max_concurrent_positions, investment_per_trade_pct
):
    tickers_to_test = list(tickers_to_test_tuple)
    portfolio_value = initial_capital; cash = initial_capital
    positions = {}; trades_log = []
    
    # Ensure first entry in portfolio_history_daily matches the actual backtest start for cleaner equity curve
    portfolio_history_daily = [] 

    bt_start_date_ts = pd.Timestamp(bt_start_date_input)
    bt_end_date_ts = pd.Timestamp(bt_end_date_input)

    # Ensure all dfs have datetime index
    for ticker_key in all_hist_data:
        if not isinstance(all_hist_data[ticker_key].index, pd.DatetimeIndex):
            all_hist_data[ticker_key].index = pd.to_datetime(all_hist_data[ticker_key].index)

    backtest_date_range = pd.date_range(start=bt_start_date_ts, end=bt_end_date_ts, freq='B')
    if backtest_date_range.empty: 
        st.warning("Backtest date range is empty or invalid.")
        return None
    
    # Add initial capital point
    portfolio_history_daily.append({'Date': backtest_date_range[0] - pd.Timedelta(days=1), 'Portfolio_Value': initial_capital})

    for current_date in stqdm(backtest_date_range, desc="Backtesting Days"):
        current_portfolio_mtm_value = 0.0 # Market value of open positions for today
        
        # --- Exit Logic ---
        tickers_to_sell_today = []
        for ticker, pos_data in positions.items():
            if ticker not in all_hist_data or all_hist_data[ticker].empty: continue
            current_price_for_exit = pos_data['entry_price'] 
            if current_date in all_hist_data[ticker].index and pd.notna(all_hist_data[ticker].loc[current_date, 'Adj Close']):
                current_price_for_exit = all_hist_data[ticker].loc[current_date, 'Adj Close']
            else: 
                price_slice_exit = all_hist_data[ticker].loc[all_hist_data[ticker].index < current_date, 'Adj Close'].dropna() # strictly before
                if not price_slice_exit.empty: current_price_for_exit = price_slice_exit.iloc[-1]
            
            current_portfolio_mtm_value += current_price_for_exit * pos_data['quantity']

            profit_target_price = pos_data['entry_price'] * (1 + target_profit_pct / 100.0)
            stop_loss_price = pos_data['entry_price'] * (1 - stop_loss_pct / 100.0)
            exit_reason = None
            if current_price_for_exit >= profit_target_price: exit_reason = "Target Profit"
            elif current_price_for_exit <= stop_loss_price: exit_reason = "Stop Loss"

            if exit_reason:
                sell_price = current_price_for_exit 
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
            potential_entries = []
            for ticker in tickers_to_test:
                if ticker in positions or ticker not in all_hist_data or all_hist_data[ticker].empty: continue
                
                signal_date = current_date - pd.Timedelta(days=1) 
                if signal_date < bt_start_date_ts : continue 

                df_slice_for_signal = all_hist_data[ticker].loc[all_hist_data[ticker].index <= signal_date]
                
                min_len_required = 50 # Default for basic EMAs/SMAs
                if setup_to_backtest == "Breakout 52w": min_len_required = 252 
                
                if df_slice_for_signal.empty or len(df_slice_for_signal) < min_len_required: continue

                inds = calculate_indicators_for_date_slice(df_slice_for_signal)
                if inds is None or pd.isna(inds.get('Adj Close')): continue # Must have a close price for signal day
                
                entry_signal = False
                adj_close_sig = inds['Adj Close']
                
                if setup_to_backtest == "Breakout 52w":
                    if pd.notna(inds.get('52W_High')) and adj_close_sig >= 0.99 * inds['52W_High']:
                        entry_signal = True
                elif setup_to_backtest == "Breakout":
                    if pd.notna(inds.get('20D_High')) and adj_close_sig >= 0.99 * inds['20D_High']:
                        entry_signal = True
                elif setup_to_backtest == "Retest":
                    if pd.notna(inds.get('50EMA')) and \
                       adj_close_sig >= inds['50EMA'] and adj_close_sig <= 1.02 * inds['50EMA']:
                        entry_signal = True
                
                if entry_signal:
                    if current_date in all_hist_data[ticker].index and pd.notna(all_hist_data[ticker].loc[current_date, 'Adj Close']):
                        potential_entries.append({'ticker': ticker, 'entry_price': all_hist_data[ticker].loc[current_date, 'Adj Close']})
            
            for entry_candidate in potential_entries:
                if len(positions) >= max_concurrent_positions: break
                ticker_to_buy = entry_candidate['ticker']; actual_entry_price = entry_candidate['entry_price']
                
                current_value_of_open_positions = 0
                for t, pos in positions.items(): # Value of *already open* positions before this new trade
                    price_val = pos['entry_price']
                    if current_date in all_hist_data[t].index and pd.notna(all_hist_data[t].loc[current_date, 'Adj Close']):
                        price_val = all_hist_data[t].loc[current_date, 'Adj Close']
                    else:
                        price_slice_sizing = all_hist_data[t].loc[all_hist_data[t].index <= current_date, 'Adj Close'].dropna()
                        if not price_slice_sizing.empty: price_val = price_slice_sizing.iloc[-1]
                    current_value_of_open_positions += price_val * pos['quantity']

                current_total_value_for_sizing = cash + current_value_of_open_positions
                amount_to_invest = current_total_value_for_sizing * (investment_per_trade_pct / 100.0)
                
                if cash >= amount_to_invest and amount_to_invest > 0 and actual_entry_price > 0:
                    quantity = int(amount_to_invest / actual_entry_price)
                    if quantity > 0:
                        cost_of_trade = quantity * actual_entry_price 
                        if cash >= cost_of_trade: # Final cash check
                            cash -= cost_of_trade
                            positions[ticker_to_buy] = {'entry_price': actual_entry_price, 'quantity': quantity, 'entry_date': current_date}
        
        # --- EOD Portfolio Value Calculation ---
        eod_mtm_value_of_positions = 0.0
        for ticker, pos_data in positions.items():
            price_to_use_eod = pos_data['entry_price']
            if ticker in all_hist_data and current_date in all_hist_data[ticker].index and pd.notna(all_hist_data[ticker].loc[current_date, 'Adj Close']):
                price_to_use_eod = all_hist_data[ticker].loc[current_date, 'Adj Close']
            else:
                 price_slice_eod = all_hist_data[ticker].loc[all_hist_data[ticker].index <= current_date, 'Adj Close'].dropna()
                 if not price_slice_eod.empty: price_to_use_eod = price_slice_eod.iloc[-1]
            eod_mtm_value_of_positions += price_to_use_eod * pos_data['quantity']
        current_portfolio_value = cash + eod_mtm_value_of_positions
        portfolio_history_daily.append({'Date': current_date, 'Portfolio_Value': current_portfolio_value})
    
    # --- Final Mark-to-Market & KPI Calculation ---
    final_cash = cash; final_market_value_of_positions = 0
    bt_end_date_actual = backtest_date_range[-1] if not backtest_date_range.empty else bt_end_date_ts
    for ticker, pos_data in positions.items():
        final_price = pos_data['entry_price'] 
        if ticker in all_hist_data:
            price_slice_final = all_hist_data[ticker].loc[all_hist_data[ticker].index <= bt_end_date_actual, 'Adj Close'].dropna()
            if not price_slice_final.empty: final_price = price_slice_final.iloc[-1]
        final_market_value_of_positions += final_price * pos_data['quantity']
        if not any(t['Ticker'] == ticker and t['Reason'] == 'End of Backtest' for t in trades_log):
            trades_log.append({'Ticker': ticker, 'Entry Date': pos_data['entry_date'], 'Entry Price': pos_data['entry_price'],
                               'Exit Date': bt_end_date_actual, 'Exit Price': final_price, 'Quantity': pos_data['quantity'],
                               'P/L': (final_price - pos_data['entry_price']) * pos_data['quantity'], 'Reason': 'End of Backtest'})
    final_portfolio_val = final_cash + final_market_value_of_positions
    if portfolio_history_daily:
        if not portfolio_history_daily or portfolio_history_daily[-1]['Date'] != bt_end_date_actual : 
            portfolio_history_daily.append({'Date': bt_end_date_actual, 'Portfolio_Value': final_portfolio_val})
        elif not backtest_date_range.empty : # If last entry IS for bt_end_date_actual
            portfolio_history_daily[-1]['Portfolio_Value'] = final_portfolio_val
    elif initial_capital > 0 and not backtest_date_range.empty : 
         portfolio_history_daily.append({'Date': backtest_date_range[0], 'Portfolio_Value': initial_capital})
         if bt_end_date_actual != backtest_date_range[0]: 
            portfolio_history_daily.append({'Date': bt_end_date_actual, 'Portfolio_Value': final_portfolio_val})
    
    df_trades_final = pd.DataFrame(trades_log)
    df_portfolio_history_final = pd.DataFrame(portfolio_history_daily)
    if not df_portfolio_history_final.empty: 
        df_portfolio_history_final = df_portfolio_history_final.set_index('Date')
    
    total_return_pct,max_drawdown_pct,num_trades,win_rate_pct = 0.0,0.0,0,0.0
    if not df_portfolio_history_final.empty and initial_capital > 0:
        df_portfolio_history_final['Portfolio_Value'] = pd.to_numeric(df_portfolio_history_final['Portfolio_Value'], errors='coerce').fillna(initial_capital) # Fillna before calc
        if not df_portfolio_history_final['Portfolio_Value'].empty:
            total_return_pct=(df_portfolio_history_final['Portfolio_Value'].iloc[-1]/initial_capital - 1)*100
            roll_max=df_portfolio_history_final['Portfolio_Value'].cummax()
            daily_drawdown=df_portfolio_history_final['Portfolio_Value']/roll_max - 1.0
            max_drawdown_pct=daily_drawdown.min()*100 if not daily_drawdown.empty and not daily_drawdown.isnull().all() else 0.0
    if not df_trades_final.empty:
        num_trades=len(df_trades_final)
        if num_trades > 0: win_rate_pct=(len(df_trades_final[df_trades_final['P/L'] > 0])/num_trades*100)
    
    return {"total_return_pct": total_return_pct, "max_drawdown_pct": max_drawdown_pct,
            "num_trades": num_trades, "win_rate_pct": win_rate_pct,
            "df_portfolio_history": df_portfolio_history_final, "df_trades": df_trades_final}

# --- UI for Backtesting Parameters ---
st.sidebar.header("Backtest Configuration")
index_choice_bt = st.sidebar.selectbox("Index Universe", ["NIFTY 500", "NIFTY 200"], key="bt_idx_choice_sidebar")
bt_list_to_load = "nifty500" if index_choice_bt == "NIFTY 500" else "nifty200"

if 'bt_start_date' not in st.session_state: st.session_state.bt_start_date = datetime.today() - timedelta(days=365*2)
if 'bt_end_date' not in st.session_state: st.session_state.bt_end_date = datetime.today() - timedelta(days=1)

bt_start_date = st.sidebar.date_input("Start Date", value=st.session_state.bt_start_date, key="bt_start_dt_picker")
bt_end_date = st.sidebar.date_input("End Date", value=st.session_state.bt_end_date, max_value=datetime.today() - timedelta(days=1), key="bt_end_dt_picker")
st.session_state.bt_start_date = bt_start_date; st.session_state.bt_end_date = bt_end_date

initial_capital_bt = st.sidebar.number_input("Initial Capital", min_value=10000, value=100000, step=10000, key="bt_cap_input")
# These are the setup types hardcoded in this backtester's run_backtest function
setup_options_bt = ["Breakout 52w", "Breakout", "Retest"] 
setup_to_backtest_bt = st.sidebar.selectbox("Setup Type to Backtest", setup_options_bt, key="bt_setup_sel") 

st.sidebar.markdown("--- Exit Strategy ---")
target_profit_pct_bt = st.sidebar.slider("Target Profit (%)", 1, 100, 20, 1, key="bt_tp_sld") 
stop_loss_pct_bt = st.sidebar.slider("Stop Loss (%)", 1, 50, 10, 1, key="bt_sl_sld") 
st.sidebar.markdown("--- Portfolio Allocation ---")
max_concurrent_positions_bt = st.sidebar.slider("Max Open Positions", 1, 20, 5, 1, key="bt_max_pos_sld")
investment_per_trade_pct_bt = st.sidebar.slider("Investment/Trade (% of Portfolio)", 1, 50, 10, 1, key="bt_inv_pct_sld") 

run_backtest_button = st.sidebar.button("🚀 Run Backtest", type="primary", key="bt_run_btn_main") 

if run_backtest_button:
    if not bt_start_date or not bt_end_date or bt_start_date >= bt_end_date: st.error("Invalid date range.")
    elif initial_capital_bt <=0: st.error("Initial capital must be > 0.")
    else:
        with st.spinner(f"Loading {index_choice_bt} list..."): 
            tickers_for_backtest = load_index_list_for_backtest(list_name=bt_list_to_load)
        
        if not tickers_for_backtest: st.error(f"Failed to load tickers for {index_choice_bt}.")
        else:
            # Fetch data from (start_date - buffer) to (end_date)
            data_fetch_start_str = (bt_start_date - timedelta(days=400)).strftime('%Y-%m-%d') # Buffer for indicators
            data_fetch_end_str = bt_end_date.strftime('%Y-%m-%d') # Actual end of user-selected range

            with st.spinner(f"Fetching historical data for {len(tickers_for_backtest)} stocks ({data_fetch_start_str} to {data_fetch_end_str})..."):
                 all_hist_data_bt = fetch_historical_data(
                    tuple(tickers_for_backtest), 
                    data_fetch_start_str, 
                    data_fetch_end_str # Pass the user-defined end date for fetching
                )

            if not all_hist_data_bt: st.error("Failed to fetch sufficient historical data for backtesting.")
            else:
                with st.spinner(f"Executing backtest for '{setup_to_backtest_bt}' on {index_choice_bt}... This may take some time."):
                    # Pass datetime.date objects directly to run_backtest
                    backtest_results_run = run_backtest(
                        all_hist_data_bt, 
                        tuple(tickers_for_backtest), 
                        bt_start_date, # datetime.date object
                        bt_end_date,   # datetime.date object
                        initial_capital_bt, 
                        target_profit_pct_bt, 
                        stop_loss_pct_bt,
                        setup_to_backtest_bt, 
                        max_concurrent_positions_bt, 
                        investment_per_trade_pct_bt
                    )
                st.session_state.backtest_results_data = backtest_results_run
                st.session_state.backtest_params_used = {
                    "Index Universe": index_choice_bt, 
                    "Start Date": bt_start_date.strftime("%Y-%m-%d"),
                    "End Date": bt_end_date.strftime("%Y-%m-%d"), 
                    "Initial Capital": f"₹{initial_capital_bt:,.0f}",
                    "Setup Type": setup_to_backtest_bt, 
                    "Target Profit (%)": target_profit_pct_bt, 
                    "Stop Loss (%)": stop_loss_pct_bt, 
                    "Max Open Positions": max_concurrent_positions_bt, 
                    "Investment/Trade (%)": investment_per_trade_pct_bt
                }
                st.rerun()

if 'backtest_results_data' in st.session_state:
    results = st.session_state.backtest_results_data
    if results is None:
         st.error("Backtest execution failed or returned no results (results object is None). Check for errors during date range validation or data fetching.")
    elif not isinstance(results, dict) or 'df_portfolio_history' not in results :
        st.error("Backtest returned unexpected data structure or was incomplete.")
        st.write("Debug: Received results object:", results) 
    else: # Results seem valid
        st.markdown("---"); st.header("📜 Backtest Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{results.get('total_return_pct', 0.0):.2f}%")
        col2.metric("Max Drawdown", f"{results.get('max_drawdown_pct', 0.0):.2f}%")
        col3.metric("Number of Trades", results.get('num_trades', 0))
        col4.metric("Win Rate", f"{results.get('win_rate_pct', 0.0):.2f}%")

        df_portfolio_history = results.get('df_portfolio_history')
        if df_portfolio_history is not None and not df_portfolio_history.empty:
            st.subheader("Equity Curve")
            df_portfolio_history['Portfolio_Value'] = pd.to_numeric(df_portfolio_history['Portfolio_Value'], errors='coerce')
            if not df_portfolio_history['Portfolio_Value'].dropna().empty:
                 st.line_chart(df_portfolio_history['Portfolio_Value'])
            else: st.info("Equity curve data is empty or invalid after processing.")
        
        df_trades = results.get('df_trades')
        if df_trades is not None and not df_trades.empty:
            st.subheader("Trades Log")
            st.dataframe(df_trades.style.format({
                "Entry Price":"{:.2f}", "Exit Price":"{:.2f}", "P/L":"{:.2f}",
                "Entry Date":"{:%Y-%m-%d}", "Exit Date":"{:%Y-%m-%d}" 
            },na_rep="-"))
        elif results.get('num_trades', 0) == 0 : 
             st.info("No trades were executed during the backtest period with the given criteria.")
        
        if 'backtest_params_used' in st.session_state: # Display parameters
            st.markdown("---"); st.subheader("📋 Backtest Parameters Used")
            params_to_display = st.session_state.backtest_params_used
            param_col1, param_col2 = st.columns(2)
            param_items = list(params_to_display.items())
            mid_point = len(param_items)//2 + (len(param_items)%2)
            with param_col1:
                for key, value in param_items[:mid_point]: st.markdown(f"- **{key}:** {value}")
            with param_col2:
                for key, value in param_items[mid_point:]: st.markdown(f"- **{key}:** {value}")
else: 
    st.info("Configure backtest parameters in the sidebar and click 'Run Backtest'.")

st.markdown("---")
st.markdown("Disclaimer: Historical backtesting does not guarantee future results. Informational purposes only. DYOR.")
