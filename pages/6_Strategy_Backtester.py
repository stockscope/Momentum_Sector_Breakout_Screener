# pages/6_Strategy_Backtester.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from stqdm import stqdm # For progress bar in Streamlit

st.set_page_config(
    page_title="Strategy Backtester - StockScopePro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛠️ Strategy Backtester")
st.markdown("Test the historical performance of predefined trading setups.")

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

@st.cache_data(ttl=timedelta(hours=4), show_spinner=False) # Cache historical data longer
def fetch_historical_data(tickers_tuple, start_date_str, end_date_str):
    tickers_list = list(tickers_tuple)
    if not tickers_list: return {}
    try:
        # Add one day to end_date_str for yf.download because it's exclusive for date strings too sometimes
        end_date_dt_plus_one = (datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        
        data = yf.download(tickers_list, start=start_date_str, end=end_date_dt_plus_one,
                           interval='1d', group_by='ticker', auto_adjust=False, progress=False, timeout=120)
        stock_data_processed = {}
        if data.empty: return {}

        if isinstance(data.columns, pd.MultiIndex):
            for ticker in tickers_list:
                if ticker in data and isinstance(data[ticker], pd.DataFrame) and not data[ticker].empty:
                    df_ticker = data[ticker].copy()
                    df_ticker.index = pd.to_datetime(df_ticker.index)
                    # Forward fill missing values for up to 2 days, then backfill - helps with holidays
                    df_ticker = df_ticker.ffill(limit=2).bfill(limit=2)
                    stock_data_processed[ticker] = df_ticker
        elif len(tickers_list) == 1 and isinstance(data, pd.DataFrame) and not data.empty:
             df_single = data.copy()
             df_single.index = pd.to_datetime(df_single.index)
             df_single = df_single.ffill(limit=2).bfill(limit=2)
             stock_data_processed[tickers_list[0]] = df_single
        return stock_data_processed
    except Exception as e:
        st.warning(f"Could not download all historical data: {str(e)[:100]}")
        return {}

def calculate_indicators_for_date_slice(df_slice):
    """Calculates indicators needed for setups on a given slice of data."""
    if len(df_slice) < 252: # Minimum for 52W High, adjust as needed
        return None # Not enough data

    indicators = {}
    indicators['Adj Close'] = df_slice['Adj Close'].iloc[-1]
    indicators['High'] = df_slice['High'].iloc[-1] # Current day's high
    indicators['Volume'] = df_slice['Volume'].iloc[-1]

    # Make sure enough data for rolling/ewm before calculating
    if len(df_slice) >= 50:
        indicators['50EMA'] = df_slice['Adj Close'].ewm(span=50, adjust=False, min_periods=45).mean().iloc[-1]
    else: indicators['50EMA'] = np.nan

    if len(df_slice) >= 20:
        indicators['20D_High'] = df_slice['High'].rolling(window=20, min_periods=15).max().iloc[-1]
    else: indicators['20D_High'] = np.nan
    
    indicators['52W_High'] = df_slice['High'].rolling(window=252, min_periods=200).max().iloc[-1]

    return indicators

# --- Backtesting Engine ---
def run_backtest(
    all_hist_data, # Dict of DataFrames {ticker: df_hist}
    tickers_to_test,
    bt_start_date_dt, 
    bt_end_date_dt, 
    initial_capital,
    target_profit_pct,
    stop_loss_pct,
    setup_to_backtest,
    max_concurrent_positions=5,
    investment_per_trade_pct = 10 # % of current portfolio_value for each new trade
):
    portfolio_value = initial_capital
    cash = initial_capital
    positions = {} 
    trades_log = []
    portfolio_history_daily = []

    # Ensure dataframes in all_hist_data are indexed by DatetimeIndex
    for ticker_df in all_hist_data.values():
        if not isinstance(ticker_df.index, pd.DatetimeIndex):
            ticker_df.index = pd.to_datetime(ticker_df.index)

    backtest_date_range = pd.date_range(start=bt_start_date_dt, end=bt_end_date_dt, freq='B')

    for current_date in stqdm(backtest_date_range, desc="Backtesting Days"):
        current_portfolio_market_value = 0
        
        # 1. Mark to Market and Check Exits
        tickers_to_sell_today = []
        for ticker, pos_data in positions.items():
            if ticker not in all_hist_data or all_hist_data[ticker].empty: continue
            
            # Get price for current_date. If not available, use last known price (or ffill if data is clean)
            if current_date in all_hist_data[ticker].index:
                current_price = all_hist_data[ticker].loc[current_date, 'Adj Close']
                if pd.isna(current_price): # If current date has NaN price, try to get most recent valid price
                    # Take data up to and including current_date, get last valid Adj Close
                    recent_slice = all_hist_data[ticker].loc[all_hist_data[ticker].index <= current_date]
                    if not recent_slice.empty and pd.notna(recent_slice['Adj Close'].iloc[-1]):
                        current_price = recent_slice['Adj Close'].iloc[-1]
                    else: # Still no valid price, might hold or use previous day's portfolio value
                        current_price = pos_data['entry_price'] # Fallback to entry price if no current price
            else: # Date not in index (e.g. holiday, stock not listed yet)
                # Use the last available price before or on current_date
                recent_slice = all_hist_data[ticker].loc[all_hist_data[ticker].index <= current_date]
                if not recent_slice.empty and pd.notna(recent_slice['Adj Close'].iloc[-1]):
                    current_price = recent_slice['Adj Close'].iloc[-1]
                else: # If no data at all up to this point for the ticker (should not happen if bought)
                    current_price = pos_data['entry_price'] # Fallback

            if pd.isna(current_price): # Final fallback
                current_price = pos_data['entry_price']

            current_portfolio_market_value += current_price * pos_data['quantity']

            # Check Exits
            profit_target_price = pos_data['entry_price'] * (1 + target_profit_pct / 100.0)
            stop_loss_price = pos_data['entry_price'] * (1 - stop_loss_pct / 100.0)
            
            exit_reason = None
            if current_price >= profit_target_price: exit_reason = "Target Profit"
            elif current_price <= stop_loss_price: exit_reason = "Stop Loss"

            if exit_reason:
                sell_price = current_price # Assume exit at current day's close
                cash += sell_price * pos_data['quantity']
                trades_log.append({
                    'Ticker': ticker, 'Entry Date': pos_data['entry_date'], 'Entry Price': pos_data['entry_price'],
                    'Exit Date': current_date, 'Exit Price': sell_price, 'Quantity': pos_data['quantity'],
                    'P/L': (sell_price - pos_data['entry_price']) * pos_data['quantity'], 'Reason': exit_reason
                })
                tickers_to_sell_today.append(ticker)
        
        for ticker in tickers_to_sell_today:
            del positions[ticker]

        # 2. Check for Entries
        if len(positions) < max_concurrent_positions:
            potential_entries = []
            for ticker in tickers_to_test:
                if ticker in positions: continue # Already holding
                if ticker not in all_hist_data or all_hist_data[ticker].empty: continue

                # Get data slice up to *previous day* for signal generation to avoid lookahead bias
                # Signals generated on D-1 data are traded on D (current_date) open/close
                signal_date = current_date - pd.Timedelta(days=1) # Assuming signals from previous close
                if signal_date < bt_start_date_dt : continue

                df_slice_for_signal = all_hist_data[ticker].loc[all_hist_data[ticker].index <= signal_date]
                if df_slice_for_signal.empty or len(df_slice_for_signal) < 252: continue # Need enough history

                # Calculate indicators on this historical slice
                inds = calculate_indicators_for_date_slice(df_slice_for_signal)
                if inds is None or any(pd.isna(v) for v in [inds.get('52W_High'), inds.get('20D_High'), inds.get('50EMA')]):
                    continue
                
                entry_signal = False
                if setup_to_backtest == "Breakout 52w" and inds['Adj Close'] >= 0.99 * inds['52W_High']:
                    entry_signal = True
                elif setup_to_backtest == "Breakout" and inds['Adj Close'] >= 0.99 * inds['20D_High']:
                    entry_signal = True
                elif setup_to_backtest == "Retest" and inds['Adj Close'] >= inds['50EMA'] and inds['Adj Close'] <= 1.02 * inds['50EMA']:
                    entry_signal = True
                
                if entry_signal:
                    # Check if we have price data for current_date to make the trade
                    if current_date in all_hist_data[ticker].index and pd.notna(all_hist_data[ticker].loc[current_date, 'Adj Close']):
                        potential_entries.append({'ticker': ticker, 'entry_price': all_hist_data[ticker].loc[current_date, 'Adj Close']})
                    # else: could try to enter next day, or skip - simplified: skip if no price today

            # Prioritize entries (e.g., by some ranking if many signals, or just take first few)
            # For simplicity, just take them if slots available
            for entry_candidate in potential_entries:
                if len(positions) >= max_concurrent_positions: break
                
                ticker_to_buy = entry_candidate['ticker']
                actual_entry_price = entry_candidate['entry_price'] # Assumed entry at current_date close
                
                # Position Sizing based on % of current total portfolio value
                current_total_value_for_sizing = cash + current_portfolio_market_value # Value before new trades
                amount_to_invest = current_total_value_for_sizing * (investment_per_trade_pct / 100.0)
                
                if cash >= amount_to_invest and amount_to_invest > 0 and actual_entry_price > 0:
                    quantity = int(amount_to_invest / actual_entry_price)
                    if quantity > 0:
                        cost_of_trade = quantity * actual_entry_price
                        cash -= cost_of_trade
                        positions[ticker_to_buy] = {'entry_price': actual_entry_price, 'quantity': quantity, 'entry_date': current_date}
                        # Log entry (optional here, full log at exit)
                        current_portfolio_market_value += cost_of_trade # Add value of new position

        # Update daily portfolio value
        # Re-calculate market value of open positions for today's EOD portfolio value
        eod_market_value_of_positions = 0
        for ticker, pos_data in positions.items():
            if ticker in all_hist_data and current_date in all_hist_data[ticker].index and pd.notna(all_hist_data[ticker].loc[current_date, 'Adj Close']):
                eod_market_value_of_positions += all_hist_data[ticker].loc[current_date, 'Adj Close'] * pos_data['quantity']
            else: # If no price today, use last known value (approximation)
                 recent_slice = all_hist_data[ticker].loc[all_hist_data[ticker].index <= current_date]
                 if not recent_slice.empty and pd.notna(recent_slice['Adj Close'].iloc[-1]):
                    eod_market_value_of_positions += recent_slice['Adj Close'].iloc[-1] * pos_data['quantity']
                 else: # Fallback to entry price value
                    eod_market_value_of_positions += pos_data['entry_price'] * pos_data['quantity']


        portfolio_value = cash + eod_market_value_of_positions
        portfolio_history_daily.append({'Date': current_date, 'Portfolio_Value': portfolio_value})

    # Final mark-to-market for any open positions at the end of backtest period
    final_cash = cash
    final_market_value_of_positions = 0
    bt_end_date_actual = backtest_date_range[-1] # Actual last date used in loop

    for ticker, pos_data in positions.items():
        if ticker in all_hist_data:
            final_price_data = all_hist_data[ticker].loc[all_hist_data[ticker].index <= bt_end_date_actual]
            if not final_price_data.empty and pd.notna(final_price_data['Adj Close'].iloc[-1]):
                final_price = final_price_data['Adj Close'].iloc[-1]
                final_market_value_of_positions += final_price * pos_data['quantity']
                # Optionally log these as "closed at end of backtest"
                trades_log.append({
                    'Ticker': ticker, 'Entry Date': pos_data['entry_date'], 'Entry Price': pos_data['entry_price'],
                    'Exit Date': bt_end_date_actual, 'Exit Price': final_price, 'Quantity': pos_data['quantity'],
                    'P/L': (final_price - pos_data['entry_price']) * pos_data['quantity'], 'Reason': 'End of Backtest'
                })
            else: # Fallback if no price data on last day
                final_market_value_of_positions += pos_data['entry_price'] * pos_data['quantity']
        else: # Should not happen if stock was bought
             final_market_value_of_positions += pos_data['entry_price'] * pos_data['quantity']


    final_portfolio_val = final_cash + final_market_value_of_positions
    if not portfolio_history_daily or portfolio_history_daily[-1]['Date'] != bt_end_date_actual:
         portfolio_history_daily.append({'Date': bt_end_date_actual, 'Portfolio_Value': final_portfolio_val})
    else:
         portfolio_history_daily[-1]['Portfolio_Value'] = final_portfolio_val


    df_trades_final = pd.DataFrame(trades_log)
    df_portfolio_history_final = pd.DataFrame(portfolio_history_daily).set_index('Date')

    # Calculate KPIs
    total_return_pct = 0
    num_trades = 0
    win_rate_pct = 0
    if not df_portfolio_history_final.empty:
        total_return_pct = (df_portfolio_history_final['Portfolio_Value'].iloc[-1] / initial_capital - 1) * 100
    if not df_trades_final.empty:
        num_trades = len(df_trades_final)
        if num_trades > 0:
            win_rate_pct = (len(df_trades_final[df_trades_final['P/L'] > 0]) / num_trades * 100)
    
    # Max Drawdown Calculation
    max_drawdown_pct = 0
    if not df_portfolio_history_final.empty:
        roll_max = df_portfolio_history_final['Portfolio_Value'].cummax()
        daily_drawdown = df_portfolio_history_final['Portfolio_Value'] / roll_max - 1.0
        max_drawdown_pct = daily_drawdown.min() * 100


    return {
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "num_trades": num_trades,
        "win_rate_pct": win_rate_pct,
        "df_portfolio_history": df_portfolio_history_final,
        "df_trades": df_trades_final
    }

# --- UI for Backtesting Parameters ---
st.sidebar.header("Backtest Configuration")
index_choice_bt = st.sidebar.selectbox("Index Universe for Backtest", ["NIFTY 500", "NIFTY 200"], key="bt_index")
bt_list_to_load = "nifty500" if index_choice_bt == "NIFTY 500" else "nifty200"

# Use session state to remember dates
if 'bt_start_date' not in st.session_state:
    st.session_state.bt_start_date = datetime.today() - timedelta(days=365*2) # Default 2 years
if 'bt_end_date' not in st.session_state:
    st.session_state.bt_end_date = datetime.today() - timedelta(days=1)

bt_start_date = st.sidebar.date_input("Start Date", value=st.session_state.bt_start_date, key="bt_start")
bt_end_date = st.sidebar.date_input("End Date", value=st.session_state.bt_end_date, key="bt_end")
st.session_state.bt_start_date = bt_start_date # Update session state
st.session_state.bt_end_date = bt_end_date   # Update session state


initial_capital_bt = st.sidebar.number_input("Initial Capital", min_value=10000, value=100000, step=10000, key="bt_capital")
setup_to_backtest_bt = st.sidebar.selectbox(
    "Setup Type to Backtest", 
    ["Breakout 52w", "Breakout", "Retest"], 
    help="Select the entry signal from your momentum screener.",
    key="bt_setup"
)

st.sidebar.markdown("--- Exit Strategy ---")
target_profit_pct_bt = st.sidebar.slider("Target Profit (%)", 1, 100, 20, 1, key="bt_tp")
stop_loss_pct_bt = st.sidebar.slider("Stop Loss (%)", 1, 50, 10, 1, key="bt_sl")

st.sidebar.markdown("--- Portfolio Allocation ---")
max_concurrent_positions_bt = st.sidebar.slider("Max Concurrent Open Positions", 1, 20, 5, 1, key="bt_max_pos")
investment_per_trade_pct_bt = st.sidebar.slider("Investment per Trade (% of Portfolio)", 1, 100, 10, 1, key="bt_inv_pct")


run_backtest_button = st.sidebar.button("🚀 Run Backtest", type="primary", key="bt_run_button")

# --- Display Backtest Results ---
if run_backtest_button:
    if not bt_start_date or not bt_end_date or bt_start_date >= bt_end_date:
        st.error("Invalid backtest date range. Start date must be before end date.")
    elif initial_capital_bt <=0:
        st.error("Initial capital must be greater than zero.")
    else:
        with st.spinner(f"Loading {index_choice_bt} list for backtest..."):
            tickers_for_backtest = load_index_list_for_backtest(list_name=bt_list_to_load)
        
        if not tickers_for_backtest:
            st.error(f"Failed to load tickers for {index_choice_bt}.")
        else:
            # Fetch data for the selected tickers over the ENTIRE backtest period + buffer for indicators
            # Buffer for indicators like 252 day rolling max needs to be before bt_start_date
            data_fetch_start_str = (bt_start_date - timedelta(days=400)).strftime('%Y-%m-%d') 
            all_hist_data_bt = fetch_historical_data(
                tuple(tickers_for_backtest), 
                data_fetch_start_str, 
                bt_end_date.strftime('%Y-%m-%d') # Pass user-selected end date
            )

            if not all_hist_data_bt or len(all_hist_data_bt) < len(tickers_for_backtest) * 0.5: # Check if significant data was fetched
                st.warning("Could not fetch sufficient historical data for all selected tickers. Backtest results might be limited.")
            
            if not all_hist_data_bt: # If completely empty
                 st.error("Failed to fetch any historical data for backtesting.")
            else:
                with st.spinner(f"Executing backtest for '{setup_to_backtest_bt}' strategy on {index_choice_bt}..."):
                    backtest_results = run_backtest(
                        all_hist_data_bt,
                        tickers_for_backtest, # Pass the list of tickers
                        bt_start_date, 
                        bt_end_date,
                        initial_capital_bt,
                        target_profit_pct_bt,
                        stop_loss_pct_bt,
                        setup_to_backtest_bt,
                        max_concurrent_positions_bt,
                        investment_per_trade_pct_bt
                    )

                st.session_state.backtest_results = backtest_results # Store in session state

if 'backtest_results' in st.session_state:
    results = st.session_state.backtest_results
    if results:
        st.markdown("---")
        st.header("📜 Backtest Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{results['total_return_pct']:.2f}%")
        col2.metric("Max Drawdown", f"{results['max_drawdown_pct']:.2f}%")
        col3.metric("Number of Trades", results['num_trades'])
        col4.metric("Win Rate", f"{results['win_rate_pct']:.2f}%")

        if not results['df_portfolio_history'].empty:
            st.subheader("Equity Curve")
            # Ensure Portfolio_Value is numeric for plotting
            results['df_portfolio_history']['Portfolio_Value'] = pd.to_numeric(results['df_portfolio_history']['Portfolio_Value'], errors='coerce')
            st.line_chart(results['df_portfolio_history']['Portfolio_Value'])
        
        if not results['df_trades'].empty:
            st.subheader("Trades Log")
            st.dataframe(results['df_trades'].style.format({ # Basic formatting for trades log
                "Entry Price": "{:.2f}", "Exit Price": "{:.2f}", "P/L": "{:.2f}"
            }))
        else:
            st.info("No trades were executed during the backtest period with the given criteria.")
    else:
        st.error("Backtest execution failed or returned no results.")
else:
    st.info("Configure backtest parameters in the sidebar and click 'Run Backtest'.")


st.markdown("---")
st.markdown("Disclaimer: Backtesting results are based on historical data and do not guarantee future performance. This tool is for informational purposes only. DYOR.")
