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

@st.cache_data(ttl=timedelta(hours=4), show_spinner=False)
def fetch_historical_data(tickers_tuple, start_date_str, end_date_str):
    tickers_list = list(tickers_tuple)
    if not tickers_list: return {}
    try:
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
    if len(df_slice) < 20: return None 
    indicators = {}
    indicators['Adj Close'] = df_slice['Adj Close'].iloc[-1]
    indicators['High'] = df_slice['High'].iloc[-1]
    indicators['Volume'] = df_slice['Volume'].iloc[-1]

    if len(df_slice) >= 50:
        indicators['50EMA'] = df_slice['Adj Close'].ewm(span=50, adjust=False, min_periods=45).mean().iloc[-1]
    else: indicators['50EMA'] = np.nan
    indicators['20D_High'] = df_slice['High'].rolling(window=20, min_periods=15).max().iloc[-1]
    if len(df_slice) >= 252:
        indicators['52W_High'] = df_slice['High'].rolling(window=252, min_periods=200).max().iloc[-1]
    else: indicators['52W_High'] = np.nan
    return indicators

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False) 
def run_backtest(
    all_hist_data, tickers_to_test_tuple, bt_start_date_input, bt_end_date_input,   
    initial_capital, target_profit_pct, stop_loss_pct, setup_to_backtest,
    max_concurrent_positions=5, investment_per_trade_pct = 10
):
    tickers_to_test = list(tickers_to_test_tuple)
    portfolio_value = initial_capital; cash = initial_capital
    positions = {}; trades_log = []
    portfolio_history_daily = [{'Date': pd.Timestamp(bt_start_date_input) - pd.Timedelta(days=1), 'Portfolio_Value': initial_capital}]
    bt_start_date_ts = pd.Timestamp(bt_start_date_input)
    bt_end_date_ts = pd.Timestamp(bt_end_date_input)

    for ticker_df in all_hist_data.values():
        if not isinstance(ticker_df.index, pd.DatetimeIndex):
            all_hist_data[ticker_df.name if hasattr(ticker_df, 'name') else list(all_hist_data.keys())[list(all_hist_data.values()).index(ticker_df)]].index = pd.to_datetime(ticker_df.index)


    backtest_date_range = pd.date_range(start=bt_start_date_ts, end=bt_end_date_ts, freq='B')
    if backtest_date_range.empty: return None

    for current_date in stqdm(backtest_date_range, desc="Backtesting Days"):
        current_portfolio_market_value_of_positions = 0
        tickers_to_sell_today = []
        for ticker, pos_data in positions.items():
            if ticker not in all_hist_data or all_hist_data[ticker].empty: continue
            current_price = pos_data['entry_price'] 
            if current_date in all_hist_data[ticker].index and pd.notna(all_hist_data[ticker].loc[current_date, 'Adj Close']):
                current_price = all_hist_data[ticker].loc[current_date, 'Adj Close']
            else: 
                price_slice = all_hist_data[ticker].loc[all_hist_data[ticker].index <= current_date, 'Adj Close'].dropna()
                if not price_slice.empty: current_price = price_slice.iloc[-1]
            current_portfolio_market_value_of_positions += current_price * pos_data['quantity']
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

        if len(positions) < max_concurrent_positions:
            potential_entries = []
            for ticker in tickers_to_test:
                if ticker in positions or ticker not in all_hist_data or all_hist_data[ticker].empty: continue
                signal_date = current_date - pd.Timedelta(days=1) 
                if signal_date < bt_start_date_ts : continue 
                df_slice_for_signal = all_hist_data[ticker].loc[all_hist_data[ticker].index <= signal_date]
                if df_slice_for_signal.empty or len(df_slice_for_signal) < 252: continue
                inds = calculate_indicators_for_date_slice(df_slice_for_signal)
                if inds is None or any(pd.isna(inds.get(k)) for k in ['Adj Close','50EMA', '20D_High', '52W_High']): continue
                entry_signal = False; adj_close_sig=inds['Adj Close']; ema50_sig=inds['50EMA']; d20high_sig=inds['20D_High']; w52high_sig=inds['52W_High']
                if setup_to_backtest == "Breakout 52w" and adj_close_sig >= 0.99 * w52high_sig: entry_signal = True
                elif setup_to_backtest == "Breakout" and adj_close_sig >= 0.99 * d20high_sig: entry_signal = True
                elif setup_to_backtest == "Retest" and adj_close_sig >= ema50_sig and adj_close_sig <= 1.02 * ema50_sig: entry_signal = True
                if entry_signal and current_date in all_hist_data[ticker].index and pd.notna(all_hist_data[ticker].loc[current_date, 'Adj Close']):
                    potential_entries.append({'ticker': ticker, 'entry_price': all_hist_data[ticker].loc[current_date, 'Adj Close']})
            for entry_candidate in potential_entries:
                if len(positions) >= max_concurrent_positions: break
                ticker_to_buy = entry_candidate['ticker']; actual_entry_price = entry_candidate['entry_price']
                open_pos_value = sum((all_hist_data[t].loc[current_date, 'Adj Close'] if current_date in all_hist_data[t].index and pd.notna(all_hist_data[t].loc[current_date, 'Adj Close']) else pos['entry_price']) * pos['quantity'] for t, pos in positions.items())
                current_total_value_for_sizing = cash + open_pos_value
                amount_to_invest = current_total_value_for_sizing * (investment_per_trade_pct / 100.0)
                if cash >= amount_to_invest and amount_to_invest > 0 and actual_entry_price > 0:
                    quantity = int(amount_to_invest / actual_entry_price)
                    if quantity > 0:
                        cost_of_trade = quantity * actual_entry_price; cash -= cost_of_trade
                        positions[ticker_to_buy] = {'entry_price': actual_entry_price, 'quantity': quantity, 'entry_date': current_date}
        eod_market_value_of_positions = 0
        for ticker, pos_data in positions.items():
            price_to_use = pos_data['entry_price']
            if ticker in all_hist_data and current_date in all_hist_data[ticker].index and pd.notna(all_hist_data[ticker].loc[current_date, 'Adj Close']):
                price_to_use = all_hist_data[ticker].loc[current_date, 'Adj Close']
            else:
                 price_slice = all_hist_data[ticker].loc[all_hist_data[ticker].index <= current_date, 'Adj Close'].dropna()
                 if not price_slice.empty: price_to_use = price_slice.iloc[-1]
            eod_market_value_of_positions += price_to_use * pos_data['quantity']
        portfolio_value = cash + eod_market_value_of_positions
        portfolio_history_daily.append({'Date': current_date, 'Portfolio_Value': portfolio_value})

    final_cash = cash; final_market_value_of_positions = 0
    bt_end_date_actual = backtest_date_range[-1] if not backtest_date_range.empty else bt_end_date_ts
    for ticker, pos_data in positions.items():
        final_price = pos_data['entry_price'] 
        if ticker in all_hist_data:
            price_slice = all_hist_data[ticker].loc[all_hist_data[ticker].index <= bt_end_date_actual, 'Adj Close'].dropna()
            if not price_slice.empty: final_price = price_slice.iloc[-1]
        final_market_value_of_positions += final_price * pos_data['quantity']
        if not any(t['Ticker'] == ticker and t['Reason'] == 'End of Backtest' for t in trades_log):
            trades_log.append({'Ticker': ticker, 'Entry Date': pos_data['entry_date'], 'Entry Price': pos_data['entry_price'],
                               'Exit Date': bt_end_date_actual, 'Exit Price': final_price, 'Quantity': pos_data['quantity'],
                               'P/L': (final_price - pos_data['entry_price']) * pos_data['quantity'], 'Reason': 'End of Backtest'})
    final_portfolio_val = final_cash + final_market_value_of_positions
    if portfolio_history_daily:
        if portfolio_history_daily[-1]['Date'] != bt_end_date_actual : portfolio_history_daily.append({'Date': bt_end_date_actual, 'Portfolio_Value': final_portfolio_val})
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
index_choice_bt = st.sidebar.selectbox("Index Universe", ["NIFTY 500", "NIFTY 200"], key="bt_idx_choice_sidebar") # Changed key
bt_list_to_load = "nifty500" if index_choice_bt == "NIFTY 500" else "nifty200"

if 'bt_start_date' not in st.session_state: st.session_state.bt_start_date = datetime.today() - timedelta(days=365*2)
if 'bt_end_date' not in st.session_state: st.session_state.bt_end_date = datetime.today() - timedelta(days=1)

bt_start_date = st.sidebar.date_input("Start Date", value=st.session_state.bt_start_date, key="bt_start_dt_picker") # Changed key
bt_end_date = st.sidebar.date_input("End Date", value=st.session_state.bt_end_date, max_value=datetime.today() - timedelta(days=1), key="bt_end_dt_picker") # Changed key
st.session_state.bt_start_date = bt_start_date; st.session_state.bt_end_date = bt_end_date

initial_capital_bt = st.sidebar.number_input("Initial Capital", min_value=10000, value=100000, step=10000, key="bt_cap_input") # Changed key
setup_to_backtest_bt = st.sidebar.selectbox("Setup Type", ["Breakout 52w", "Breakout", "Retest"], key="bt_setup_sel") # Changed key
st.sidebar.markdown("--- Exit Strategy ---")
target_profit_pct_bt = st.sidebar.slider("Target Profit (%)", 1, 100, 20, 1, key="bt_tp_sld") # Changed key
stop_loss_pct_bt = st.sidebar.slider("Stop Loss (%)", 1, 50, 10, 1, key="bt_sl_sld") # Changed key
st.sidebar.markdown("--- Portfolio Allocation ---")
max_concurrent_positions_bt = st.sidebar.slider("Max Open Positions", 1, 20, 5, 1, key="bt_max_pos_sld") # Changed key
investment_per_trade_pct_bt = st.sidebar.slider("Investment/Trade (%)", 1, 50, 10, 1, key="bt_inv_pct_sld") # Changed key

run_backtest_button = st.sidebar.button("🚀 Run Backtest", type="primary", key="bt_run_btn_main") # Changed key

if run_backtest_button:
    if not bt_start_date or not bt_end_date or bt_start_date >= bt_end_date: st.error("Invalid date range.")
    elif initial_capital_bt <=0: st.error("Initial capital must be > 0.")
    else:
        with st.spinner(f"Loading {index_choice_bt} list..."): tickers_for_backtest = load_index_list_for_backtest(list_name=bt_list_to_load)
        if not tickers_for_backtest: st.error(f"Failed to load tickers for {index_choice_bt}.")
        else:
            data_fetch_start_str = (bt_start_date - timedelta(days=400)).strftime('%Y-%m-%d') 
            all_hist_data_bt = fetch_historical_data(tuple(tickers_for_backtest), data_fetch_start_str, bt_end_date.strftime('%Y-%m-%d'))
            if not all_hist_data_bt: st.error("Failed to fetch historical data.")
            else:
                with st.spinner(f"Executing backtest: '{setup_to_backtest_bt}' on {index_choice_bt}... This may take time."):
                    backtest_results_run = run_backtest(all_hist_data_bt, tuple(tickers_for_backtest), bt_start_date, bt_end_date,
                                                        initial_capital_bt, target_profit_pct_bt, stop_loss_pct_bt,
                                                        setup_to_backtest_bt, max_concurrent_positions_bt, investment_per_trade_pct_bt)
                st.session_state.backtest_results_data = backtest_results_run
                st.session_state.backtest_params_used = { # Store parameters
                    "Index Universe": index_choice_bt, "Start Date": bt_start_date.strftime("%Y-%m-%d"),
                    "End Date": bt_end_date.strftime("%Y-%m-%d"), "Initial Capital": f"₹{initial_capital_bt:,.0f}",
                    "Setup Type": setup_to_backtest_bt, "Target Profit (%)": target_profit_pct_bt,
                    "Stop Loss (%)": stop_loss_pct_bt, "Max Open Positions": max_concurrent_positions_bt,
                    "Investment/Trade (%)": investment_per_trade_pct_bt
                }
                st.rerun()

if 'backtest_results_data' in st.session_state:
    results = st.session_state.backtest_results_data
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
        
        if 'backtest_params_used' in st.session_state: # Display parameters
            st.markdown("---"); st.subheader("📋 Backtest Parameters Used")
            params_to_display = st.session_state.backtest_params_used
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
