import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from io import BytesIO

# Set Streamlit page config
st.set_page_config(page_title="NIFTY 500 Swing Trade Screener", layout="wide")
st.title("📈 NIFTY 500 Swing Trade Screener")

@st.cache_data(show_spinner=False)
def load_nifty500():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    df = pd.read_csv(url)
    df = df[['Symbol', 'Company Name', 'Industry']]
    df['Symbol'] = df['Symbol'].str.upper()
    return df

def fetch_price_data(tickers):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=90)
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, group_by='ticker', progress=False, threads=True)
    return data

def calculate_indicators(df, ticker):
    df = df.copy()
    df['50_EMA'] = df['Close'].ewm(span=50).mean()
    df['20D_High'] = df['High'].rolling(window=20).max()
    df['20D_Avg_Vol'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Spike'] = df['Volume'] > 1.5 * df['20D_Avg_Vol']
    df['Breakout'] = (df['Close'] > df['20D_High'].shift(1)) & (df['Close'] > df['50_EMA'])
    return df.iloc[-1][['Close', '50_EMA', '20D_High', 'Volume', '20D_Avg_Vol', 'Volume_Spike', 'Breakout']]

def top_performing_sectors(df):
    sector_returns = {}
    for sector in df['Industry'].unique():
        sector_symbols = df[df['Industry'] == sector]['Symbol'].tolist()
        sector_data = fetch_price_data(sector_symbols)
        returns = []
        for symbol in sector_symbols:
            try:
                close_prices = sector_data[symbol]['Close']
                ret = close_prices.iloc[-1] / close_prices.iloc[-21] - 1
                returns.append(ret)
            except:
                continue
        if returns:
            sector_returns[sector] = sum(returns) / len(returns)
    sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)
    top_sectors = [sector for sector, _ in sorted_sectors[:5]]
    return top_sectors

nifty500_df = load_nifty500()
top_sectors = top_performing_sectors(nifty500_df)
filtered_df = nifty500_df[nifty500_df['Industry'].isin(top_sectors)]
st.subheader("Top Performing Sectors")
st.write(", ".join(top_sectors))

st.subheader("Screened Stocks")

price_data = fetch_price_data(filtered_df['Symbol'].tolist())

results = []
for _, row in filtered_df.iterrows():
    symbol = row['Symbol']
    company = row['Company Name']
    industry = row['Industry']
    try:
        indicators = calculate_indicators(price_data[symbol], symbol)
        if indicators['Breakout']:
            results.append({
                'Ticker': symbol,
                'Company': company,
                'Sector': industry,
                'Price': round(indicators['Close'], 2),
                '50 EMA': round(indicators['50_EMA'], 2),
                '20D High': round(indicators['20D_High'], 2),
                'Volume': int(indicators['Volume']),
                'Avg 20D Vol': int(indicators['20D_Avg_Vol']),
                'Volume Spike': indicators['Volume_Spike']
            })
    except:
        continue

results_df = pd.DataFrame(results)
if not results_df.empty:
    results_df = results_df.sort_values(by='Volume Spike', ascending=False)
    st.dataframe(results_df, use_container_width=True)

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📅 Download CSV",
        data=csv,
        file_name='swing_trade_screened_stocks.csv',
        mime='text/csv',
    )
else:
    st.warning("No breakout setups found in top-performing sectors.")
