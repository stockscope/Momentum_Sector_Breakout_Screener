# Home.py
import streamlit as st

# Page Configuration for Home page - sidebar initially collapsed
st.set_page_config(
    page_title="StockScopePro Hub",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# --- Custom CSS ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #f0f2f6; 
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        h1 {
            color: #2c3e50; 
            text-align: center;
        }
        /* Removed h2, h3 styling for now as the "Our Screeners" header is removed */
        /* If you add other h2/h3, you might want to re-add specific styling */

        .card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
            border: 1px solid #e0e0e0;
            display: block; 
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        .card h3 { 
            margin-top: 0;
            color: #3498db; 
            margin-bottom: 0.75rem;
        }
        .card p {
            color: #555;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        .card a { 
            text-decoration: none;
            color: inherit; 
        }
        hr {
            border-top: 1px solid #eee;
            margin-top: 1.5rem; /* Adjusted margin */
            margin-bottom: 1.5rem; /* Adjusted margin */
        }
    </style>
""", unsafe_allow_html=True)


# --- Page Content ---
st.title("📈 Welcome to StockScopePro!")
# Removed the introductory markdown here
st.markdown("---") 

# Removed the "Our Screeners" header
st.markdown("Select a screener to begin your analysis:")

col1, col2 = st.columns(2)

with col1:
    # Screener 1: NIFTY 200 Momentum
    # IMPORTANT: Ensure href matches the filename in your 'pages' directory (without .py and leading numbers/underscores if desired for cleaner URLs, Streamlit handles it)
    # e.g., for "pages/1_NIFTY_200_Screener.py", href="1_NIFTY_200_Screener"
    st.markdown(
        """<a href="1_NIFTY_200_Screener" target="_self"><div class="card">
        <h3>📊 NIFTY 200 Momentum</h3>
        <p>Identifies breakout or retest setups in top-performing sectors within the NIFTY 200 universe.</p>
        </div></a>""", unsafe_allow_html=True
    )
    # Screener 2: NIFTY 500 Advanced (Moved here to balance columns)
    st.markdown(
        """<a href="3_NIFTY_500_Advanced_Screener" target="_self"><div class="card">
        <h3>⚙️ NIFTY 500 Advanced</h3>
        <p>Provides advanced filtering options for in-depth analysis of NIFTY 500 stocks. (Update this description!)</p>
        </div></a>""", unsafe_allow_html=True
    )


with col2:
    # Screener 3: NIFTY 500 Momentum
    st.markdown(
        """<a href="2_NIFTY_500_Screener" target="_self"><div class="card">
        <h3>🚀 NIFTY 500 Momentum</h3>
        <p>A broader momentum scan focusing on breakout/retest setups within the NIFTY 500 index.</p>
        </div></a>""", unsafe_allow_html=True
    )
    # Screener 4: NIFTY 500 Value & Trend
    st.markdown(
        """<a href="4_NIFTY_500_Value_Screener" target="_self"><div class="card">
        <h3>📈 NIFTY 500 Value & Trend</h3>
        <p>Scans the NIFTY 500 for stocks that appear reasonably valued and are exhibiting signs of an uptrend.</p>
        </div></a>""", unsafe_allow_html=True
    )

st.markdown("---")
st.subheader("💡 How to Use")
st.markdown(
    """
    1.  **Click on a Screener Card:** Choose one of the screeners listed above.
    2.  **Sidebar Navigation (on Screener Pages):** Once on a screener page, the sidebar will appear, allowing you to switch between different screeners easily.
    3.  **Understand Criteria:** Each screener page explains the specific criteria used for stock selection.
    4.  **Analyze & Download:** Review the results table and download data for your own further research.
    """
)

st.markdown("---")
st.subheader("⚠️ Important Disclaimer")
st.warning(
    """
    StockScopePro provides tools for informational and educational purposes only. It should **not** be considered financial advice. 
    Investing in the stock market involves risks, including the potential loss of principal. 
    Always conduct your own thorough research (DYOR) and consult with a qualified financial advisor before making any investment decisions. 
    Data is primarily sourced from Yahoo Finance and may be subject to inaccuracies or delays. We are not responsible for any trading or investment decisions based on the information provided by these tools.
    """
)
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Happy Screening with StockScopePro! ✨</p>", unsafe_allow_html=True)
