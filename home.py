# Home.py
import streamlit as st

# Page Configuration for Home page - sidebar initially collapsed
st.set_page_config(
    page_title="StockScopePro Hub",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapsed on the home page
)

# --- Custom CSS ---
# Removed the aggressive mobile sidebar CSS as initial_sidebar_state handles it better
st.markdown("""
    <style>
        /* Sidebar styling (will apply when it's visible on other pages or if user opens it) */
        [data-testid="stSidebar"] {
            background-color: #f0f2f6; 
        }
        
        /* Main content area styling for better readability */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Title styling */
        h1 {
            color: #2c3e50; 
            text-align: center;
        }
        h2, h3 { /* For section headers */
            color: #2980b9; 
        }

        /* Card-like styling for links */
        .card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
            border: 1px solid #e0e0e0;
            display: block; /* Make the whole card area behave like a block for the link */
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
        .card a { /* Style for the link itself if needed, but we made the div clickable */
            text-decoration: none;
            color: inherit; 
        }
        hr {
            border-top: 1px solid #eee;
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)


# --- Page Content ---
st.title("📈 Welcome to StockScopePro!")
st.markdown("---")
st.markdown(
    """
    Your comprehensive hub for specialized stock screeners. 
    Navigate through our tools to identify potential investment opportunities in the Indian stock market 
    based on momentum, value, and trend criteria.
    """
)
st.markdown("---")

st.header("🛠️ Our Screeners")
st.markdown("Select a screener to begin your analysis:")

col1, col2 = st.columns(2)

# Ensure your page paths are correct. For a multi-page app, these would be
# relative to the 'pages' directory, e.g., "NIFTY_200_Screener" if the file is pages/NIFTY_200_Screener.py
# Streamlit constructs the URLs based on these filenames.

with col1:
    st.markdown( # Using st.page_link for cleaner navigation if Streamlit version >= 1.29
        """<a href="NIFTY_200_Screener" target="_self"><div class="card">
        <h3>📊 NIFTY 200 Momentum Screener</h3>
        <p>Identifies breakout or retest setups in top-performing sectors within the NIFTY 200 universe.</p>
        </div></a>""", unsafe_allow_html=True
    )
    st.markdown(
        """<a href="NIFTY_500_Value_Screener" target="_self"><div class="card">
        <h3>📈 NIFTY 500 Value & Trend Screener</h3>
        <p>Scans the NIFTY 500 for stocks that appear reasonably valued and are exhibiting signs of a potential uptrend.</p>
        </div></a>""", unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """<a href="NIFTY_500_Screener" target="_self"><div class="card">
        <h3>🚀 NIFTY 500 Momentum Screener</h3>
        <p>A broader momentum scan focusing on breakout/retest setups within the NIFTY 500 index.</p>
        </div></a>""", unsafe_allow_html=True
    )
    # Example for a fourth card if you add another screener
    # st.markdown(
    #     """<a href="Some_Other_Screener" target="_self"><div class="card">
    #     <h3>⚙️ Another Screener</h3>
    #     <p>Description of the other screener.</p>
    #     </div></a>""", unsafe_allow_html=True
    # )

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
