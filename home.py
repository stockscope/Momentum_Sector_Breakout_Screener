# Home.py
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="📈 Stock Screener Hub",
    page_icon="🔍",  # Favicon
    layout="wide",  # Use wide layout for better content spacing
    initial_sidebar_state="expanded" # Keep sidebar expanded by default
)

# --- Custom CSS (from your original, with minor tweaks for clarity if any) ---
# This CSS attempts to keep the sidebar visible and expanded on mobile.
# Note: This can be aggressive on small screens. Streamlit's default responsive
# behavior usually hides it behind a hamburger menu.
st.markdown("""
    <style>
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f0f2f6; /* Light grey background for sidebar */
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
            color: #2c3e50; /* Dark blue for title */
        }

        /* Card-like styling for links (optional) */
        .card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            transition: transform 0.2s;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .card h3 {
            margin-top: 0;
            color: #3498db; /* Blue for card titles */
        }
        .card p {
            color: #555;
            font-size: 0.95rem;
        }
        .card a {
            text-decoration: none;
            color: inherit; /* Inherit color from parent for the whole card to be clickable */
        }

    </style>
""", unsafe_allow_html=True)


# --- Page Content ---

# Header Section
st.image("https://img.freepik.com/free-vector/stock-market-uptrend-arrow-chart-background_1017-39203.jpg?w=1060&t=st=1700000000~exp=1700000600~hmac=examplehash", use_column_width=True) # Replace with a real, good quality, royalty-free image URL or local path
st.title("📈 Welcome to the Stock Screener Hub!")
st.markdown("---")
st.markdown(
    """
    Discover potential investment opportunities in the Indian stock market using our specialized screeners. 
    These tools are designed to help you identify stocks based on momentum, value, and trend criteria.
    """
)
st.markdown("---")


# Navigation / Screener Links Section
st.header("🛠️ Our Screeners")
st.markdown("Select a screener from the sidebar or use the links below to get started:")

# Using columns for a better layout of screener descriptions
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <a href="NIFTY_200_Screener" target="_self">
            <div class="card">
                <h3>📊 NIFTY 200 Momentum Screener</h3>
                <p>Identifies breakout or retest setups in top-performing sectors within the NIFTY 200 universe based on trend, volume, and returns.</p>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <a href="NIFTY_500_Value_Screener" target="_self"> # Assuming your value screener page is named this
            <div class="card">
                <h3>📈 NIFTY 500 Value & Trend Screener</h3>
                <p>Scans the NIFTY 500 for stocks that appear reasonably valued and are exhibiting signs of a potential uptrend, suitable for various market conditions.</p>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <a href="NIFTY_500_Screener" target="_self">
            <div class="card">
                <h3>🚀 NIFTY 500 Momentum Screener</h3>
                <p>A broader momentum scan focusing on breakout/retest setups within the NIFTY 500, similar to the NIFTY 200 version but with a larger stock pool.</p>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )
    # Add more screeners here if you have them, e.g., an Advanced Screener
    # st.markdown(
    #     """
    #     <a href="Your_Other_Screener_Page_Name" target="_self">
    #         <div class="card">
    #             <h3>⚙️ Advanced Screener (Example)</h3>
    #             <p>Description of your advanced screener...</p>
    #         </div>
    #     </a>
    #     """,
    #     unsafe_allow_html=True
    # )


st.markdown("---")

# How to Use Section
st.subheader("💡 How to Use")
st.markdown(
    """
    1.  **Select a Screener:** Use the navigation links above or the sidebar menu (usually a `>` or `☰` icon on the top-left, especially on mobile) to choose your desired stock screener.
    2.  **Review Criteria:** Each screener page details the specific criteria it uses for filtering stocks.
    3.  **Analyze Results:** The screened stocks will be displayed in a table. You can review key metrics and download the data for further analysis.
    """
)

# Disclaimer Section
st.markdown("---")
st.subheader("⚠️ Important Disclaimer")
st.warning(
    """
    The information provided by these screeners is for educational and informational purposes only and should not be considered financial advice. 
    Stock market investments are subject to market risks. Always conduct your own thorough research (DYOR) and consult with a qualified financial advisor before making any investment decisions. 
    Data is sourced from Yahoo Finance and may be subject to inaccuracies or delays.
    """
)

# Footer
st.markdown("---")
st.markdown("Happy Screening! ✨")

# Note on st.page_link (if you upgrade Streamlit):
# If your Streamlit version is 1.29 or higher, you can use st.page_link for cleaner navigation:
# st.page_link("pages/1_NIFTY_200_Screener.py", label="NIFTY 200 Momentum Screener", icon="📊")
# st.page_link("pages/2_NIFTY_500_Screener.py", label="NIFTY 500 Momentum Screener", icon="🚀")
# st.page_link("pages/4_NIFTY_500_Value_Screener.py", label="NIFTY 500 Value & Trend Screener", icon="📈")
# This requires your pages to be in a 'pages' subdirectory.
