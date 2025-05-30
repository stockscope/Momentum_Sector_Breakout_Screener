# Home.py
import streamlit as st
import os # Import the os module
from pathlib import Path # For more robust path handling

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
        /* ... (Your existing CSS for cards, etc.) ... */
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
        .card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
            border: 1px solid #e0e0e0;
            display: block; 
            height: 100%; /* Make cards in a row same height */
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
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)


# --- Page Content ---
st.title("📈 Welcome to StockScopePro!")
st.markdown("---") 
st.markdown("Select a screener to begin your analysis:")

# --- Dynamic Screener Listing ---
PAGES_DIR = Path("pages") # Define the path to your pages directory

# Function to generate a user-friendly name from a filename
def get_page_display_name(filename_with_ext):
    filename_no_ext = filename_with_ext.stem # Removes .py
    # Replace underscores with spaces and capitalize words
    display_name = filename_no_ext.replace("_", " ").title()
    return display_name

# Function to get a generic description or allow for custom ones
def get_page_description(filename_no_ext):
    # You can create a dictionary for custom descriptions if needed
    custom_descriptions = {
        "NIFTY_200_Screener": "Identifies breakout or retest setups in top-performing sectors within the NIFTY 200 universe.",
        "NIFTY_500_Screener": "A broader momentum scan focusing on breakout/retest setups within the NIFTY 500 index.",
        "NIFTY_500_Advanced_Screener": "Advanced filtering options for in-depth analysis of NIFTY 500 stocks.",
        "NIFTY_500_Value_Screener": "Scans the NIFTY 500 for stocks that appear reasonably valued and are exhibiting signs of an uptrend."
    }
    return custom_descriptions.get(filename_no_ext, "Explore this screener to find potential stock opportunities.")

# Function to get a generic icon or allow for custom ones
def get_page_icon(filename_no_ext):
    custom_icons = {
        "NIFTY_200_Screener": "📊",
        "NIFTY_500_Screener": "🚀",
        "NIFTY_500_Advanced_Screener": "⚙️",
        "NIFTY_500_Value_Screener": "📈"
    }
    return custom_icons.get(filename_no_ext, "🛠️")


if PAGES_DIR.is_dir():
    # Get .py files, sort them (e.g., alphabetically for consistent order)
    # Exclude files starting with '.' (like .DS_Store) or '__init__.py'
    screener_files = sorted([f for f in PAGES_DIR.iterdir() if f.is_file() and f.suffix == '.py' and not f.name.startswith('.') and not f.name == "__init__.py"])
    
    if not screener_files:
        st.info("No screener pages found in the 'pages' directory.")
    else:
        # Create columns dynamically based on number of screeners, aiming for 2-3 per row
        num_screeners = len(screener_files)
        cols_per_row = 2 # You can adjust this
        
        # Create rows of columns
        for i in range(0, num_screeners, cols_per_row):
            row_files = screener_files[i : i + cols_per_row]
            cols = st.columns(len(row_files)) # Create as many columns as files in this row
            
            for idx, screener_file_path in enumerate(row_files):
                filename_no_ext = screener_file_path.stem # e.g., "NIFTY_200_Screener"
                display_name = get_page_display_name(screener_file_path)
                description = get_page_description(filename_no_ext)
                icon = get_page_icon(filename_no_ext)
                
                # Streamlit constructs the URL path from the filename in the pages dir
                # The href should be just the filename without .py
                href_target = filename_no_ext 

                with cols[idx]:
                    st.markdown(
                        f"""<a href="{href_target}" target="_self"><div class="card">
                        <h3>{icon} {display_name}</h3>
                        <p>{description}</p>
                        </div></a>""", 
                        unsafe_allow_html=True
                    )
else:
    st.warning("The 'pages' directory was not found. Please create it and add your screener Python files there.")


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
