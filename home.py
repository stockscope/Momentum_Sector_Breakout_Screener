# Home.py
import streamlit as st
import os 
from pathlib import Path 

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
            padding-top: 1rem; /* Reduced top padding */
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        h1 {
            color: #2c3e50; 
            text-align: center;
            margin-bottom: 0.5rem; /* Reduced margin below title */
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
            height: 100%; 
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        .card h3 { 
            margin-top: 0;
            color: #3498db; 
            margin-bottom: 0.75rem;
            font-size: 1.25rem; /* Slightly larger card titles */
        }
        .card p {
            color: #555;
            font-size: 0.9rem; /* Slightly smaller card paragraph text */
            line-height: 1.5;
        }
        .card a { 
            text-decoration: none;
            color: inherit; 
        }
        hr {
            border-top: 1px solid #eee;
            margin-top: 1rem; /* Reduced hr margin */
            margin-bottom: 1rem; /* Reduced hr margin */
        }
        /* Reduce space around the main title and first hr */
        div[data-testid="stHorizontalBlock"] > div:first-child > div[data-testid="stVerticalBlock"] > div:first-child > div[data-testid="stVerticalBlock"] > div:first-child {
            margin-bottom: 0.5rem !important;
        }
        div[data-testid="stHorizontalBlock"] > div:first-child > div[data-testid="stVerticalBlock"] > div:nth-child(2) > div > hr {
            margin-top: 0.5rem !important;
            margin-bottom: 1rem !important;
        }


    </style>
""", unsafe_allow_html=True)


# --- Page Content ---
st.title("📈 Welcome to StockScopePro!")
st.markdown("---") 
st.markdown("<p style='text-align:center; font-size: 1.1em;'>Select a screener to begin your analysis:</p>", unsafe_allow_html=True)


# --- Dynamic Screener Listing ---
# Get the directory of the current script (Home.py)
current_script_dir = Path(__file__).parent
# Define the path to your pages directory relative to Home.py
PAGES_DIR = current_script_dir / "pages"


# Function to generate a user-friendly name from a filename
def get_page_display_name(filename_path_obj):
    filename_no_ext = filename_path_obj.stem 
    # Attempt to strip leading "number_" pattern for display name
    parts = filename_no_ext.split("_", 1)
    if len(parts) > 1 and parts[0].isdigit():
        display_name_base = parts[1]
    else:
        display_name_base = filename_no_ext
    
    display_name = display_name_base.replace("_", " ").title()
    return display_name

# Function to get a generic description or allow for custom ones
def get_page_description(filename_stem): # Takes filename without .py
    custom_descriptions = {
        "NIFTY_200_Screener": "Identifies breakout or retest setups in top-performing sectors within the NIFTY 200 universe.",
        "1_NIFTY_200_Screener": "Identifies breakout or retest setups in top-performing sectors within the NIFTY 200 universe.", # Fallback if number not stripped for key
        "NIFTY_500_Screener": "A broader momentum scan focusing on breakout/retest setups within the NIFTY 500 index.",
        "2_NIFTY_500_Screener": "A broader momentum scan focusing on breakout/retest setups within the NIFTY 500 index.",
        "NIFTY_500_Advanced_Screener": "Advanced filtering for in-depth analysis of NIFTY 500 stocks. (Update this description!)",
        "3_NIFTY_500_Advanced_Screener": "Advanced filtering for in-depth analysis of NIFTY 500 stocks. (Update this description!)",
        "NIFTY_500_Value_Screener": "Scans the NIFTY 500 for stocks that appear reasonably valued and are exhibiting signs of an uptrend.",
        "4_NIFTY_500_Value_Screener": "Scans the NIFTY 500 for stocks that appear reasonably valued and are exhibiting signs of an uptrend."
    }
    # Try to match without number prefix first for cleaner keys
    base_name = filename_stem.split("_", 1)[1] if filename_stem.split("_", 1)[0].isdigit() and len(filename_stem.split("_", 1)) > 1 else filename_stem
    return custom_descriptions.get(base_name, custom_descriptions.get(filename_stem, "Explore this screener to find potential stock opportunities."))


def get_page_icon(filename_stem):
    custom_icons = {
        "NIFTY_200_Screener": "📊", "1_NIFTY_200_Screener": "📊",
        "NIFTY_500_Screener": "🚀", "2_NIFTY_500_Screener": "🚀",
        "NIFTY_500_Advanced_Screener": "⚙️", "3_NIFTY_500_Advanced_Screener": "⚙️",
        "NIFTY_500_Value_Screener": "📈", "4_NIFTY_500_Value_Screener": "📈"
    }
    base_name = filename_stem.split("_", 1)[1] if filename_stem.split("_", 1)[0].isdigit() and len(filename_stem.split("_", 1)) > 1 else filename_stem
    return custom_icons.get(base_name, custom_icons.get(filename_stem, "🛠️"))


if PAGES_DIR.is_dir():
    screener_files = sorted([
        f for f in PAGES_DIR.iterdir() 
        if f.is_file() and f.suffix == '.py' and 
           not f.name.startswith('.') and not f.name == "__init__.py"
    ])
    
    if not screener_files:
        st.info("No screener pages found in the 'pages' directory.")
    else:
        st.sidebar.subheader("Debug: Page Links")
        for f_path in screener_files:
            debug_href = f_path.stem # Filename without .py, including any numbers_
            st.sidebar.caption(f"File: {f_path.name}  -> Href used: '{debug_href}'")
            
        num_screeners = len(screener_files)
        cols_per_row = 2 
        
        for i in range(0, num_screeners, cols_per_row):
            row_files = screener_files[i : i + cols_per_row]
            # Ensure enough columns are created, even if last row is not full
            cols = st.columns(cols_per_row) # Always create full number of columns for the row design
            
            for idx, screener_file_path in enumerate(row_files):
                filename_stem_for_href = screener_file_path.stem # e.g., "1_NIFTY_200_Screener" or "NIFTY_200_Screener"
                
                display_name = get_page_display_name(screener_file_path) # Tries to make it pretty
                description = get_page_description(filename_stem_for_href) # Uses stem for lookup
                icon = get_page_icon(filename_stem_for_href) # Uses stem for lookup
                
                with cols[idx]:
                    st.markdown(
                        f"""<a href="{filename_stem_for_href}" target="_self"><div class="card">
                        <h3>{icon} {display_name}</h3>
                        <p>{description}</p>
                        </div></a>""", 
                        unsafe_allow_html=True
                    )
            # If the last row has fewer items than cols_per_row, fill remaining columns with empty space
            for j in range(len(row_files), cols_per_row):
                with cols[j]:
                    st.empty() # Keeps the column structure
else:
    st.warning(f"The 'pages' directory was not found at the expected location: {PAGES_DIR}. Please create it relative to Home.py and add your screener Python files there.")

st.markdown("---")
st.subheader("💡 How to Use")
st.markdown(
    """
    1.  **Click on a Screener Card:** Choose one of the screeners listed above.
    2.  **Sidebar Navigation (on Screener Pages):** Once on a screener page, the sidebar will appear, allowing you to switch between different screeners easily. Streamlit also adds discovered pages to the sidebar automatically.
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
