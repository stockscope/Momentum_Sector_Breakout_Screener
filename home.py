# Home.py
import streamlit as st
import os 
from pathlib import Path 

st.set_page_config(
    page_title="StockScopePro Hub",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

st.markdown("""
    <style>
        /* ... Your CSS ... */
        [data-testid="stSidebar"] { background-color: #f0f2f6; }
        .main .block-container { padding: 1rem 2rem 2rem 2rem; } /* Adjusted top padding */
        h1 { color: #2c3e50; text-align: center; margin-bottom: 0.5rem; }
        .card { background-color: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 1.5rem; transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; border: 1px solid #e0e0e0; display: block; height: 100%; }
        .card:hover { transform: translateY(-5px); box-shadow: 0 6px 12px rgba(0,0,0,0.15); }
        .card h3 { margin-top: 0; color: #3498db; margin-bottom: 0.75rem; font-size: 1.25rem; }
        .card p { color: #555; font-size: 0.9rem; line-height: 1.5; }
        .card a { text-decoration: none; color: inherit; }
        hr { border-top: 1px solid #eee; margin: 1rem 0; } /* Simplified hr margin */
        div[data-testid="stHorizontalBlock"] > div:first-child > div[data-testid="stVerticalBlock"] > div:first-child > div[data-testid="stVerticalBlock"] > div:first-child { margin-bottom: 0.5rem !important; }
        div[data-testid="stHorizontalBlock"] > div:first-child > div[data-testid="stVerticalBlock"] > div:nth-child(2) > div > hr { margin-top: 0.5rem !important; margin-bottom: 1rem !important; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Welcome to StockScopePro!")
st.markdown("---") 
st.markdown("<p style='text-align:center; font-size: 1.1em;'>Select a screener to begin your analysis:</p>", unsafe_allow_html=True)

current_script_dir = Path(__file__).parent
PAGES_DIR = current_script_dir / "pages"

def get_page_display_name_and_href(filename_path_obj):
    filename_stem = filename_path_obj.stem # e.g., "1_NIFTY_200_Screener"
    
    parts = filename_stem.split("_", 1)
    if len(parts) > 1 and parts[0].isdigit():
        # For href, Streamlit might be using the name without the number prefix
        href_base = parts[1] 
        display_name_base = parts[1]
    else:
        href_base = filename_stem
        display_name_base = filename_stem
    
    display_name = display_name_base.replace("_", " ").title()
    return display_name, href_base # Return the base name for href

def get_page_description(href_base_name): # Now takes the href_base_name
    custom_descriptions = {
        "NIFTY_200_Screener": "Identifies breakout or retest setups in top-performing sectors within the NIFTY 200 universe.",
        "NIFTY_500_Screener": "A broader momentum scan focusing on breakout/retest setups within the NIFTY 500 index.",
        "NIFTY_500_Advanced_Screener": "Advanced filtering for in-depth analysis of NIFTY 500 stocks. (Update description!)",
        "NIFTY_500_Value_Screener": "Scans the NIFTY 500 for stocks that appear reasonably valued and are exhibiting signs of an uptrend."
    }
    return custom_descriptions.get(href_base_name, "Explore this screener to find potential stock opportunities.")

def get_page_icon(href_base_name):
    custom_icons = {
        "NIFTY_200_Screener": "📊", 
        "NIFTY_500_Screener": "🚀", 
        "NIFTY_500_Advanced_Screener": "⚙️",
        "NIFTY_500_Value_Screener": "📈" 
    }
    return custom_icons.get(href_base_name, "🛠️")

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
            _, debug_href = get_page_display_name_and_href(f_path) # Get the href Streamlit likely uses
            st.sidebar.caption(f"File: {f_path.name}  -> Href generated: '{debug_href}'")
            
        num_screeners = len(screener_files)
        cols_per_row = 2 
        
        for i in range(0, num_screeners, cols_per_row):
            row_files = screener_files[i : i + cols_per_row]
            cols = st.columns(cols_per_row) 
            
            for idx, screener_file_path in enumerate(row_files):
                display_name, href_target = get_page_display_name_and_href(screener_file_path)
                description = get_page_description(href_target) # Use href_target (base name) for lookup
                icon = get_page_icon(href_target) # Use href_target (base name) for lookup
                
                with cols[idx]:
                    st.markdown(
                        f"""<a href="{href_target}" target="_self"><div class="card">
                        <h3>{icon} {display_name}</h3>
                        <p>{description}</p>
                        </div></a>""", 
                        unsafe_allow_html=True
                    )
            for j in range(len(row_files), cols_per_row):
                with cols[j]:
                    st.empty() 
else:
    st.warning(f"The 'pages' directory was not found at the expected location: {PAGES_DIR}.")

# ... (rest of the How to Use, Disclaimer, Footer sections remain the same) ...
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
