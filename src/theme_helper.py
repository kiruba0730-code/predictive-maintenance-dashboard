"""
Theme and custom styling helper for Predictive Maintenance Dashboard.
Allows real-time switching between Dark Industrial and Light Engineering themes,
as well as accessible color palette customization.
"""

import streamlit as st

THEME_STYLES = {
    "Dark Industrial": """
        <style>
        .stApp {
            background-color: #0F172A;
            color: #F8FAFC;
        }
        .metric-card {
            background-color: #1E293B;
            border-radius: 8px;
            padding: 16px 20px;
            border: 1px solid #334155;
            margin-bottom: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        .metric-title {
            color: #94A3B8;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            color: #F8FAFC;
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 4px;
        }
        .metric-delta {
            font-size: 0.8rem;
            color: #38BDF8;
            margin-top: 2px;
        }
        .status-badge-ok {
            background-color: rgba(16, 185, 129, 0.2);
            color: #10B981;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        .status-badge-warn {
            background-color: rgba(239, 68, 68, 0.2);
            color: #EF4444;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        </style>
    """,
    "Light Engineering": """
        <style>
        .stApp {
            background-color: #F8FAFC;
            color: #0F172A;
        }
        .metric-card {
            background-color: #FFFFFF;
            border-radius: 8px;
            padding: 16px 20px;
            border: 1px solid #CBD5E1;
            margin-bottom: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        .metric-title {
            color: #475569;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            color: #0F172A;
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 4px;
        }
        .metric-delta {
            font-size: 0.8rem;
            color: #0284C7;
            margin-top: 2px;
        }
        .status-badge-ok {
            background-color: #DCFCE7;
            color: #166534;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        .status-badge-warn {
            background-color: #FEE2E2;
            color: #991B1B;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        </style>
    """
}

def render_theme_sidebar():
    """Renders theme and color customizer controls in sidebar and injects CSS."""
    st.sidebar.subheader("🎨 Dashboard Appearance")
    
    col_th1, col_th2 = st.sidebar.columns(2)
    with col_th1:
        theme_choice = st.selectbox(
            "Theme Mode:",
            options=["Dark Industrial", "Light Engineering"],
            index=0 if st.session_state.get("theme_mode", "Dark Industrial") == "Dark Industrial" else 1,
            key="theme_mode_selector",
        )
    with col_th2:
        palette_choice = st.selectbox(
            "Color Palette:",
            options=["Emerald & Crimson", "Cyan & Rose", "Cobalt & Orange"],
            index=0,
            key="palette_selector",
        )

    st.session_state["theme_mode"] = theme_choice
    st.session_state["chart_palette"] = palette_choice

    # Inject dynamic CSS based on chosen theme
    css = THEME_STYLES.get(theme_choice, THEME_STYLES["Dark Industrial"])
    st.markdown(css, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    return theme_choice, palette_choice
