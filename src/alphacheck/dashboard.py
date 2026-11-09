"""Interactive Streamlit Dashboard for AlphaCheck research results."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="AlphaCheck Dashboard", layout="wide")
st.title("AlphaCheck: Quantitative Research Framework on Nifty 50")
st.markdown(
    "Leakage-safe walk-forward validation testing whether ML and AI models beat simple baselines, net of transaction costs."
)

tab1, tab2, tab3 = st.tabs(["Performance Summary", "Equity Curves & Drawdowns", "Robustness & SHAP"])

with tab1:
    st.subheader("Out-of-Sample Performance Scorecard")
    results_path = "reports/tables/results_summary.csv"
    if Path(results_path).exists():
        df = pd.read_csv(results_path)
        st.dataframe(df.style.format(precision=4))
    else:
        st.info("Run `python -m alphacheck.run_all` to generate summary metrics.")

with tab2:
    st.subheader("Cumulative Portfolio Trajectories")
    fig_path = "reports/figures/equity_curves.png"
    if Path(fig_path).exists():
        st.image(fig_path, caption="Out-of-Sample Equity Curves (Net of 12 bps Friction)")
    else:
        st.info("Figures will appear here after executing the pipeline.")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cost Sensitivity Breakdown")
        cost_path = "reports/tables/cost_sensitivity.csv"
        if Path(cost_path).exists():
            cdf = pd.read_csv(cost_path)
            st.dataframe(cdf)
    with col2:
        st.subheader("Tree SHAP Attribution")
        shap_path = "reports/figures/shap_summary.png"
        if Path(shap_path).exists():
            st.image(shap_path, caption="Mean |SHAP| Feature Importances")

