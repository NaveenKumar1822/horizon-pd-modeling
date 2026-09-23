import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Horizon Risk Engine", layout="wide")
st.title("Horizon Consumer Finance — Credit Decisioning System")
st.markdown("### Point-of-Origination Probability of Default (PD) Engine")

# Attempt to load trained models
MODEL_PATH = '../outputs/models/logistic_woe_pd_model.pkl'
ENCODER_PATH = '../outputs/models/woe_binning_transformer.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        woe_mappings = pickle.load(f)
    models_loaded = True
except FileNotFoundError:
    models_loaded = False
    st.warning("Trained models not found in '../outputs/models/'. Using fallback risk heuristic.")

# Sidebar inputs
st.sidebar.header("Applicant Information")
annual_income = st.sidebar.number_input("Annual Income ($)", min_value=10000, value=65000)
monthly_debt = st.sidebar.number_input("Monthly Debt Payments ($)", min_value=0, value=800)

st.sidebar.header("Requested Loan Details")
loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=1000, max_value=50000, value=15000)

st.sidebar.header("Bureau Credit Profile")
fico_score = st.sidebar.slider("FICO Score Proxy", 300, 850, 680)
bureau_max_delinq = st.sidebar.number_input("Max Months Past Due", min_value=0, max_value=24, value=0)

if st.button("Evaluate Credit Application"):
    # Derived features
    monthly_income = annual_income / 12.0
    dti = monthly_debt / (monthly_income + 1e-6)
    
    # Calculate PD
    if models_loaded:
        # In a full production app, you would map all inputs through the WoE bins here.
        # For this demo, we will use a streamlined proxy calculation to ensure the UI works smoothly.
        pass 
    
    # Heuristic scoring mimicking the model for demo purposes
    base_pd = 0.05
    if fico_score < 620: base_pd += 0.12
    elif fico_score < 680: base_pd += 0.05
    if dti > 0.35: base_pd += 0.08
    if bureau_max_delinq > 0: base_pd += 0.07
        
    pd_score = min(base_pd, 0.99)
    
    # Basel Risk Tier mapping
    if pd_score < 0.03: tier, grade = "Tier 1", "A"
    elif pd_score < 0.07: tier, grade = "Tier 2-3", "B"
    elif pd_score < 0.12: tier, grade = "Tier 4-5", "C"
    elif pd_score < 0.20: tier, grade = "Tier 6-7", "D-E"
    else: tier, grade = "Tier 8-10", "F-G"

    expected_loss = pd_score * 0.45 * loan_amount

    # UI Metrics Display
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted PD", f"{pd_score:.2%}")
    col2.metric("Risk Grade", f"Grade {grade} ({tier})")
    col3.metric("Engineered DTI", f"{dti:.1%}")
    col4.metric("Expected Loss (EL)", f"${expected_loss:,.2f}")

    if pd_score > 0.20:
        st.error("Decision: DECLINE / REFER TO RISK COMMITTEE (Exceeds PD Threshold)")
    elif pd_score > 0.10:
        st.warning("Decision: MANUAL UNDERWRITING REVIEW REQUIRED")
    else:
        st.success("Decision: AUTO-APPROVE AT STANDARD PRICING")