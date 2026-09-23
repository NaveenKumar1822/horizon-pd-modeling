import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Horizon Risk Decision Engine", layout="wide")

st.title("Horizon Consumer Finance — Credit Decisioning System")
st.markdown("### Point-of-Origination Probability of Default (PD) Engine")

# Sidebar inputs
st.sidebar.header("Applicant Information")
applicant_age = st.sidebar.number_input("Applicant Age", min_value=18, max_value=80, value=35)
annual_income = st.sidebar.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000)
emp_length = st.sidebar.selectbox("Employment Length", ["< 1 year", "1-3 years", "4-9 years", "10+ years"])
home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])

st.sidebar.header("Requested Loan Details")
loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=1000, max_value=50000, value=15000)
term_months = st.sidebar.selectbox("Term (Months)", [36, 60])
interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=5.0, max_value=35.0, value=12.5)
monthly_debt = st.sidebar.number_input("Monthly Debt Payments ($)", min_value=0, max_value=10000, value=800)

st.sidebar.header("Bureau Credit Profile")
fico_score = st.sidebar.slider("FICO Score Proxy", 300, 850, 680)
bureau_num_accounts = st.sidebar.number_input("Total Bureau Accounts", min_value=0, max_value=50, value=8)
bureau_max_delinq = st.sidebar.number_input("Max Months Past Due", min_value=0, max_value=24, value=0)

# Decisioning Calculation
if st.button("Evaluate Credit Application"):
    # Derived features
    monthly_income = annual_income / 12.0
    dti = monthly_debt / (monthly_income + 1e-6)
    
    # Placeholder rule-based risk evaluation / Model inference logic
    # (In production, load saved woe_encoder.pkl and logistic_woe_model.pkl)
    
    # Simple risk scoring logic mirroring fitted model
    base_pd = 0.05
    if fico_score < 620:
        base_pd += 0.12
    elif fico_score < 680:
        base_pd += 0.05
        
    if dti > 0.35:
        base_pd += 0.08
    if bureau_max_delinq > 0:
        base_pd += 0.07
        
    pd_score = min(base_pd, 0.99)
    
    # Basel Risk Tier mapping
    if pd_score < 0.03:
        tier, grade = "Tier 1", "A"
    elif pd_score < 0.07:
        tier, grade = "Tier 2-3", "B"
    elif pd_score < 0.12:
        tier, grade = "Tier 4-5", "C"
    elif pd_score < 0.20:
        tier, grade = "Tier 6-7", "D-E"
    else:
        tier, grade = "Tier 8-10", "F-G"

    lgd = 0.45
    expected_loss = pd_score * lgd * loan_amount

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