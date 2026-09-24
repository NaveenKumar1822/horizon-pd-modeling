import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==============================================================================
# Horizon Consumer Finance — Point-of-Origination PD Decision Engine
# Uses the same WoE mappings, Logistic Regression model, and prior-shift
# calibration logic used in the modeling notebook.
# ==============================================================================

st.set_page_config(
    page_title="Horizon Risk Decision Engine",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "outputs" / "models" / "logistic_woe_pd_model.pkl"
WOE_PATH = BASE_DIR / "outputs" / "models" / "woe_binning_transformer.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "model_ready_joined_dataset.csv"

FEATURES = [
    "applicant_age",
    "annual_income",
    "emp_length_years",
    "loan_amount",
    "term_months",
    "interest_rate",
    "fico_score_proxy",
    "dti_engineered",
    "revol_util_clean",
    "bureau_num_accounts",
    "bureau_active_accounts",
    "bureau_total_credit_limit",
    "bureau_total_current_balance",
    "bureau_max_months_past_due",
    "bureau_avg_account_age",
]

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(WOE_PATH, "rb") as f:
        woe_mappings = pickle.load(f)

    return model, woe_mappings


@st.cache_data
def load_training_prevalence():
    """
    Recreates the notebook's train/test split so the same balanced-class
    prior-shift calibration can be applied to new model predictions.
    """
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df["target"]

    _, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    return y_train


def transform_feature_with_woe(value, mapping):
    """
    Apply the exact saved WoE mapping for one feature.

    The notebook saves a dictionary whose keys are pandas Interval objects
    and whose values are WoE scores. New values are assigned to the interval
    containing them. Values outside the training range use the nearest
    boundary bin.
    """
    if pd.isna(value):
        return 0.0

    intervals = list(mapping.keys())

    if not intervals:
        return 0.0

    # Normal case: value falls inside one of the training intervals.
    for interval in intervals:
        if interval.left <= value <= interval.right:
            return float(mapping[interval])

    # Out-of-range values: use the closest boundary interval.
    if value < intervals[0].left:
        return float(mapping[intervals[0]])

    if value > intervals[-1].right:
        return float(mapping[intervals[-1]])

    return 0.0


def transform_to_woe(input_df, woe_mappings):
    transformed = pd.DataFrame(index=input_df.index)

    for feature in FEATURES:
        transformed[feature + "_woe"] = input_df[feature].apply(
            lambda value: transform_feature_with_woe(
                value,
                woe_mappings[feature]
            )
        )

    return transformed


def calibrate_balanced_probability(raw_probability, y_train):
    """
    Same logit prior-shift calibration used in the notebook.

    The Logistic Regression was trained with class_weight='balanced'.
    This reverses the prior adjustment introduced by balanced class weights.
    """
    n = len(y_train)
    n_pos = y_train.sum()
    n_neg = n - n_pos

    w1 = n / (2.0 * n_pos)
    w0 = n / (2.0 * n_neg)

    raw_probability = float(np.clip(raw_probability, 1e-7, 1 - 1e-7))

    logit_raw = np.log(raw_probability / (1.0 - raw_probability))
    logit_calibrated = logit_raw - np.log(w1 / w0)

    return float(1.0 / (1.0 + np.exp(-logit_calibrated)))


def assign_risk_tier(pd_score):
    """
    Risk bands based on the project's displayed 10-tier framework.
    """
    if pd_score < 0.03:
        return "Tier 1", "A"
    elif pd_score < 0.07:
        return "Tier 2–3", "B"
    elif pd_score < 0.12:
        return "Tier 4–5", "C"
    elif pd_score < 0.20:
        return "Tier 6–7", "D–E"
    else:
        return "Tier 8–10", "F–G"


# ------------------------------------------------------------------------------
# Load trained artifacts
# ------------------------------------------------------------------------------
try:
    model, woe_mappings = load_artifacts()
    y_train = load_training_prevalence()
    models_loaded = True
    load_error = None
except Exception as exc:
    models_loaded = False
    load_error = str(exc)

# ------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------
st.title("Horizon Consumer Finance — Credit Decisioning System")
st.markdown("### Point-of-Origination Probability of Default (PD) Engine")
st.caption(
    "Interactive demonstration of the trained WoE + Logistic Regression "
    "credit-risk modeling pipeline."
)

if not models_loaded:
    st.error(
        "The trained model artifacts could not be loaded. "
        "The application cannot produce a model-based PD."
    )
    st.code(load_error or "Unknown model-loading error")
    st.stop()

st.success("Trained WoE transformer and Logistic Regression model loaded.")

# ------------------------------------------------------------------------------
# Sidebar — application inputs
# ------------------------------------------------------------------------------
st.sidebar.header("Applicant Information")

applicant_age = st.sidebar.number_input(
    "Applicant Age",
    min_value=18,
    max_value=80,
    value=35,
    step=1
)

annual_income = st.sidebar.number_input(
    "Annual Income ($)",
    min_value=10000.0,
    max_value=500000.0,
    value=65000.0,
    step=1000.0
)

emp_length_years = st.sidebar.number_input(
    "Employment Length (Years)",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=1.0
)

st.sidebar.header("Requested Loan Details")

loan_amount = st.sidebar.number_input(
    "Loan Amount ($)",
    min_value=1000.0,
    max_value=50000.0,
    value=15000.0,
    step=500.0
)

term_months = st.sidebar.selectbox(
    "Term (Months)",
    [36, 60],
    index=0
)

interest_rate = st.sidebar.number_input(
    "Interest Rate (%)",
    min_value=5.0,
    max_value=35.0,
    value=12.5,
    step=0.1
)

monthly_debt = st.sidebar.number_input(
    "Monthly Debt Payments ($)",
    min_value=0.0,
    max_value=20000.0,
    value=800.0,
    step=100.0
)

st.sidebar.header("Credit Bureau Profile")

fico_score = st.sidebar.slider(
    "FICO Score Proxy",
    min_value=300,
    max_value=850,
    value=680
)

revol_util = st.sidebar.slider(
    "Revolving Utilization (%)",
    min_value=0.0,
    max_value=200.0,
    value=35.0,
    step=1.0
)

bureau_num_accounts = st.sidebar.number_input(
    "Total Bureau Accounts",
    min_value=0,
    max_value=100,
    value=8,
    step=1
)

bureau_active_accounts = st.sidebar.number_input(
    "Active Bureau Accounts",
    min_value=0,
    max_value=100,
    value=5,
    step=1
)

bureau_total_credit_limit = st.sidebar.number_input(
    "Total Bureau Credit Limit ($)",
    min_value=0.0,
    max_value=1000000.0,
    value=50000.0,
    step=1000.0
)

bureau_total_current_balance = st.sidebar.number_input(
    "Total Current Bureau Balance ($)",
    min_value=0.0,
    max_value=1000000.0,
    value=15000.0,
    step=1000.0
)

bureau_max_months_past_due = st.sidebar.number_input(
    "Maximum Months Past Due",
    min_value=0,
    max_value=24,
    value=0,
    step=1
)

bureau_avg_account_age = st.sidebar.number_input(
    "Average Bureau Account Age (Years)",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.5
)

# ------------------------------------------------------------------------------
# Derived feature
# ------------------------------------------------------------------------------
monthly_income = annual_income / 12.0
dti_engineered = monthly_debt / monthly_income if monthly_income > 0 else 0.0

# ------------------------------------------------------------------------------
# Main screen
# ------------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Annual Income", f"${annual_income:,.0f}")

with col2:
    st.metric("Loan Amount", f"${loan_amount:,.0f}")

with col3:
    st.metric("Engineered DTI", f"{dti_engineered:.1%}")

st.divider()

if st.button("Evaluate Credit Application", type="primary", use_container_width=True):

    input_row = pd.DataFrame([{
        "applicant_age": applicant_age,
        "annual_income": annual_income,
        "emp_length_years": emp_length_years,
        "loan_amount": loan_amount,
        "term_months": term_months,
        "interest_rate": interest_rate,
        "fico_score_proxy": fico_score,
        "dti_engineered": dti_engineered,
        "revol_util_clean": revol_util,
        "bureau_num_accounts": bureau_num_accounts,
        "bureau_active_accounts": bureau_active_accounts,
        "bureau_total_credit_limit": bureau_total_credit_limit,
        "bureau_total_current_balance": bureau_total_current_balance,
        "bureau_max_months_past_due": bureau_max_months_past_due,
        "bureau_avg_account_age": bureau_avg_account_age,
    }])

    try:
        # 1. Apply the saved WoE mappings.
        input_woe = transform_to_woe(input_row, woe_mappings)

        # 2. Generate raw probability using the trained Logistic Regression.
        raw_pd = float(model.predict_proba(input_woe)[:, 1][0])

        # 3. Apply the same prior-shift calibration used during modeling.
        calibrated_pd = calibrate_balanced_probability(raw_pd, y_train)

        # 4. Assign risk tier.
        tier, grade = assign_risk_tier(calibrated_pd)

        # 5. Expected Loss = PD × LGD × EAD.
        lgd = 0.45
        ead = loan_amount
        expected_loss = calibrated_pd * lgd * ead

        # ----------------------------------------------------------------------
        # Results
        # ----------------------------------------------------------------------
        st.subheader("Credit Risk Assessment")

        result1, result2, result3, result4 = st.columns(4)

        result1.metric(
            "Calibrated PD",
            f"{calibrated_pd:.2%}"
        )

        result2.metric(
            "Risk Grade",
            f"Grade {grade}"
        )

        result3.metric(
            "Risk Tier",
            tier
        )

        result4.metric(
            "Expected Loss",
            f"${expected_loss:,.2f}"
        )

        # ----------------------------------------------------------------------
        # Decision guidance
        # ----------------------------------------------------------------------
        st.subheader("Decision Guidance")

        if calibrated_pd > 0.20:
            st.error(
                "High-risk application — route for decline or risk committee review."
            )
        elif calibrated_pd > 0.10:
            st.warning(
                "Elevated-risk application — manual underwriting review recommended."
            )
        else:
            st.success(
                "Lower modeled default risk — eligible for standard underwriting review."
            )

        # ----------------------------------------------------------------------
        # Model transparency
        # ----------------------------------------------------------------------
        with st.expander("Model Details"):
            st.write("**Raw Logistic Regression probability:**", f"{raw_pd:.4%}")
            st.write("**Calibrated PD:**", f"{calibrated_pd:.4%}")
            st.write("**LGD assumption:**", f"{lgd:.0%}")
            st.write("**EAD:**", f"${ead:,.2f}")
            st.write("**Expected Loss:**", f"${expected_loss:,.2f}")

            st.caption(
                "The PD is generated by the saved WoE-transformed Logistic Regression "
                "model and then adjusted using the same class-weight prior-shift "
                "calibration implemented in the modeling notebook."
            )

        with st.expander("Engineered Model Inputs"):
            st.dataframe(
                input_row[FEATURES].T.rename(columns={0: "Value"}),
                use_container_width=True
            )

        with st.expander("WoE-Transformed Inputs"):
            st.dataframe(
                input_woe.T.rename(columns={0: "WoE Value"}),
                use_container_width=True
            )

    except Exception as exc:
        st.error("Model inference failed.")
        st.exception(exc)
