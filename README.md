Horizon Consumer Finance — Point-of-Origination Probability of Default (PD) Engine

An end-to-end Probability of Default (PD) credit risk modeling pipeline and interactive Streamlit risk assessment application built in Python. The project translates raw application and credit bureau data into an interpretable Logistic Regression model using monotonic Weight of Evidence (WoE), calibrated probability estimates, risk grading, and expected loss analysis.

📌 Executive Summary

Horizon Consumer Finance required an auditable, data-driven credit risk engine to replace legacy heuristic underwriting rules. This repository implements the modeling workflow, serialized model artifacts, validation outputs, and a Streamlit application that applies the trained PD model to new credit applications.

Primary Model Architecture: Logistic Regression fit on monotonic Weight of Evidence (WoE) features.

Predictive Power: ROC-AUC of 0.7415 (Gini: 0.4831) and a KS Statistic of 0.3778, outperforming a tuned monotonic XGBoost benchmark.

Probability Calibration: Applied an explicit logit-shift prior adjustment to eliminate sample-weight distortion, reducing the Brier Score by 55.6% (0.2082 $\rightarrow$ 0.0924).

Risk Grading & Expected Loss: Mapped probabilities into a 10-tier Basel risk framework exhibiting strict default monotonicity (1.42% in Tier 1 to 34.12% in Tier 10).

📊 Key Performance Highlights

Metric

Raw Logistic Regression

Calibrated Logistic (Production)

Tuned XGBoost (Benchmark)

Industry Target

ROC-AUC

0.7415

0.7415

0.7384

$> 0.7000$

Gini Coefficient

0.4831

0.4831

0.4768

$> 0.4000$

KS Statistic

0.3778

0.3778

0.3685

$> 0.3000$

Brier Score

0.2082

0.0924

0.0929

$\to 0.0000$

Calibration Status

Uncalibrated (Overestimates PD)

Fully Calibrated

Calibrated

Well-Calibrated

🖥️ Interactive Streamlit Application

The project includes a Streamlit application in src/app.py that applies the trained Logistic Regression PD model to a new credit application.

The application:

Loads the serialized Logistic Regression model from outputs/models/logistic_woe_pd_model.pkl.

Loads the saved WoE mappings from outputs/models/woe_binning_transformer.pkl.

Accepts the same 15 model features used during training, including applicant, loan, FICO, DTI, utilization, and bureau attributes.

Applies the saved WoE mappings before generating the model prediction.

Reconstructs the training split used by the modeling notebook to reproduce the notebook's class-weight probability calibration.

Displays both raw and calibrated PD estimates.

Assigns a risk tier based on the application's PD bands.

Calculates expected loss dynamically using EL = PD × LGD × EAD, with LGD set to 45%.

Displays the engineered DTI value and WoE-transformed model inputs for transparency.

Uses project-root-relative paths so the application can locate model and data artifacts consistently when launched locally or deployed.

The application no longer relies on the earlier heuristic PD calculation. If the trained model artifacts cannot be loaded, the app reports the error rather than silently substituting a heuristic score.

Important: The Streamlit app is a portfolio demonstration of the trained PD modeling workflow. Its risk-tier bands and expected-loss assumptions should not be interpreted as a production lending policy without independent model validation, policy approval, monitoring, and regulatory review.

📁 Repository Architecture

horizon_pd_modeling/
├── data/
│   ├── raw/                             # Input application and bureau datasets
│   └── processed/                       # Model-ready joined dataset & test predictions
├── notebooks/
│   ├── 01_data_audit_and_feature_engineering.ipynb  # Cleaning, multi-source joins, censoring
│   ├── 02_woe_and_pd_modeling.ipynb                 # Monotonic WoE, Logit shift, XGB tuning
│   └── 03_risk_grading_and_expected_loss.ipynb     # Decile bucketing, monotonicity, EL report
├── outputs/
│   ├── figures/                         # Saved evaluation plots (Calibration curves, ROC)
│   ├── models/                          # Serialized model artifacts (.pkl files)
│   └── reports/                         # Validation compliance memo & loss reports
├── src/
│   └── app.py                           # Streamlit app using the trained PD model
├── .gitignore                           # Git exclusion rules
├── README.md                            # Project documentation
└── requirements.txt                     # Python dependencies


⚙️ Methodology & Technical Highlights

1. Data Ingestion & Censoring Control

Right-Censoring Treatment: Unobserved outcomes (Current and Late 31–120 days) were filtered out during model training to eliminate target leakage. Training relies strictly on fully resolved outcomes (Fully Paid vs. Charged Off).

Feature Aggregation: Aggregated multi-trade-line bureau data into applicant-level aggregates prior to loan disbursement.

2. Monotonic Weight of Evidence (WoE) Transformation

Continuous metrics (e.g., FICO, DTI, Income) were discretized into non-linear, monotonic WoE bins using Spearman rank correlation checks. This handles non-linear risk trends while enforcing 100% feature explainability.

3. Logit-Shift Prior Adjustment Calibration

Training with class_weight='balanced' artificially shifts raw probabilities toward 50%. To fix this without altering ROC-AUC or KS rank-ordering, predictions were recalibrated using an exact logit intercept shift:

$$\text{logit}{\text{calibrated}} = \text{logit}{\text{raw}} - \ln\left(\frac{w_1}{w_0}\right)$$

Where $w_1$ and $w_0$ represent the positive and negative class weights used during balanced training.

📈 Risk Grading & Expected Loss Matrix

Expected Loss ($\text{EL}$) is estimated using standard Basel parameters: $\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$ (assuming static $\text{LGD} = 45%$ and $\text{EAD} = $15,000$).

Risk Tier

Loan Count

Observed Default Rate

Avg Predicted PD

Expected Loss / Loan

Tier Total Expected Loss

Tier 1

422

1.42%

3.34%

$225.42

$95,128

Tier 2

422

4.27%

4.22%

$284.89

$120,222

Tier 3

422

4.74%

5.04%

$340.43

$143,661

Tier 4

421

6.41%

6.30%

$424.94

$178,901

Tier 5

422

6.40%

7.92%

$534.27

$225,463

Tier 6

422

8.77%

9.60%

$648.14

$273,515

Tier 7

421

11.88%

11.44%

$772.15

$325,074

Tier 8

422

13.51%

13.91%

$938.91

$396,218

Tier 9

422

22.51%

18.68%

$1,261.02

$532,150

Tier 10

422

34.12%

32.56%

$2,197.68

$927,419

Total / Avg

4,218

11.40%

11.30%

$762.86

$3,217,751

🚀 Quickstart & Setup

1. Clone & Setup Environment

# Clone repository
git clone https://github.com/YOUR_USERNAME/horizon_pd_modeling.git
cd horizon_pd_modeling

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # On macOS/Linux
.\venv\Scripts\Activate.ps1     # On Windows PowerShell

# Install dependencies
pip install -r requirements.txt


2. Execute Modeling Pipeline

Run the notebooks sequentially from the notebooks/ directory:

01_data_audit_and_feature_engineering.ipynb

02_woe_and_pd_modeling.ipynb

03_risk_grading_and_expected_loss.ipynb

3. Launch the Streamlit PD Risk Assessment App Locally

From the project root:

streamlit run src/app.py

The application loads the serialized model and WoE artifacts from outputs/models/ and the processed training dataset from data/processed/.

If Streamlit is not available as a command, use:

python -m streamlit run src/app.py

🤖 Model Artifacts & Inference Flow

The trained artifacts are stored in the repository so the Streamlit application can reproduce the modeling workflow without retraining the model at application startup.

New Application Inputs
        ↓
Feature Engineering (DTI)
        ↓
Saved WoE Mappings
        ↓
Trained Logistic Regression
        ↓
Raw PD
        ↓
Class-Weight Prior Calibration
        ↓
Calibrated PD
        ↓
Risk Tier + Expected Loss

The model uses these 15 features:

applicant_age, annual_income, emp_length_years, loan_amount, term_months, interest_rate, fico_score_proxy, dti_engineered, revol_util_clean, bureau_num_accounts, bureau_active_accounts, bureau_total_credit_limit, bureau_total_current_balance, bureau_max_months_past_due, and bureau_avg_account_age.

🛡️ Governance & Regulatory Compliance

Fair Lending Compliance: Model relies solely on credit-relevant financial attributes, excluding protected characteristics under the Equal Credit Opportunity Act (ECOA).

Interpretability: WoE binning combined with Logistic Regression ensures every decision can be mapped directly to adverse action codes required by the Fair Credit Reporting Act (FCRA).

Model Validation Memo: A formal Model Validation Memo documenting assumptions, limitations, and stress testing considerations is saved at outputs/reports/model_validation_memo.md.