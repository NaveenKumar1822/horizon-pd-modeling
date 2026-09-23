# Model Validation & Governance Memo
**To:** Horizon Consumer Finance, Risk & Compliance Committee
**From:** Data Science & Risk Analytics Team
**Subject:** Implementation of Point-of-Origination Probability of Default (PD) Model

## 1. Objective and Scope
This document outlines the methodological decisions, assumptions, and constraints associated with the new Probability of Default (PD) model designed for Horizon Consumer Finance's unsecured personal loan portfolio. The model's primary objective is to estimate default probability at the time of origination to standardize risk-based pricing and support Basel-aligned capital adequacy calculations.

## 2. Data Treatment & Exclusions
* **Right-Censoring Treatment:** Loans with `Current` or `Late (31-120 days)` statuses were explicitly excluded from the training and validation datasets. Including them would introduce target leakage (as their final default state is unobserved). 
* **Observation Point:** All bureau aggregated metrics (e.g., total accounts, max delinquency) and application features (e.g., DTI, income) were calculated strictly using data available prior to loan disbursement.
* **Missing Data:** Missing annual income values (flagged by system sentinels `-1` and `9999999`) were coerced to nulls and managed natively via Weight of Evidence (WoE) missing-value binning.

## 3. Feature Engineering & Interpretability
To comply with Fair Lending and model explainability standards, continuous features were discretized using Weight of Evidence (WoE) binning. This linearizes the relationship between non-linear variables (like FICO and DTI) and the log-odds of default, allowing the use of an interpretable Logistic Regression architecture. 

## 4. Model Selection & Performance
A Logistic Regression model (trained on WoE features) was selected over an XGBoost challenger. While XGBoost demonstrated a marginally higher ROC-AUC, the Logistic Regression model provides exact coefficient transparency required by internal audit and external regulators. 
* The model successfully stratifies risk across a 10-tier grading system.
* Monotonicity is preserved: observed default rates increase strictly in tandem with predicted PD deciles.

## 5. Limitations
The expected loss (EL) calculations assume a static Loss Given Default (LGD) of 45% and an Exposure at Default (EAD) equal to the origination loan amount. Future iterations should incorporate distinct LGD and EAD sub-models to further refine capital reserve estimates.