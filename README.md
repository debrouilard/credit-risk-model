# This project involves building an end-to-end Credit Scoring Model for Bati Bank in partnership with an eCommerce platform. The goal is to enable a "Buy-Now-Pay-Later" service by predicting credit risk using behavioral transaction data.

#📌 Project Overview
As an Analytics Engineer at Bati Bank, I am tasked with developing a system that:
1. Defines a **proxy variable** for credit risk using RFM (Recency, Frequency, Monetary) analysis.
2. Develops a model to assign **risk probabilities** and **credit scores**.
3. Predicts optimal loan amounts and durations.
4. Deploys the model as a containerized **FastAPI** service.

---

## 🏦 Credit Scoring Business Understanding

### (a) Basel II Accord & Model Interpretability
The **Basel II Capital Accord** sets global regulatory expectations for risk measurement. In a regulated financial context, Bati Bank must ensure that credit models are **interpretable and well-documented**. 
* **Regulatory Compliance:** Banks must explain *why* a customer was denied credit to ensure fairness and transparency.
* **Risk Management:** Documentation allows the bank to justify its risk weightings to regulators, directly impacting the capital reserves the bank is legally required to hold.

### (b) Proxy Variables & Business Risks
The provided Xente dataset contains transaction records but lacks an explicit "default" label. To build a predictive model, a **proxy variable** (labeled `is_high_risk`) is engineered using customer behavioral patterns (RFM clustering).
* **Why Necessary:** Supervised learning requires a target variable to train the model to distinguish between "good" and "bad" borrowers.
* **Business Risks:** 
    * **Label Noise:** A customer might be labeled "high risk" simply for being inactive, even if they are financially stable.
    * **Generalization Risk:** The proxy might not perfectly capture the actual likelihood of a loan default, leading to potential loss of revenue or increased credit risk.

### (c) Model Trade-offs: Interpretability vs. Performance
* **Simple/Interpretable Models (e.g., Logistic Regression with WoE):**
    * *Pros:* Transparent, easily audited by regulators, and follows traditional banking standards.
    * *Cons:* May struggle to capture complex, non-linear relationships in alternative eCommerce data.
* **High-Performance Models (e.g., Gradient Boosting):**
    * *Pros:* Higher predictive accuracy ($AUC/F1$), leading to better risk differentiation.
    * *Cons:* Often viewed as "black-box" models, making them harder to explain to risk committees and more difficult to document for Basel II compliance.

---

## 📊 Exploratory Data Analysis (EDA) Insights
Based on the analysis conducted in `notebooks/eda.ipynb`:
* **Extreme Class Imbalance:** Fraudulent and high-risk transactions are a small minority, requiring specialized sampling techniques.
* **Skewed Monetary Features:** Transaction `Amount` and `Value` are highly right-skewed; normalization/standardization is required for model stability.
* **Channel Distribution:** Specific `ChannelId` values show higher average transaction volumes, indicating that the source of the transaction is a strong risk indicator.

---
