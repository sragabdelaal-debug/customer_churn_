# ============================================================
# Customer Churn Prediction - Streamlit App
# ============================================================
# This app loads a trained XGBoost model and predicts
# whether a telecom customer is likely to churn or not.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page configuration ──────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📡",
    layout="centered"
)

# ── Load the saved model ────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained XGBoost model from disk."""
    model = joblib.load("churn_model.pkl")
    return model

model = load_model()

# ── App title & description ─────────────────────────────────
st.title("📡 Customer Churn Prediction")
st.markdown(
    "Enter the customer's information below, then click **Predict** "
    "to find out if they are likely to churn."
)
st.divider()

# ── Input section ────────────────────────────────────────────
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    # How long the customer has been with the company (months)
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0, max_value=72, value=12,
        help="How many months the customer has been with the company"
    )

    # How much the customer pays each month
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0, max_value=150.0, value=65.0, step=0.5
    )

with col2:
    # Type of contract the customer has
    contract = st.selectbox(
        "Contract Type",
        options=["Month-to-month", "One year", "Two year"]
    )

    # Whether the customer uses Fiber Optic internet
    internet_service = st.selectbox(
        "Internet Service",
        options=["Fiber optic", "DSL", "No"]
    )

col3, col4 = st.columns(2)

with col3:
    # Whether the customer has tech support
    tech_support = st.selectbox(
        "Tech Support",
        options=["No", "Yes", "No internet service"]
    )

with col4:
    # Whether the customer is a senior citizen
    senior_citizen = st.selectbox(
        "Senior Citizen",
        options=["No", "Yes"]
    )

st.divider()

# ── Feature Engineering ──────────────────────────────────────
# We must apply the EXACT same transformations used during training.

def build_features(tenure, monthly_charges, contract,
                   internet_service, tech_support, senior_citizen):
    """
    Recreate the engineered features used to train the model.
    The model was trained on features_v2:
      ['tenure', 'MonthlyCharges', 'contract_commitment',
       'is_fiber', 'has_tech_support', 'is_senior',
       'early_high_charges', 'long_tenure_no_commitment']
    """

    # 1. Contract encoded as ordered number (same map as training)
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    contract_commitment = contract_map[contract]

    # 2. Fiber optic flag (higher churn risk)
    is_fiber = 1 if internet_service == "Fiber optic" else 0

    # 3. Tech support flag
    has_tech_support = 1 if tech_support == "Yes" else 0

    # 4. Senior citizen flag
    is_senior = 1 if senior_citizen == "Yes" else 0

    # 5. New customer paying more than the training-set median ($65)
    #    This was computed at training time; we use the same threshold.
    MONTHLY_MEDIAN = 65.0
    early_high_charges = 1 if (tenure <= 6 and monthly_charges > MONTHLY_MEDIAN) else 0

    # 6. Customer has stayed long but still has no commitment (Month-to-month)
    long_tenure_no_commitment = 1 if (tenure > 24 and contract == "Month-to-month") else 0

    # Assemble into a DataFrame matching training column order
    features = pd.DataFrame([{
        'tenure':                    tenure,
        'MonthlyCharges':            monthly_charges,
        'contract_commitment':       contract_commitment,
        'is_fiber':                  is_fiber,
        'has_tech_support':          has_tech_support,
        'is_senior':                 is_senior,
        'early_high_charges':        early_high_charges,
        'long_tenure_no_commitment': long_tenure_no_commitment,
    }])

    return features

# ── Predict button ────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):

    # Build the feature row
    input_features = build_features(
        tenure, monthly_charges, contract,
        internet_service, tech_support, senior_citizen
    )

    # Get prediction and probability
    prediction   = model.predict(input_features)[0]          # 0 = No Churn, 1 = Churn
    probability  = model.predict_proba(input_features)[0][1] # Probability of churn

    st.divider()
    st.subheader("Prediction Result")

    # Show result
    if prediction == 1:
        st.error("⚠️  This customer is likely to **CHURN**.")
    else:
        st.success("✅  This customer is likely to **STAY**.")

    # Show churn probability as a percentage
    st.metric(
        label="Churn Probability",
        value=f"{probability:.1%}"
    )

    # Show a progress bar for visual clarity
    st.progress(float(probability))

    # Show a brief explanation of the engineered features used
    with st.expander("🔎 Features sent to the model"):
        st.dataframe(input_features, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("Customer Churn Prediction · Telecom Dataset · XGBoost Model")
