# ============================================================
# Customer Churn Prediction - Streamlit App
# Telecom Dataset | XGBoost Model | Epsilon AI Final Project
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Analysis",
    page_icon="📡",
    layout="wide"
)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

# ── Load and preprocess dataset for analysis ─────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("telco_customer_data_v2.csv")

    # --- same cleaning done in the notebook ---
    general_clean_map = {
        'Y': 'Yes', 'True': 'Yes', 'CHURNED': 'Yes', '1': 'Yes',
        'N': 'No', 'False': 'No', 'NO CHURN': 'No', 'not senior': 'No',
        '0': 'No', 'Unknown': np.nan
    }
    cols_to_clean = [
        'Dependents', 'Partner', 'PhoneService', 'PaperlessBilling',
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'SeniorCitizen', 'Churn'
    ]
    df[cols_to_clean] = df[cols_to_clean].replace(general_clean_map)
    df['gender'] = df['gender'].replace({
        'female': 'Female', 'f': 'Female', 'FEMALE': 'Female',
        'male': 'Male', 'm': 'Male', 'Man': 'Male'
    })
    df['PaymentMethod'] = df['PaymentMethod'].replace(
        {'BANK TRANSFER': 'Bank transfer (automatic)'}
    )
    df['Contract'] = df['Contract'].replace({'M-M': 'Month-to-month'})
    df['TotalCharges'] = (
        df['TotalCharges'].astype(str)
        .str.replace('USD', '', regex=False)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # drop rows with no churn label
    df = df.dropna(subset=['Churn'])

    # fill missing values
    cat_cols = [
        'Dependents', 'Partner', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'gender',
        'SeniorCitizen', 'PaymentMethod', 'Contract', 'InternetService'
    ]
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df['MonthlyCharges'].fillna(df['MonthlyCharges'].median(), inplace=True)
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['tenure'].fillna(df['tenure'].median(), inplace=True)

    # remove outliers same as notebook
    df = df[
        (df['tenure'].between(0, 72)) &
        (df['MonthlyCharges'] <= 150) &
        (df['TotalCharges'] <= 10000)
    ]

    # service columns
    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['num_services'] = (df[service_cols] == 'Yes').sum(axis=1)
    df['early_high_charges'] = (
        (df['tenure'] <= 6) &
        (df['MonthlyCharges'] > df['MonthlyCharges'].median())
    ).astype(int)

    return df

# ── Color palette ─────────────────────────────────────────────
RED   = "#E24B4A"
BLUE  = "#4A7FE2"
GRAY  = "#CCCCCC"

# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
model = load_model()

tab1, tab2 = st.tabs(["Analysis", "Predict"])

# ════════════════════════════════════════════════════════════
# TAB 1 — ANALYSIS
# ════════════════════════════════════════════════════════════
with tab1:
    st.title("Customer Churn Analysis")
    st.markdown("Understanding **why** customers leave — and what we can do about it.")

    try:
        df = load_data()
        churned = df[df['Churn'] == 'Yes']
        stayed  = df[df['Churn'] == 'No']

        total        = len(df)
        churn_count  = len(churned)
        churn_rate   = churn_count / total * 100
        mtm_churners = len(churned[churned['Contract'] == 'Month-to-month'])

        # ── KPI row ──────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Customers",         f"{total:,}")
        k2.metric("Churned Customers",        f"{churn_count:,}")
        k3.metric("Overall Churn Rate",       f"{churn_rate:.1f}%")
        k4.metric("Month-to-Month Churners",  f"{mtm_churners:,}")

        st.divider()

        # ── ROW 1 ─────────────────────────────────────────────
        st.subheader("The Two Biggest Risk Factors")
        col1, col2 = st.columns(2)

        with col1:
            # Chart 1 — Churn rate by Contract Type
            contract_churn = (
                df.groupby('Contract')['Churn']
                .apply(lambda x: (x == 'Yes').mean() * 100)
                .reset_index()
                .rename(columns={'Churn': 'ChurnRate'})
                .sort_values('ChurnRate', ascending=False)
            )
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors = [RED if c == 'Month-to-month' else GRAY
                      for c in contract_churn['Contract']]
            bars = ax.bar(contract_churn['Contract'], contract_churn['ChurnRate'],
                          color=colors, width=0.5)
            for bar, val in zip(bars, contract_churn['ChurnRate']):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5, f"{val:.1f}%",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_title("Churn Rate by Contract Type", fontweight='bold')
            ax.set_ylabel("Churn Rate (%)")
            ax.set_ylim(0, contract_churn['ChurnRate'].max() * 1.2)
            ax.tick_params(axis='x', labelsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(
                "Month-to-month customers churn at nearly 3x the rate of one-year "
                "contract holders. Converting customers to annual plans is the "
                "single most effective retention lever."
            )

        with col2:
            # Chart 2 — Churn rate by Internet Service
            internet_churn = (
                df.groupby('InternetService')['Churn']
                .apply(lambda x: (x == 'Yes').mean() * 100)
                .reset_index()
                .rename(columns={'Churn': 'ChurnRate'})
                .sort_values('ChurnRate', ascending=False)
            )
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors = [RED if s == 'Fiber optic' else GRAY
                      for s in internet_churn['InternetService']]
            bars = ax.bar(internet_churn['InternetService'],
                          internet_churn['ChurnRate'],
                          color=colors, width=0.5)
            for bar, val in zip(bars, internet_churn['ChurnRate']):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5, f"{val:.1f}%",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_title("Churn Rate by Internet Service", fontweight='bold')
            ax.set_ylabel("Churn Rate (%)")
            ax.set_ylim(0, internet_churn['ChurnRate'].max() * 1.2)
            ax.tick_params(axis='x', labelsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(
                "Fiber optic — the most expensive service — has the highest churn rate. "
                "This signals a quality or value-perception problem, not a pricing one."
            )

        st.divider()

        # ── ROW 2 ─────────────────────────────────────────────
        st.subheader("Customers Leave Early — or Not at All")
        col3, col4 = st.columns(2)

        with col3:
            # Chart 3 — Tenure distribution
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(stayed['tenure'],  bins=30, color=BLUE, alpha=0.6,
                    label='Stayed',  density=True)
            ax.hist(churned['tenure'], bins=30, color=RED,  alpha=0.6,
                    label='Churned', density=True)
            ax.axvline(x=6, color='black', linestyle='--', linewidth=1,
                       label='6-month mark')
            ax.set_title("Tenure Distribution: Churned vs Stayed", fontweight='bold')
            ax.set_xlabel("Tenure (months)")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(
                "The churn spike is concentrated in the first 6 months. "
                "Customers who survive past month 12 rarely leave — "
                "the first half-year is the critical retention window."
            )

        with col4:
            # Chart 4 — Early high charges churn rate
            early_churn_rate = (
                df.groupby('early_high_charges')['Churn']
                .apply(lambda x: (x == 'Yes').mean() * 100)
                .reset_index()
            )
            early_churn_rate['label'] = early_churn_rate['early_high_charges'].map(
                {0: 'Normal Start', 1: 'High Bill + New Customer'}
            )
            fig, ax = plt.subplots(figsize=(5, 3.5))
            bars = ax.bar(early_churn_rate['label'],
                          early_churn_rate['Churn'],
                          color=[GRAY, RED], width=0.4)
            for bar, val in zip(bars, early_churn_rate['Churn']):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5, f"{val:.1f}%",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_title("Churn Rate: Early High Charges", fontweight='bold')
            ax.set_ylabel("Churn Rate (%)")
            ax.set_ylim(0, early_churn_rate['Churn'].max() * 1.25)
            ax.tick_params(axis='x', labelsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(
                "New customers paying above-median bills in their first 6 months "
                "churn at a significantly higher rate. A high bill before "
                "trust is built is a red flag for departure."
            )

        st.divider()

        # ── ROW 3 ─────────────────────────────────────────────
        st.subheader("Who Churns — and What They Use")
        col5, col6 = st.columns(2)

        with col5:
            # Chart 5 — Services vs churn rate
            service_churn = (
                df.groupby('num_services')['Churn']
                .apply(lambda x: (x == 'Yes').mean() * 100)
                .reset_index()
                .rename(columns={'Churn': 'ChurnRate'})
            )
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(service_churn['num_services'], service_churn['ChurnRate'],
                    marker='o', color=RED, linewidth=2, markersize=7)
            ax.fill_between(service_churn['num_services'],
                            service_churn['ChurnRate'], alpha=0.1, color=RED)
            ax.set_title("Churn Rate by Number of Services Used", fontweight='bold')
            ax.set_xlabel("Number of Add-on Services")
            ax.set_ylabel("Churn Rate (%)")
            ax.set_xticks(service_churn['num_services'])
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(
                "Customers using zero add-on services churn the most. "
                "Every service a customer adopts increases their switching cost — "
                "encourage trial early."
            )

        with col6:
            # Chart 6 — Senior vs churn
            senior_churn = (
                df.groupby('SeniorCitizen')['Churn']
                .apply(lambda x: (x == 'Yes').mean() * 100)
                .reset_index()
                .rename(columns={'Churn': 'ChurnRate'})
            )
            fig, ax = plt.subplots(figsize=(5, 3.5))
            bars = ax.bar(senior_churn['SeniorCitizen'],
                          senior_churn['ChurnRate'],
                          color=[GRAY, RED], width=0.4)
            for bar, val in zip(bars, senior_churn['ChurnRate']):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5, f"{val:.1f}%",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_title("Churn Rate: Senior vs Non-Senior", fontweight='bold')
            ax.set_ylabel("Churn Rate (%)")
            ax.set_ylim(0, senior_churn['ChurnRate'].max() * 1.25)
            ax.set_xticklabels(['Non-Senior', 'Senior'])
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(
                "Senior citizens churn at a notably higher rate. "
                "Targeted retention plans for this segment could reduce this gap."
            )

        st.divider()

        # ── ROW 4 ─────────────────────────────────────────────
        st.subheader("Spending Behavior and Payment Patterns")
        col7, col8 = st.columns(2)

        with col7:
            # Chart 7 — Monthly Charges boxplot
            fig, ax = plt.subplots(figsize=(5, 3.5))
            data_to_plot = [
                stayed['MonthlyCharges'].dropna().values,
                churned['MonthlyCharges'].dropna().values
            ]
            bp = ax.boxplot(data_to_plot, patch_artist=True,
                            labels=['Stayed', 'Churned'], widths=0.4)
            bp['boxes'][0].set_facecolor(BLUE)
            bp['boxes'][1].set_facecolor(RED)
            for patch in bp['boxes']:
                patch.set_alpha(0.7)
            ax.set_title("Monthly Charges: Stayed vs Churned", fontweight='bold')
            ax.set_ylabel("Monthly Charges ($)")
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(
                "Churned customers tend to pay higher monthly charges. "
                "Higher spend without perceived value accelerates the decision to leave."
            )

        with col8:
            # Chart 8 — Payment Method churn rate
            pay_churn = (
                df.groupby('PaymentMethod')['Churn']
                .apply(lambda x: (x == 'Yes').mean() * 100)
                .reset_index()
                .rename(columns={'Churn': 'ChurnRate'})
                .sort_values('ChurnRate', ascending=True)
            )
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors = [RED if 'Electronic' in p else GRAY
                      for p in pay_churn['PaymentMethod']]
            bars = ax.barh(pay_churn['PaymentMethod'],
                           pay_churn['ChurnRate'], color=colors, height=0.5)
            for bar, val in zip(bars, pay_churn['ChurnRate']):
                ax.text(bar.get_width() + 0.3,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va='center', fontsize=8, fontweight='bold')
            ax.set_title("Churn Rate by Payment Method", fontweight='bold')
            ax.set_xlabel("Churn Rate (%)")
            ax.tick_params(axis='y', labelsize=7)
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(
                "Electronic check users have the highest churn rate. "
                "Customers on automatic payments show much stronger retention."
            )

        st.divider()

        # ── ROW 5 ─────────────────────────────────────────────
        st.subheader("Services That Actually Retain Customers")
        col9, col10 = st.columns(2)

        with col9:
            # Chart 9 — Tech Support vs Churn
            tech_churn = (
                df[df['TechSupport'].isin(['Yes', 'No'])]
                .groupby('TechSupport')['Churn']
                .apply(lambda x: (x == 'Yes').mean() * 100)
                .reset_index()
                .rename(columns={'Churn': 'ChurnRate'})
            )
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors = [RED if t == 'No' else BLUE
                      for t in tech_churn['TechSupport']]
            bars = ax.bar(tech_churn['TechSupport'],
                          tech_churn['ChurnRate'], color=colors, width=0.4)
            for bar, val in zip(bars, tech_churn['ChurnRate']):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5, f"{val:.1f}%",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_title("Churn Rate by Tech Support", fontweight='bold')
            ax.set_ylabel("Churn Rate (%)")
            ax.set_ylim(0, tech_churn['ChurnRate'].max() * 1.25)
            ax.set_xticklabels(['No Tech Support', 'Has Tech Support'])
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(
                "Customers without tech support churn at roughly double the rate. "
                "A free trial of tech support for at-risk customers could be a "
                "low-cost, high-impact retention tactic."
            )

        with col10:
            # Chart 10 — Stacked bar contract x churn
            contract_counts = (
                df.groupby(['Contract', 'Churn'])
                .size()
                .unstack(fill_value=0)
            )
            fig, ax = plt.subplots(figsize=(5, 3.5))
            contract_counts.plot(
                kind='bar', stacked=True, ax=ax,
                color=[BLUE, RED], width=0.5
            )
            ax.set_title("Customer Count by Contract & Churn Status", fontweight='bold')
            ax.set_xlabel("")
            ax.set_ylabel("Number of Customers")
            ax.tick_params(axis='x', rotation=15, labelsize=8)
            ax.legend(title='Churn', labels=['No', 'Yes'], fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(
                "Month-to-month contracts dominate the dataset and contribute "
                "the vast majority of churned customers. Shifting even 20% of "
                "these to annual contracts would dramatically reduce total churn."
            )

    except FileNotFoundError:
        st.warning(
            "Dataset file not found. Please make sure "
            "`telco_customer_data_v2.csv` is uploaded to the GitHub repo."
        )

# ════════════════════════════════════════════════════════════
# TAB 2 — PREDICT
# ════════════════════════════════════════════════════════════
with tab2:
    st.title("Churn Prediction")
    st.markdown("Enter customer details below to predict churn risk.")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=72, value=12,
            help="How many months the customer has been with the company"
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, max_value=150.0,
            value=65.0, step=0.5
        )
        contract = st.selectbox(
            "Contract Type",
            options=["Month-to-month", "One year", "Two year"]
        )
        internet_service = st.selectbox(
            "Internet Service",
            options=["Fiber optic", "DSL", "No"]
        )

    with col2:
        tech_support = st.selectbox(
            "Tech Support",
            options=["No", "Yes", "No internet service"]
        )
        senior_citizen = st.selectbox(
            "Senior Citizen", options=["No", "Yes"]
        )

    st.divider()

    def build_features(tenure, monthly_charges, contract,
                       internet_service, tech_support, senior_citizen):
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        MONTHLY_MEDIAN = 65.0
        return pd.DataFrame([{
            'tenure':                    tenure,
            'MonthlyCharges':            monthly_charges,
            'contract_commitment':       contract_map[contract],
            'is_fiber':                  1 if internet_service == "Fiber optic" else 0,
            'has_tech_support':          1 if tech_support == "Yes" else 0,
            'is_senior':                 1 if senior_citizen == "Yes" else 0,
            'early_high_charges':        1 if (tenure <= 6 and monthly_charges > MONTHLY_MEDIAN) else 0,
            'long_tenure_no_commitment': 1 if (tenure > 24 and contract == "Month-to-month") else 0,
        }])

    if st.button("Predict", use_container_width=True, type="primary"):
        input_features = build_features(
            tenure, monthly_charges, contract,
            internet_service, tech_support, senior_citizen
        )
        prediction  = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1]

        st.subheader("Result")
        if prediction == 1:
            st.error("This customer is likely to CHURN.")
        else:
            st.success("This customer is likely to STAY.")

        st.metric("Churn Probability", f"{probability:.1%}")
        st.progress(float(probability))

        with st.expander("Features used by the model"):
            st.dataframe(input_features, use_container_width=True)

    st.divider()
    st.caption("Telecom Churn Prediction | XGBoost | Epsilon AI Final Project")
