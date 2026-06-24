# Customer Churn Prediction — Streamlit Deployment

## Folder Structure

```
churn_app/
│
├── app.py               ← Main Streamlit application
├── churn_model.pkl      ← Saved XGBoost model (generated from notebook)
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## Step 1 — Save the Model (inside your Jupyter Notebook)

Add this cell at the end of your notebook and run it:

```python
import joblib
joblib.dump(best_model, "churn_model.pkl")
print("Model saved!")
```

This creates `churn_model.pkl` in the same folder as your notebook.

---

## Step 2 — Test Locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Step 3 — Deploy to Streamlit Community Cloud

1. Create a GitHub repository and push these files:
   - `app.py`
   - `churn_model.pkl`
   - `requirements.txt`

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Click **New app** → connect your GitHub repo

4. Set **Main file path** to `app.py`

5. Click **Deploy** — done!

---

## Features Used by the Model

| Feature | Description |
|---|---|
| `tenure` | Months the customer has been with the company |
| `MonthlyCharges` | Monthly bill amount |
| `contract_commitment` | Contract type encoded: 0=Month-to-month, 1=One year, 2=Two year |
| `is_fiber` | 1 if Fiber Optic internet, else 0 |
| `has_tech_support` | 1 if Tech Support = Yes, else 0 |
| `is_senior` | 1 if Senior Citizen = Yes, else 0 |
| `early_high_charges` | 1 if new customer (≤6 months) paying above median charges |
| `long_tenure_no_commitment` | 1 if customer stayed >24 months but still on Month-to-month |
