# ============================================================
# SAVE THE TRAINED MODEL
# ============================================================
# Run this cell in your Jupyter Notebook AFTER training is done.
# It saves the final tuned model (best_model from GridSearchCV)
# so the Streamlit app can load it.
# ============================================================

import joblib

# Save the best model found by GridSearchCV
# 'best_model' is the variable name used in your notebook
joblib.dump(best_model, "churn_model.pkl")

print("✅ Model saved successfully as churn_model.pkl")
print(f"   Model type : {type(best_model).__name__}")
print(f"   Best params: {best_model.get_params()}")
