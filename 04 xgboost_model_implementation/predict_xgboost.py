# predict_xgboost.py

import pandas as pd
import joblib

# Cargar modelo
model = joblib.load("xgboost_final_model.pkl")

# Nuevos datos (ejemplo)
new_data = pd.DataFrame(
    {
        "Age": [65],
        "IL-8": [12.3],
        "OPN": [45.1],
        "NSE": [18.2],
        "IL-6": [9.8],
        "Prolactin": [7.4],
        "Omega_Score": [0.62],
        "TGFa": [3.1],
    }
)

# Predicción
pred = model.predict(new_data)[0]
prob_cancer = model.predict_proba(new_data)[0][0]

label = "Cancer" if pred == 0 else "Normal"

print(f"Prediction: {label}")
print(f"Probability of Cancer: {prob_cancer:.3f}")
