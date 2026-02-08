# predict_xgboost.py

import pandas as pd
import joblib

# Cargar modelo
model = joblib.load("xgboost_final_model.pkl")

# Nuevos datos (ejemplo)
new_data = { 
    "Age": float(input("Edad: ")), 
    "IL-8": float(input("IL-8: ")), 
    "OPN": float(input("OPN: ")), 
    "NSE": float(input("NSE: ")), 
    "IL-6": float(input("IL-6: ")), 
    "Prolactin": float(input("Prolactina: ")), 
    "Omega_Score": float(input("Omega Score: ")), 
    "TGFa": float(input("TGFa: ")), 
}

# Predicción
pred = model.predict(new_data)[0]
prob_cancer = model.predict_proba(new_data)[0][0]

label = "Cancer" if pred == 0 else "Normal"

print(f"Prediction: {label}")
print(f"Probability of Cancer: {prob_cancer:.3f}")
