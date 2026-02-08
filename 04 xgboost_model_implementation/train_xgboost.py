# train_xgboost.py

import pandas as pd
from xgboost import XGBClassifier
import joblib

# =========================
# 1. Cargar datos
# =========================
# df debe contener las 8 features + columna Label
# Ejemplo: df = pd.read_csv("data.csv")

X = df[
    [
        "Age",
        "IL-8",
        "OPN",
        "NSE",
        "IL-6",
        "Prolactin",
        "Omega_Score",
        "TGFa",
    ]
]

# Codificación: Cancer = 0, Normal = 1
y = df["Label"].map({"Cancer": 0, "Normal": 1})

# =========================
# 2. Definir modelo final
# =========================
xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    subsample=1.0,
    colsample_bytree=1.0,
    gamma=0.1,
    min_child_weight=1,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False
)

# =========================
# 3. Entrenamiento final
# =========================
xgb_model.fit(X, y)

# =========================
# 4. Guardar modelo
# =========================
joblib.dump(xgb_model, "xgboost_final_model.pkl")

print("✅ XGBoost final model trained and saved as xgboost_final_model.pkl")
