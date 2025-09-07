# CONFIGURATION
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "model_xgboost_v1.joblib"
FEATS_PATH = ROOT / "model" / "feature_order_xgboost_v1.json"
DATA_TEST  = ROOT / "data" / "application_test.csv"
OUT_CSV    = ROOT / "model" / "predictions_test_xgboost_v1.csv"

assert MODEL_PATH.exists(), f"Modèle introuvable : {MODEL_PATH}"
assert FEATS_PATH.exists(), f"Fichier features introuvable : {FEATS_PATH}"
assert DATA_TEST.exists(),  f"Fichier test introuvable : {DATA_TEST}"


# LOAD MODEL + FEATURES
clf = joblib.load(MODEL_PATH)
features = json.loads(FEATS_PATH.read_text())


# LOAD TEST DATA
df = pd.read_csv(DATA_TEST)

# FEATURE ENGINEERING 
df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, pd.NA)
df["AGE_YEARS"]     = (-df["DAYS_BIRTH"] / 365.25).round(1)
df["INCOME"]        = df["AMT_INCOME_TOTAL"].fillna(df["AMT_INCOME_TOTAL"].median())
df["DEBT_RATIO"]    = (df["AMT_ANNUITY"].fillna(0) / (df["INCOME"] + 1e-6)).clip(0, 2.0)
df["TENURE_YEARS"]  = (df["DAYS_EMPLOYED"].abs() / 365.25).astype("float64").fillna(0.0).clip(0, 60)
df["HAS_MORTGAGE"]  = (df["FLAG_OWN_REALTY"].astype(str).str.upper().eq("Y")).astype(int)
df["FAMILY_SIZE"]   = df["CNT_FAM_MEMBERS"].fillna(df["CNT_FAM_MEMBERS"].median())

df["EXT_SOURCE_1"] = df["EXT_SOURCE_1"].fillna(df["EXT_SOURCE_1"].median())
df["EXT_SOURCE_2"] = df["EXT_SOURCE_2"].fillna(df["EXT_SOURCE_2"].median())
df["EXT_SOURCE_3"] = df["EXT_SOURCE_3"].fillna(df["EXT_SOURCE_3"].median())
df["CREDIT_TO_INCOME"] = (df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1e-6)).clip(0, 20)
df["CREDIT_TERM"]      = (df["AMT_CREDIT"] / (df["AMT_ANNUITY"].fillna(0) + 1e-6)).clip(0, 600)
df["DAYS_BIRTH_ABS"]   = (df["DAYS_BIRTH"].abs() / 365.25).round(1)
df["INCOME_LOG"]       = np.log1p(df["AMT_INCOME_TOTAL"])
df["ANNUITY_LOG"]      = np.log1p(df["AMT_ANNUITY"].fillna(0))

X = df[features].copy()

# PREDICT ET EXPORT
p_default = clf.predict_proba(X)[:, 1]
p_accept  = 1.0 - p_default

print("---- Résumé des probabilités d'acceptation (XGBoost v1) ----")
print(pd.Series(p_accept).describe())

pd.DataFrame({
    "SK_ID_CURR": df["SK_ID_CURR"],
    "PROBA_ACCEPTATION": p_accept
}).to_csv(OUT_CSV, index=False)

print(f"\nPrédictions générées > {OUT_CSV}")
