# IMPORTS
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# 1. CONFIGURATION & PATHS
ROOT = Path(__file__).resolve().parent.parent
DATA_TRAIN = ROOT / "data" / "application_train.csv"
assert DATA_TRAIN.exists(), f"Fichier introuvable : {DATA_TRAIN}"

OUT_DIR = ROOT / "model"
OUT_DIR.mkdir(exist_ok=True)

BASE_FEATURES = ["AGE_YEARS", "INCOME", "DEBT_RATIO", "TENURE_YEARS", "HAS_MORTGAGE", "FAMILY_SIZE"]
ENRICHED_FEATURES = BASE_FEATURES + [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "CREDIT_TO_INCOME", "CREDIT_TERM",
    "DAYS_BIRTH_ABS", "INCOME_LOG", "ANNUITY_LOG"
]


# 2. FEATURE ENGINEERING FUNCTIONS
def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les 6 features de base utilisées dans la LogReg."""
    df = df.copy()
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)
    df["INCOME"] = df["AMT_INCOME_TOTAL"].fillna(df["AMT_INCOME_TOTAL"].median())
    df["DEBT_RATIO"] = (df["AMT_ANNUITY"].fillna(0) / (df["INCOME"] + 1e-6)).clip(0, 2.0)
    df["TENURE_YEARS"] = (df["DAYS_EMPLOYED"].abs() / 365.25).fillna(0).clip(0, 60)
    df["HAS_MORTGAGE"] = (df["FLAG_OWN_REALTY"].astype(str).str.upper().eq("Y")).astype(int)
    df["FAMILY_SIZE"] = df["CNT_FAM_MEMBERS"].fillna(df["CNT_FAM_MEMBERS"].median())
    return df


def add_enriched_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les 14 features complètes utilisées dans XGBoost."""
    df = add_base_features(df)
    df["EXT_SOURCE_1"] = df["EXT_SOURCE_1"].fillna(df["EXT_SOURCE_1"].median())
    df["EXT_SOURCE_2"] = df["EXT_SOURCE_2"].fillna(df["EXT_SOURCE_2"].median())
    df["EXT_SOURCE_3"] = df["EXT_SOURCE_3"].fillna(df["EXT_SOURCE_3"].median())
    df["CREDIT_TO_INCOME"] = (df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1e-6)).clip(0, 20)
    df["CREDIT_TERM"] = (df["AMT_CREDIT"] / (df["AMT_ANNUITY"].fillna(0) + 1e-6)).clip(0, 600)
    df["DAYS_BIRTH_ABS"] = (df["DAYS_BIRTH"].abs() / 365.25).round(1)
    df["INCOME_LOG"] = np.log1p(df["AMT_INCOME_TOTAL"])
    df["ANNUITY_LOG"] = np.log1p(df["AMT_ANNUITY"].fillna(0))
    return df


# 3. CHARGEMENT DES DONNÉES
raw = pd.read_csv(DATA_TRAIN)
y = raw["TARGET"].astype(int)


# 4. LOGISTIC REGRESSION V1
df_base = add_base_features(raw)
X_base = df_base[BASE_FEATURES].copy()

Xtr_b, Xte_b, ytr_b, yte_b = train_test_split(
    X_base, y, test_size=0.20, random_state=42, stratify=y
)

logreg_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("clf", LogisticRegression(solver="liblinear", max_iter=3000, class_weight="balanced"))
])

param_grid = {
    "clf__C": [0.02, 0.1, 0.5, 1.0, 2.0],
    "clf__penalty": ["l1", "l2"],
}

logreg_grid = GridSearchCV(
    estimator=logreg_pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    verbose=0
)

logreg_grid.fit(Xtr_b, ytr_b)
probas_lr = logreg_grid.predict_proba(Xte_b)[:, 1]
auc_lr = roc_auc_score(yte_b, probas_lr)

# Sauvegarde du modèle de référence
joblib.dump(logreg_grid, OUT_DIR / "model_logreg_v1.joblib")
(Path(OUT_DIR / "feature_order_logreg_v1.json")
 .write_text(json.dumps(BASE_FEATURES, ensure_ascii=False, indent=2)))


#            5. XGBOOST V1 - MODELE DÉPLOYÉ - API / DASHBOARD
df_enr = add_enriched_features(raw)
X_enr = df_enr[ENRICHED_FEATURES].copy()

Xtr_x, Xte_x, ytr_x, yte_x = train_test_split(
    X_enr, y, test_size=0.20, random_state=42, stratify=y
)

pos_ratio = ytr_x.mean()
scale_pos_weight = (1 - pos_ratio) / max(pos_ratio, 1e-6)

xgb = XGBClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

xgb.fit(Xtr_x, ytr_x)
probas_xgb = xgb.predict_proba(Xte_x)[:, 1]
auc_xgb = roc_auc_score(yte_x, probas_xgb)

# Sauvegarde du modèle
joblib.dump(xgb, OUT_DIR / "model_xgboost_v1.joblib")
(Path(OUT_DIR / "feature_order_xgboost_v1.json")
 .write_text(json.dumps(ENRICHED_FEATURES, ensure_ascii=False, indent=2)))


# 6. RANDOM FOREST - BENCHMARK
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced_subsample"
)
rf.fit(Xtr_b, ytr_b)
auc_rf = roc_auc_score(yte_b, rf.predict_proba(Xte_b)[:, 1])

# 7. SAUVEGARDE DES MÉTRIQUES
results = {
    "logreg_v1": float(auc_lr),
    "random_forest_baseline": float(auc_rf),
    "xgboost_v1": float(auc_xgb)
}

metrics = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "features_logreg": BASE_FEATURES,
    "features_xgboost_v1": ENRICHED_FEATURES,
    "results_auc": results,
    "deployed_model": "xgboost_v1",
    "artifacts": {
        "logreg_v1": {
            "model": "model_logreg_v1.joblib",
            "features": "feature_order_logreg_v1.json"
        },
        "xgboost_v1": {
            "model": "model_xgboost_v1.joblib",
            "features": "feature_order_xgboost_v1.json"
        }
    }
}

(Path(OUT_DIR / "metrics.json")
 .write_text(json.dumps(metrics, ensure_ascii=False, indent=2)))


# 8. RÉCAPITULATIF CONSOLE

print("\n=== AUC PAR MODÈLE ===")
print(f"- Logistic Regression v1 : {auc_lr:.3f}")
print(f"- Random Forest baseline : {auc_rf:.3f}")
print(f"- XGBoost v1 : {auc_xgb:.3f}")

print(f"\nModèle déployé - xgboost_v1")
print(f"Artefacts générés - {OUT_DIR}")
print(" - model_xgboost_v1.joblib / feature_order_xgboost_v1.json (API/Dashboard)")
print(" - model_logreg_v1.joblib  / feature_order_logreg_v1.json  (Référence)")
print(" - metrics.json")
