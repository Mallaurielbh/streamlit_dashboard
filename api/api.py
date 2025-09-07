# api/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import os
import json
import joblib
import numpy as np
from typing import Dict, Any

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "model" / "model_xgboost_v1.joblib"
FEATS_PATH = ROOT / "model" / "feature_order_xgboost_v1.json"
MEDIANS_PATH = ROOT / "model" / "feature_medians_xgboost_v1.json"

ELIGIBLE_T   = float(os.getenv("ELIGIBLE_T", "0.55"))
SURVEILLER_T = float(os.getenv("SURVEILLER_T", "0.45"))

DR_RULE   = os.getenv("DR_RULE", "on").lower() in {"on", "true", "1"}
DR_HIGH   = float(os.getenv("DR_HIGH",  "0.50"))
DR_LOW    = float(os.getenv("DR_LOW",   "0.20"))
DR_MALUS  = float(os.getenv("DR_MALUS", "0.10"))
DR_BONUS  = float(os.getenv("DR_BONUS", "0.05"))

_loaded = joblib.load(MODEL_PATH)
if isinstance(_loaded, dict) and "clf" in _loaded and "features" in _loaded:
    clf = _loaded["clf"]
    FEATURE_ORDER = list(_loaded["features"])
else:
    clf = _loaded
    FEATURE_ORDER = json.loads(FEATS_PATH.read_text())

DEFAULTS: Dict[str, float] = {
    "AGE_YEARS": 35.0,
    "INCOME": 36000.0,
    "DEBT_RATIO": 0.30,
    "TENURE_YEARS": 5.0,
    "HAS_MORTGAGE": 0.0,
    "FAMILY_SIZE": 2.0,
    "EXT_SOURCE_1": 0.50,
    "EXT_SOURCE_2": 0.50,
    "EXT_SOURCE_3": 0.50,
    "CREDIT_TO_INCOME": 3.0,
    "CREDIT_TERM": 120.0,
    "DAYS_BIRTH_ABS": 35.0,
    "INCOME_LOG": float(np.log1p(36000.0)),
    "ANNUITY_LOG": float(np.log1p(10000.0)),
}
if MEDIANS_PATH.exists():
    try:
        _med = json.loads(MEDIANS_PATH.read_text())
        for k, v in _med.items():
            if isinstance(v, (int, float)):
                DEFAULTS[k] = float(v)
    except Exception:
        pass

RANGES: Dict[str, tuple] = {
    "AGE_YEARS": (0.0, 120.0),
    "INCOME": (0.0, 1e7),
    "DEBT_RATIO": (0.0, 2.0),
    "TENURE_YEARS": (0.0, 60.0),
    "HAS_MORTGAGE": (0.0, 1.0),
    "FAMILY_SIZE": (1.0, 20.0),
    "EXT_SOURCE_1": (0.0, 1.0),
    "EXT_SOURCE_2": (0.0, 1.0),
    "EXT_SOURCE_3": (0.0, 1.0),
    "CREDIT_TO_INCOME": (0.0, 20.0),
    "CREDIT_TERM": (0.0, 600.0),
    "DAYS_BIRTH_ABS": (0.0, 120.0),
    "INCOME_LOG": (0.0, 20.0),
    "ANNUITY_LOG": (0.0, 20.0),
}


app = FastAPI(title="API d'inférence")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

class ClientData(BaseModel):
    features: Dict[str, Any]


def _coerce_float(x: Any) -> float:
    if x is None:
        return np.nan
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if isinstance(x, str):
        xx = x.strip().upper()
        if xx in {"Y", "YES", "TRUE"}:
            return 1.0
        if xx in {"N", "NO", "FALSE"}:
            return 0.0
        try:
            return float(x)
        except Exception:
            return np.nan
    return np.nan

def _sanitize_features(payload: Dict[str, Any]) -> tuple[Dict[str, float], Dict[str, Any]]:

    clean: Dict[str, float] = {}
    report = {"imputed": [], "clipped": []}

    for feat in FEATURE_ORDER:
        raw_val = payload.get(feat, None)
        val = _coerce_float(raw_val)
        if np.isnan(val):
            val = DEFAULTS.get(feat, 0.0)
            report["imputed"].append(feat)

        lo, hi = RANGES.get(feat, (-np.inf, np.inf))
        clipped = val
        if clipped < lo:
            clipped = lo
            report["clipped"].append(feat)
        elif clipped > hi:
            clipped = hi
            report["clipped"].append(feat)

        clean[feat] = float(clipped)

    return clean, report

def _predict_proba_ordered(values_ordered: Dict[str, float]) -> float:
    X = pd.DataFrame([[values_ordered[k] for k in FEATURE_ORDER]], columns=FEATURE_ORDER)
    p_default = float(clf.predict_proba(X)[0][1])
    return p_default

def _apply_policy(p_accept: float, feats: Dict[str, float]) -> tuple[float, Dict[str, Any]]:

    effects = {"debt_ratio_rule": None}
    if not DR_RULE:
        return p_accept, effects

    ratio = float(feats.get("DEBT_RATIO", 0.0))
    original = p_accept

    if ratio >= DR_HIGH:
        t = min(1.0, (ratio - DR_HIGH) / max(1e-6, (2.0 - DR_HIGH)))
        p_accept = max(0.0, p_accept * (1.0 - DR_MALUS * t))
        effects["debt_ratio_rule"] = {
            "applied": True, "type": "malus",
            "ratio": ratio, "factor": float(1.0 - DR_MALUS * t),
            "delta": float(p_accept - original)
        }
    elif ratio <= DR_LOW:
        t = min(1.0, (DR_LOW - ratio) / max(1e-6, DR_LOW))
        p_accept = min(1.0, p_accept * (1.0 + DR_BONUS * t))
        effects["debt_ratio_rule"] = {
            "applied": True, "type": "bonus",
            "ratio": ratio, "factor": float(1.0 + DR_BONUS * t),
            "delta": float(p_accept - original)
        }
    else:
        effects["debt_ratio_rule"] = {"applied": False, "ratio": ratio}

    return p_accept, effects

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
        "features": FEATURE_ORDER,
        "thresholds": {
            "eligible": ELIGIBLE_T,
            "surveiller": SURVEILLER_T
        }
    }

class ClientData(BaseModel):
    features: Dict[str, Any]

@app.post("/predict")
def predict(data: ClientData):

    feats_in = data.features

    clean_feats, _ = _sanitize_features(feats_in)

    p_default = _predict_proba_ordered(clean_feats)
    p_accept_model = 1.0 - p_default

    p_accept_policy, _ = _apply_policy(p_accept_model, clean_feats)

    p_final = float(min(1.0, max(0.0, p_accept_policy)))
    if p_final >= ELIGIBLE_T:
        decision = "ELIGIBLE"
    elif p_final >= SURVEILLER_T:
        decision = "A REVOIR"
    else:
        decision = "REFUS"

    return {
        "probabilities": {
            "accept_model": p_accept_model,
            "accept_policy": p_final,
            "default_model": p_default
        },
        "decision": decision
    }
