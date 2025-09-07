import unicodedata
from pathlib import Path
import requests
import streamlit as st
import numpy as np
import pandas as pd
import re

import streamlit as st
import streamlit_authenticator as stauth

import streamlit as st
import streamlit_authenticator as stauth

import streamlit as st
import streamlit_authenticator as stauth


# Récupération dynamique des features
def get_feature_order():
    try:
        health_url = "http://127.0.0.1:8000/health"
        j = requests.get(health_url, timeout=5).json()
        feats = j.get("features") or []
        if feats:
            return feats
    except Exception:
        pass

    return [
        "AGE_YEARS","INCOME","DEBT_RATIO","TENURE_YEARS","HAS_MORTGAGE","FAMILY_SIZE",
        "EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3",
        "CREDIT_TO_INCOME","CREDIT_TERM","DAYS_BIRTH_ABS","INCOME_LOG","ANNUITY_LOG"
    ]

FEATURE_ORDER = get_feature_order()


# SECTION - Page config

st.set_page_config(
    page_title="Éligibilité Prêt - Néo-banque",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000/predict"

# SECTION - Styles (CSS)

def load_css():
    for p in (APP_DIR / "assets" / "styles.css", APP_DIR / "styles.css", Path("styles.css")):
        if p.exists():
            st.markdown(f"<style>{p.read_text()}</style>", unsafe_allow_html=True)
            break

load_css()

# SECTION - Constantes & chemins

ROOT_DIR = APP_DIR.parent
DATA_DIR = ROOT_DIR / "data"

@st.cache_data(show_spinner=False)
def load_clients_safe():
    train_path = DATA_DIR / "application_train.csv"
    test_path  = DATA_DIR / "application_test.csv"

    dfs = []
    for p in [train_path, test_path]:
        if p.exists():
            try:
                dfs.append(pd.read_csv(p))
            except Exception as e:
                st.warning(f"Erreur lecture {p.name} → {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    id_col = None
    for c in ["_CLIENT_ID", "CLIENT_ID", "SK_ID_CURR"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        id_col = "_INDEX_ID"
        df[id_col] = df.index

    df["CLIENT_KEY"] = df[id_col].astype(str).str.upper().str.strip()

    df["CLIENT_KEY_NUM"] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")

    s_num = df["CLIENT_KEY_NUM"].astype("Int64").astype(str)
    df["CLIENT_KEY_C3"] = "C" + s_num.str.zfill(3)   # C001
    df["CLIENT_KEY_C4"] = "C" + s_num.str.zfill(4)   # C0001

    df.attrs["id_col"] = id_col
    return df

clients_df = load_clients_safe()

def row_to_filters(row):
    if "AGE_YEARS" in row:
        age = int(_safe(row["AGE_YEARS"], 42))
    elif "DAYS_BIRTH" in row:
        age = int(max(18, round(-_safe(row["DAYS_BIRTH"], 42*365)/365.25)))
    else:
        age = 42

    if "MONTHLY_INCOME" in row:
        revenu_m = float(_safe(row["MONTHLY_INCOME"], 3200))
    elif "AMT_INCOME_TOTAL" in row:
        revenu_m = float(_safe(row["AMT_INCOME_TOTAL"], 3200*12))
    elif "INCOME" in row:
        revenu_m = float(_safe(row["INCOME"], 3200*12))
    else:
        revenu_m = 3200.0

    debt_ratio = float(_safe(row.get("DEBT_RATIO", 0.30)))

    if "TENURE_YEARS" in row:
        tenure = float(_safe(row["TENURE_YEARS"], 5.0))
    elif "DAYS_EMPLOYED" in row:
        tenure = max(0.0, -float(_safe(row["DAYS_EMPLOYED"], -5*365))/365.0)
    else:
        tenure = 5.0

    family = int(_safe(row.get("FAMILY_SIZE", row.get("CNT_FAM_MEMBERS", 2))))

    if "HAS_MORTGAGE" in row:
        mortgage = int(bool(_safe(row["HAS_MORTGAGE"], 0)))
    elif "FLAG_OWN_REALTY" in row:
        mortgage = 1 if str(_safe(row["FLAG_OWN_REALTY"], "N")).upper().startswith("Y") else 0
    else:
        mortgage = 0

    return dict(age=age, revenu=revenu_m, debt_ratio=debt_ratio, tenure=tenure, family=family, mortgage=bool(mortgage))

# SECTION - Sidebar

with st.sidebar:
    st.title("Filtres client")

    input_id = st.text_input(
        "ID client",
        value=st.session_state.get("client_id", "100000")
    ).upper().strip()

    if clients_df.empty:
        st.info("ℹDonnées clients non trouvées. Saisir manuellement les filtres.")
    else:
        if st.button("Charger la fiche"):
            
            df = clients_df.copy()

            if "CLIENT_KEY" not in df.columns:
                id_col = None
                for c in ["_CLIENT_ID", "CLIENT_ID", "SK_ID_CURR"]:
                    if c in df.columns:
                        id_col = c
                        break
                if id_col is None:
                    id_col = df.columns[0]
                df["CLIENT_KEY"] = df[id_col].astype(str).str.upper().str.strip()
            if "CLIENT_KEY_NUM" not in df.columns:
                df["CLIENT_KEY_NUM"] = pd.to_numeric(
                    df["CLIENT_KEY"].str.replace(r"\D", "", regex=True), errors="coerce"
                ).astype("Int64")
            s_num = df["CLIENT_KEY_NUM"].astype("Int64")
            df["CLIENT_KEY_C3"] = "C" + s_num.astype(str).str.zfill(3)
            df["CLIENT_KEY_C4"] = "C" + s_num.astype(str).str.zfill(4)

            norm = input_id
            digits = re.sub(r"\D", "", norm)

            masks = [df["CLIENT_KEY"] == norm]
            if digits:
                try:
                    n = int(digits)
                    masks += [
                        df["CLIENT_KEY_NUM"] == n,
                        df["CLIENT_KEY_C3"] == f"C{n:03d}",
                        df["CLIENT_KEY_C4"] == f"C{n:04d}",
                    ]
                except Exception:
                    pass

            mask = masks[0]
            for m in masks[1:]:
                mask = mask | m
            row = df.loc[mask].head(1)

            if not row.empty:
                r = row.iloc[0]

                def _safe(v, d):
                    try:
                        import math
                        return d if (v is None or (isinstance(v, float) and math.isnan(v))) else v
                    except Exception:
                        return v if v is not None else d

                if "AGE_YEARS" in r:
                    age_val = int(_safe(r["AGE_YEARS"], 42))
                elif "DAYS_BIRTH" in r:
                    age_val = int(max(18, round(-_safe(r["DAYS_BIRTH"], 42*365) / 365.25)))
                else:
                    age_val = 42

                if "MONTHLY_INCOME" in r:
                    rev_val = float(_safe(r["MONTHLY_INCOME"], 3200))
                elif "AMT_INCOME_TOTAL" in r:
                    rev_val = float(_safe(r["AMT_INCOME_TOTAL"], 3200*12)) / 12.0
                elif "INCOME" in r:
                    rev_val = float(_safe(r["INCOME"], 3200*12)) / 12.0
                else:
                    rev_val = 3200.0

                debt_val = float(_safe(r.get("DEBT_RATIO", 0.30), 0.30))
                

                if "TENURE_YEARS" in r:
                    ten_val = float(_safe(r["TENURE_YEARS"], 5.0))
                elif "DAYS_EMPLOYED" in r:
                    ten_val = max(0.0, -float(_safe(r["DAYS_EMPLOYED"], -5*365)) / 365.0)
                else:
                    ten_val = 5.0

                fam_val = int(_safe(r.get("FAMILY_SIZE", r.get("CNT_FAM_MEMBERS", 2)), 2))

                if "HAS_MORTGAGE" in r:
                    mort_val = bool(int(_safe(r["HAS_MORTGAGE"], 0)))
                elif "FLAG_OWN_REALTY" in r:
                    mort_val = str(_safe(r["FLAG_OWN_REALTY"], "N")).upper().startswith("Y")
                else:
                    mort_val = True

                

                st.session_state.update(
                    age=age_val,
                    revenu_mensuel=int(rev_val),
                    debt_ratio=float(debt_val),
                    tenure=float(ten_val),
                    family_size=int(fam_val),
                    has_mortgage=bool(mort_val),
                    client_id=input_id,
                )
                st.rerun()
            else:
                st.warning("ID introuvable en base de données.")

    debt_ratio = st.slider(
    "Taux d'endettement",
    min_value=0.0,
    max_value=1.5,
    step=0.01,
    key="debt_ratio",
    help="Proportion des revenus mensuels consacrés au remboursement"
)
    age = st.number_input(
    "Âge emprunteur principal",
    min_value=18,
    max_value=70,
    key="age",
)
    revenu_mensuel = st.number_input(
    "Revenus mensuel du foyer",
    min_value=0,
    max_value=20000,
    step=100,
    key="revenu_mensuel",
    help="Valeur exprimée en euros"
)
    tenure = st.number_input(
    "Ancienneté de l'emploi",
    min_value=0.0,
    max_value=60.0,
    step=0.5,
    key="tenure",
    help="Valeur exprimée en années"
)
    family_size = st.number_input(
    "Taille du foyer",
    min_value=1,
    max_value=10,
    key="family_size",
)
    segment = st.selectbox(
    "Segment",
    ["Particulier", "Professionnel"],
    index=0,
    help="Type de client"
)
    has_mortgage = st.checkbox(
    "Emprunt parallèle",
    key="has_mortgage",
)
    
    st.divider()
    st.subheader("Ré-entraînement IA")

    consent_training = st.checkbox(
        "Autoriser l'utilisation du profil",
        value=st.session_state.get("consent_training", False),
        key="consent_training",
        help="Optionnel. Aucune donnée nominative n’est utilisée."
    )

    
    client_id = st.session_state.get("client_id", input_id)

def find_client_row(df: pd.DataFrame, raw_id: str):
    if df.empty:
        return None, "no_df"

    norm = (raw_id or "").upper().strip()
    digits = re.sub(r"\D", "", norm)

    candidates = []

    candidates.append(df.loc[df["CLIENT_KEY"] == norm])

    if digits:
        try:
            n = int(digits)
            candidates.append(df.loc[df["CLIENT_KEY_NUM"] == n])
            c3 = f"C{n:03d}"
            c4 = f"C{n:04d}"
            candidates.append(df.loc[df["CLIENT_KEY_C3"] == c3])
            candidates.append(df.loc[df["CLIENT_KEY_C4"] == c4])
        except Exception:
            pass

    for cand in candidates:
        if cand is not None and not cand.empty:
            return cand.iloc[0], "ok"

    return None, "not_found"

def build_features():
    age_years    = float(age)
    income_year  = float(revenu_mensuel) * 12.0
    debt_ratio_v = float(debt_ratio)
    tenure_years = float(tenure)
    has_mort     = int(has_mortgage)
    family_v     = float(family_size)

    ext1_med = ext2_med = ext3_med = 0.5
    credit_med  = 100000.0
    annuity_med = income_year * 0.20 if income_year > 0 else 10000.0

    try:
        if not clients_df.empty:
            if "EXT_SOURCE_1" in clients_df.columns:
                ext1_med = float(clients_df["EXT_SOURCE_1"].median(skipna=True))
            if "EXT_SOURCE_2" in clients_df.columns:
                ext2_med = float(clients_df["EXT_SOURCE_2"].median(skipna=True))
            if "EXT_SOURCE_3" in clients_df.columns:
                ext3_med = float(clients_df["EXT_SOURCE_3"].median(skipna=True))
            if "AMT_CREDIT" in clients_df.columns:
                credit_med = float(clients_df["AMT_CREDIT"].median(skipna=True))
            if "AMT_ANNUITY" in clients_df.columns:
                a = clients_df["AMT_ANNUITY"].dropna()
                if len(a):
                    annuity_med = float(a.median())
    except Exception:
        pass

    ext1 = ext1_med
    ext2 = ext2_med
    ext3 = ext3_med
    credit_to_income = float(np.clip(credit_med / (income_year + 1e-6), 0, 20))
    credit_term      = float(np.clip(credit_med / (annuity_med + 1e-6), 0, 600))
    days_birth_abs   = float(round(age_years, 1))  # proxy OK
    income_log       = float(np.log1p(income_year))
    annuity_log      = float(np.log1p(annuity_med))

    full = {
        "AGE_YEARS": age_years,
        "INCOME": income_year,
        "DEBT_RATIO": debt_ratio_v,
        "TENURE_YEARS": tenure_years,
        "HAS_MORTGAGE": has_mort,
        "FAMILY_SIZE": family_v,
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "CREDIT_TO_INCOME": credit_to_income,
        "CREDIT_TERM": credit_term,
        "DAYS_BIRTH_ABS": days_birth_abs,
        "INCOME_LOG": income_log,
        "ANNUITY_LOG": annuity_log,
        "_SEGMENT": segment,
        "_CLIENT_ID": client_id,
    }
    ordered = {k: full.get(k, 0.0) for k in FEATURE_ORDER}
    ordered["_SEGMENT"] = segment
    ordered["_CLIENT_ID"] = client_id
    return ordered


def decision_from_score(p: float, eligible_t: float, REVOIR_t: float) -> str:
    if p >= eligible_t:
        return "ELIGIBLE"
    elif p >= REVOIR_t:
        return "A REVOIR"
    return "REFUS"

def call_api(features: dict):

    payload = {"features": {k: v for k, v in features.items() if not k.startswith("_")}}
    try:
        r = requests.post(API_URL, json=payload, timeout=8)
        r.raise_for_status()
        data = r.json()

        probs = data.get("probabilities", {}) or {}
        p_acc_model = float(probs.get("accept_model", 0.0))
        p_acc_final = float(probs.get("accept_policy", p_acc_model))

        decision = data.get("decision", "INCONNU")
        reasons = data.get("reasons", [])

        return p_acc_model, p_acc_final, decision, reasons

    except Exception as e:
        st.error(f"API indisponible : {e}")
        return 0.0, 0.0, "INCONNU", []

def get_thresholds_from_api():
    try:
        health_url = API_URL.replace("/predict", "/health")
        r = requests.get(health_url, timeout=5)
        r.raise_for_status()
        j = r.json()
        th = j.get("thresholds", {})
        eligible_t = float(th.get("eligible", 0.50))
        REVOIR_t  = float(th.get("surveiller", 0.40))
        eligible_t = max(0.0, min(1.0, eligible_t))
        REVOIR_t  = max(0.0, min(eligible_t, REVOIR_t))
        return eligible_t, REVOIR_t
    except Exception:
        return 0.50, 0.40

def compute_decision(p_acc: float, decision_ui: str, eligible_t: float, REVOIR_t: float):

    norm = (decision_ui or "").upper()
    if norm not in {"ELIGIBLE", "A REVOIR", "REFUS"}:
        if p_acc >= eligible_t:
            norm = "ELIGIBLE"
        elif p_acc >= REVOIR_t:
            norm = "A REVOIR"
        else:
            norm = "REFUS"
    if norm == "ELIGIBLE":
        return ("Dossier validé", "#129ac0",
                "Le dossier est conforme. Vous pouvez lancer la contractualisation.", norm)
    if norm == "A REVOIR":
        return ("Dossier validé sous conditions", "#1933aa",
                "Dossier sous conditions. Veuillez ajuster certains points avant contractualisation.", norm)
    return ("Dossier non finançable", "#aa0766",
            "Dossier ne répondant pas aux critères en vigeur. Veuillez proposer des alternatives.", norm)

# SECTION — UI — (Constat / Explication / Conseil)

def conseils_pour_client(revenu_mensuel, debt_ratio, tenure, has_mortgage, age_years):
    conseils = []

    # Taux d'endettement
    if debt_ratio >= 0.40:
        conseils.append(
            "Constat : Taux d’endettement très élevé (>40%).<br>"
            "Explication : Ce niveau d’endettement limite fortement la capacité d’emprunt et pèse sur le score.<br>"
            "Conseil : Proposer d’augmenter l’apport, de réduire le montant demandé ou d’allonger la durée après validation.<br>"
        )
    elif debt_ratio >= 0.33:
        conseils.append(
            "Constat : Taux d’endettement supérieur à 33%.<br>"
            "Explication : La limite de référence est dépassée.<br>"
            "Conseil : Recommander de revoir le montant emprunté, d’augmenter l’apport ou d’envisager un allègement des charges.<br>"
        )
    else:
        conseils.append(
            "Constat : Taux d’endettement maîtrisé.<br>"
            "Explication : Aucun impact négatif sur le score.<br>"
            "Conseil : Encourager le client à conserver cet équilibre.<br>"
        )

    # Ancienneté professionnelle
    if tenure < 2:
        conseils.append(
            f"Constat : Ancienneté professionnelle limitée.<br>"
            "Explication : Cela peut légèrement impacter la solidité perçue du dossier.<br>"
            "Conseil : Suggérer au client de fournir des documents prouvant la continuité de son emploi, ou d'attendre quelques mois.<br>"
        )

    # Revenus mensuels
    if revenu_mensuel < 2000:
        conseils.append(
            "Constat : Revenus mensuels limités.<br>"
            "Explication : Des revenus limités réduisent la capacité d’emprunt et peuvent pénaliser le score.<br>"
            "Conseil : Proposer d’ajouter un co-emprunteur, d’augmenter l’apport ou d’allonger la durée de remboursement.<br>"
        )
    elif revenu_mensuel < 3000 and debt_ratio >= 0.33:
        conseils.append(
            "Constat : Revenus corrects mais fragilisés par un endettement élevé.<br>"
            "Explication : La combinaison revenu/endettement impacte négativement le score.<br>"
            "Conseil : Recommander de réduire le montant demandé ou de renforcer l’apport pour améliorer l’évaluation.<br>"
        )

    # Crédits en cours
    if has_mortgage:
        conseils.append(
            "Constat : Présence de crédits en cours.<br>"
            "Explication : Les crédits existants pèsent sur la capacité d’endettement et donc sur le score.<br>"
            "Conseil : Proposer une renégociation des taux ou un regroupement des prêts pour alléger les mensualités.<br>"
        )

    return conseils


# SECTION - Features (modèle)

features = build_features()
ELIGIBLE_T, REVOIR_T = get_thresholds_from_api()
p_model, p_final, decision_api, reasons = call_api(features)
decision_ui = decision_from_score(p_final, ELIGIBLE_T, REVOIR_T)

# SECTION - Seuils & décision

decision_txt, decision_color, decision_help, decision_upper = compute_decision(
    p_final, decision_ui, ELIGIBLE_T, REVOIR_T
)
elig_pct = int(round(p_final * 100))



tips_client = conseils_pour_client(
    revenu_mensuel,
    debt_ratio,
    tenure,
    has_mortgage,
    int(features["AGE_YEARS"])
)

st.markdown(
    "\n".join([
        '<div class="topbar light">',
        '  <div class="top-left">',
        '    <div class="h1">Dashboard Éligibilité prêt</div>',
        '    <div class="subtle">Vue conseiller - modules, score & fiche client</div>',
        '  </div>',
        f'  <div class="fiche">Fiche client <b>{features["_CLIENT_ID"]}</b></div>',
        '</div>',
    ]),
    unsafe_allow_html=True,
)

# SECTION - UI - KPIs

col_score, col_decision = st.columns([0.70, 0.30])

with col_score:
    st.markdown("\n".join([
        '<div class="card kpi-card">',
        '  <div class="title">Éligibilité au prêt</div>',
        '  <div class="kpi-body">',
        '    <div class="kpi-progress"><div class="kpi-progress-bar" style="width:' + str(elig_pct) + '%"></div></div>',
        f'    <div class="kpi-value">{elig_pct}<span class="unit">%</span></div>',
        '  </div>',
        '  <div class="kpi-sub" style="font-size:0.85rem; margin-top:0.5rem; color:#64748b;">',
        f'     <b>> {int(round(ELIGIBLE_T*100))}% </b> - Dossier validé  |  '
        f'     <b>Entre {int(round(REVOIR_T*100))}% et {int(round(ELIGIBLE_T*100))}%</b> - Accepté avec conditions  |  '
        f'     <b>< {int(round(REVOIR_T*100))}%</b> - Non finançable',
        '  </div>',
        '</div>',
    ]), unsafe_allow_html=True)

with col_decision:
    st.markdown("\n".join([
        '<div class="card decision-card">',
        '  <div class="title" style="margin-bottom:.5rem;">Décision</div>',
        f'  <span style="display:inline-block;background-color:{decision_color}1A;color:{decision_color};font-weight:700;font-size:.95rem;padding:4px 10px;border-radius:6px;margin-bottom:.5rem;">{decision_txt}</span>',
        f'  <div style="font-size:.95rem; color:#374151;">{decision_help}</div>',
        '</div>',
    ]), unsafe_allow_html=True)

col_left, col_right = st.columns([0.35, 0.65])

with col_left:
    st.markdown("\n".join([
        '<div class="card profil-card">',
        f'  <div class="title" style="margin-bottom:1rem;">Profil client</div>',
        '  <div class="kv">',
        f'    <div class="k">Âge :</div><div class="v">{int(features["AGE_YEARS"])} ans</div>',
        f'    <div class="k">Revenu mensuel :</div><div class="v">{int(revenu_mensuel):,} €</div>',
        f'    <div class="k">Taux d\'endettement :</div><div class="v">{debt_ratio*100:.1f} %</div>',
        f'    <div class="k">Taille du foyer :</div><div class="v">{int(family_size)}</div>',
        f'    <div class="k">Emprunt parallèle :</div><div class="v">{"Oui" if has_mortgage else "Non"}</div>',
        '  </div>',
        '</div>',
    ]), unsafe_allow_html=True)

with col_right:
    tips_li = "\n".join([f"<li style='margin-bottom:.4rem;'> {t}</li>" for t in tips_client])
    st.markdown("\n".join([
        '<div class="card conseil-card">',
        '  <div class="title" style="margin-bottom:1rem;">Conseils pour améliorer le dossier</div>',
        '  <p style="font-size:0.95rem; color:#374151; margin-bottom:.5rem;">Quelques actions concrètes à réaliser ou communiquer au client afin de l’aider dans l’amélioration de son dossier :</p>',
        f'  <ul style="margin-top:.25rem; padding-left:1.2rem; line-height:1.6;">{tips_li}</ul>',
        '</div>',
    ]), unsafe_allow_html=True)

st.markdown("""
<div class="rgpd-banner">
  <b>Transparence RGPD</b> - Ce tableau respecte la confidentialité des données clients.
  Les identifiants sont anonymisés et aucune donnée identifiable n'est stockée.
  Les données sont utilisées uniquement pour calculer un score d’éligibilité, et conservées pendant 6 mois maximum à compter de leur traitement. <br>
  <b>Conseillers :</b> en cas de demande client concernant une suppression ou rectification de données,
  merci de transmettre l'adresse suivante :
  <a href="mailto:contact@neobank.com"><b>contact@neobank.com</b></a>.
</div>
""", unsafe_allow_html=True)