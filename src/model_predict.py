import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import FunctionTransformer

from data_preprocess import add_missing_indicator, additional_loan_features, validate_dataframe, clean_data, encode_categoricals

TOP_STATES = [
    "CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "NC",
    "VA", "MA", "MD", "AZ", "WA", "CO", "MN", "MO", "IN", "MI",
]

VALID_PURPOSES = {
    "debt_consolidation", "credit_card", "home_improvement",
    "major_purchase", "small_business", "car", "medical",
    "moving", "vacation", "house", "wedding",
    "renewable_energy", "educational",
}
 
VALID_HOME_OWNERSHIP = {"RENT", "OWN", "MORTGAGE", "OTHER"}

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
    "8 years": 8, "9 years": 9, "10+ years": 10,
}

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "production")
DECISION_THRESHOLDS = 0.199

cache = {}

def _patch_preprocessor(prep):
    # sklearn ≥1.4 no longer handles a bare 'passthrough' string stored in
    # transformers_ by older pickles — replace it with an identity transformer.
    if not hasattr(prep, 'transformers_'):
        return prep
    prep.transformers_ = [
        (name, FunctionTransformer(validate=False) if isinstance(trans, str) else trans, cols)
        for name, trans, cols in prep.transformers_
    ]
    return prep


def load_pickles(name):
    if name not in cache:
        path = os.path.join(MODEL_DIR, name)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if hasattr(obj, 'transformers_'):
            obj = _patch_preprocessor(obj)
        cache[name] = obj
    return cache[name]

def parse_emp_length(raw):
    """
    Parse employment length into years
    """
    if raw is None:
        return np.nan
    raw = raw.strip().lower()
    for label, num in EMP_LENGTH_MAP.items():
        if label.lower() == raw:
            return num
    digits = "".join(c for c in raw if c.isdigit())
    return min(int(digits), 10) if digits else np.nan


def build_feature_row(features, medians):
    """
    Build a feature row for prediction
    """
    
    # Mandatory Features
    fico = int(features["fico_score"])
    loan = float(features["loan_amount"])
    revenue = float(features["revenue"])

    # Recommended Features
    indebtedness = features.get("indebtedness")
    indebtedness = float(indebtedness) if indebtedness is not None else np.nan

    emp = parse_emp_length(features.get("emp_length"))

    if np.isnan(indebtedness):
        indebtedness = medians.get("indebtedness", 13.5)
    if np.isnan(emp):
        emp = medians.get("emp_length", 5.0)

    purpose = features.get("purpose")
    if purpose not in VALID_PURPOSES:
        purpose = "UnKnown"
    
    home = features.get("home_ownership", "")
    home = home.upper() if home.upper() in VALID_HOME_OWNERSHIP else "UnKnown"

    # Optional Features
    state = features.get("state", "")
    state = state.upper() if state.upper() in TOP_STATES else "Other"

    has_exp = features.get("has_experience")
    has_exp = int(bool(has_exp)) if has_exp is not None else "Unknown"

    now = datetime.now()
    issue_year = now.year
    issue_month = now.month
    issue_quarter = (now.month - 1) // 3 + 1
    post_crisis_recovery = 0

    data = {
        "revenue": revenue,
        "indebtedness": indebtedness,
        "loan_amount": loan,
        "fico_score": fico,
        "has_experience": has_exp,
        "emp_length": emp,
        "purpose": purpose,
        "home_ownership": home,
        "state": state,
        "issue_year": issue_year, 
        "issue_month": issue_month,
        "issue_quarter": issue_quarter,
        "post_crisis_recovery": post_crisis_recovery 
    }

    df = pd.DataFrame([data])

    df = additional_loan_features(df)
    df = add_missing_indicator(df, ['indebtedness', 'emp_length'])
    df = clean_data(df, ['indebtedness', 'emp_length'], ['purpose', 'home_ownership', 'state', 'has_experience'])
    df = encode_categoricals(df, ['purpose', 'home_ownership', 'state', 'has_experience'])
    df["indebtedness"] = df["indebtedness"].fillna(0.0)
    df["emp_length"] = df["emp_length"].fillna(0).astype(int)

    return df

def align_columns(df, training_columns):
    """
    Make sure the DataFrame has exactly the same columns as training
    """
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
 
    # Keep only training columns, in the same order
    df = df[training_columns]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df
    

def predict_default_probability(features, model=None, preprocessor=None, training_columns=None, threshold=DECISION_THRESHOLDS, medians=None):
    if model is None:
        model = load_pickles("model.pkl")
    if preprocessor is None:
        preprocessor = load_pickles("preprocessor.pkl")
    if training_columns is None:
        training_columns = load_pickles("columns.pkl")
    if medians is None:
        medians = load_pickles("medians.pkl")
    
    df = build_feature_row(features, medians=medians)
    
    imputed = []
    if df["indebtedness_missing"].iloc[0] == 1:
        imputed.append("indebtedness")
    if df["emp_length_missing"].iloc[0] == 1:
        imputed.append("emp_length")
    if features.get("purpose") is None:
        imputed.append("purpose")
    if features.get("home_ownership") is None:
        imputed.append("home_ownership")

    df = align_columns(df, training_columns)
    df_scaled = preprocessor.transform(df)
    
    proba = float(model.predict_proba(df_scaled)[:, 1][0])
    
    # Classify risk
    if proba < 0.15:
        risk = "Low"
    elif proba < 0.30:
        risk = "Moderate"
    else:
        risk = "Elevated"
    
    return {
        "probability": round(proba, 4),
        "prediction": "Default" if proba >= threshold else "Fully Paid",
        "risk_level": risk,
        "threshold": threshold,
        "imputed": imputed,
    }
        

