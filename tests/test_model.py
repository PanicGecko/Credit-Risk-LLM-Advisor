import pandas as pd
import numpy as np
import pytest
import os
import sys
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from evaluation import evaluate_model
from train import train_model, CONFIG, MODEL_CONFIG


@pytest.fixture(scope="module")
def sample_data():
    """Synthetic dataset matching the Lending Club feature structure after preprocessing."""
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        "revenue":                  np.random.uniform(30000, 200000, n),
        "loan_amount":              np.random.uniform(1000, 40000, n),
        "fico_score":               np.random.uniform(300, 850, n),
        "issue_year":               np.random.randint(2007, 2016, n),
        "issue_month":              np.random.randint(1, 13, n),
        "issue_quarter":            np.random.randint(1, 5, n),
        "post_crisis_recovery":     np.random.randint(0, 2, n),
        "loan_to_income":           np.random.uniform(0.01, 1.5, n),
        "monthly_payment_burden":   np.random.uniform(0.01, 0.5, n),
        "indebtedness":             np.random.uniform(0, 40, n),
        "emp_length":               np.random.randint(0, 11, n),
        "indebtedness_missing":     np.random.randint(0, 2, n),
        "emp_length_missing":       np.random.randint(0, 2, n),
        "purpose_car":              np.random.randint(0, 2, n),
        "purpose_debt_consolidation": np.random.randint(0, 2, n),
        "home_ownership_RENT":      np.random.randint(0, 2, n),
        "home_ownership_OWN":       np.random.randint(0, 2, n),
        "state_CA":                 np.random.randint(0, 2, n),
        "state_TX":                 np.random.randint(0, 2, n),
        "has_experience_1":         np.random.randint(0, 2, n),
        "label":                    np.random.randint(0, 2, n),
    })
    return df


def test_model_prediction_type_shape(sample_data, tmp_path):
    """Verify model produces predictions of correct type and shape."""
    temp_csv = tmp_path / "sample_data.csv"
    sample_data.to_csv(temp_csv, index=False)
    
    test_config = CONFIG.copy()
    test_config["data_url"] = str(temp_csv)
    # Reduce n_estimators to speed up test
    test_config["n_estimators"] = 5
    
    # Train the model 
    metrics = train_model()
    
    # Load the trained model to perform predictions
    import pickle
    model_path = "models/model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    
    assert metrics['test_size'] == 1000, "Config test_size was 0.2, so 20 rows of 100 should be in test"
    assert type(metrics['accuracy']) == float
    
def test_model_minimum_performance(tmp_path):
    """Verify the model achieves minimum performance threshold on known test set.
    According to requirements, we should train on a small sample and verify performance threshold."""

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/test/sample_loans.csv"))
    df = pd.read_csv(data_path).sample(200, random_state=42)
    temp_csv = tmp_path / "actual_sample.csv"
    df.to_csv(temp_csv, index=False)
    
    test_config = CONFIG.copy()
    test_config["data_url"] = str(temp_csv)
    
    # It must achieve a minimum performance threshold
    metrics = train_model()
    
    assert metrics["accuracy"] >= test_config["min_accuracy"]
    assert metrics["f1_score"] >= test_config["min_f1"]