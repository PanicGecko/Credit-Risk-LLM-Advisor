import pandas as pd
import pytest
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/train_config copy.yaml"))
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

RAW_COLUMNS = [
    "id", "issue_d", "revenue", "dti_n", "loan_amnt", "fico_n",
    "experience_c", "emp_length", "purpose", "home_ownership_n",
    "addr_state", "zip_code", "Default", "title", "desc"
]

@pytest.fixture(scope="module")
def data():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/LC_loans_granting_model_dataset.csv"))
    df = pd.read_csv(data_path)
    return df

def test_data_expected_columns(data):
    """Verify the expected raw columns are present in the dataset."""
    missing_cols = set(RAW_COLUMNS) - set(data.columns)
    assert not missing_cols, f"Missing expected columns in dataset: {missing_cols}"

def test_data_target_expected_values(data):
    """Verify target variable contains only expected binary values (0 or 1)."""
    unique_vals = set(data["Default"].dropna().unique())
    expected_vals = {0, 1}
    assert unique_vals.issubset(expected_vals), f"Unexpected values in target column: {unique_vals - expected_vals}"

def test_data_numeric_ranges(data):
    """Verify that numeric features are within expected reasonable ranges."""
    # Revenue (annual income) should be positive
    assert data["revenue"].min() > 0, "Revenue should be positive"

    # FICO score should be in valid credit score range
    assert data["fico_n"].min() >= 300, "FICO score below minimum of 300"
    assert data["fico_n"].max() <= 850, "FICO score above maximum of 850"

    # Loan amount should be positive
    assert data["loan_amnt"].min() > 0, "Loan amount should be positive"

    # DTI ratio should not be negative
    assert data["dti_n"].min() >= 0, "Debt-to-income ratio should not be negative"

def test_data_no_duplicate_ids(data):
    """Verify that loan IDs are unique."""
    assert data["id"].nunique() == len(data), "Duplicate loan IDs found in dataset"

def test_data_sufficient_rows(data):
    """Verify the dataset has enough rows for meaningful training."""
    assert len(data) >= 500, f"Dataset too small: only {len(data)} rows"

def test_data_issue_date_parseable(data):
    """Verify the issue date column can be parsed as a date."""
    try:
        pd.to_datetime(data["issue_d"], format="%b-%Y")
    except Exception as e:
        pytest.fail(f"issue_d column could not be parsed as a date: {e}")
