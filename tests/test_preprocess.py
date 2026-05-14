import pandas as pd
import numpy as np
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from data_preprocess import (
    clean_data,
    encode_categoricals,
    validate_dataframe,
    add_missing_indicator,
    map_to_nan,
    norm_preprocessor,
)


def test_clean_data_handles_numeric_missing_values():
    """Verify clean_data fills numeric missing values with the column median."""
    df = pd.DataFrame({
        "revenue": [30000.0, 60000.0, 90000.0, np.nan],
        "fico_score": [650.0, 700.0, 750.0, 800.0],
    })
    result = clean_data(df, numeric_columns=["revenue"], categorical_columns=[])

    assert result["revenue"].isna().sum() == 0, "Numeric missing values should be filled"
    assert result["revenue"].iloc[-1] == 60000.0, "Missing numeric should be filled with median"


def test_clean_data_handles_categorical_missing_values():
    """Verify clean_data fills categorical missing values with 'UnKnown'."""
    df = pd.DataFrame({"purpose": ["debt_consolidation", "home_improvement", np.nan]})
    result = clean_data(df, numeric_columns=[], categorical_columns=["purpose"])

    assert result["purpose"].isna().sum() == 0, "Categorical missing values should be filled"
    assert "UnKnown" in result["purpose"].values, "Missing categorical should be filled with 'UnKnown'"


def test_clean_data_does_not_modify_original():
    """Verify clean_data does not modify the original dataframe in place."""
    df = pd.DataFrame({
        "revenue": [50000.0, np.nan],
        "purpose": ["car", np.nan],
    })
    df_copy = df.copy()
    _ = clean_data(df, numeric_columns=["revenue"], categorical_columns=["purpose"])

    assert pd.isna(df["revenue"].iloc[1]), "Original dataframe should still have NaN"
    pd.testing.assert_frame_equal(df, df_copy)


def test_encode_categoricals_creates_dummy_columns():
    """Verify encode_categoricals properly one-hot encodes categorical columns."""
    df = pd.DataFrame({"home_ownership": ["RENT", "OWN", "MORTGAGE", "RENT"]})
    result = encode_categoricals(df, categorical_columns=["home_ownership"])

    assert "home_ownership" not in result.columns, "Original column should be dropped after encoding"
    dummy_cols = [c for c in result.columns if c.startswith("home_ownership_")]
    assert len(dummy_cols) >= 2, "Should create at least 2 dummy columns"


def test_encode_categoricals_does_not_modify_original():
    """Verify encode_categoricals does not modify the original dataframe."""
    df = pd.DataFrame({"purpose": ["car", "home_improvement"]})
    _ = encode_categoricals(df, categorical_columns=["purpose"])

    assert "purpose" in df.columns, "Original dataframe should still have the 'purpose' column"
    assert "purpose_car" not in df.columns, "Dummy columns should not appear in original dataframe"


def test_validate_dataframe_raises_missing_columns():
    """Verify validate_dataframe raises ValueError when required columns are missing."""
    df = pd.DataFrame({"revenue": [50000], "label": [0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(df, required_columns=["revenue", "loan_amount", "fico_score"], target_column="label")


def test_validate_dataframe_raises_missing_target():
    """Verify validate_dataframe raises ValueError when the target column is missing."""
    df = pd.DataFrame({"revenue": [50000], "loan_amount": [10000]})
    with pytest.raises(ValueError, match="Target column"):
        validate_dataframe(df, required_columns=["revenue", "loan_amount"], target_column="label")


def test_validate_dataframe_raises_empty_dataframe():
    """Verify validate_dataframe raises ValueError on an empty dataframe."""
    df = pd.DataFrame(columns=["revenue", "loan_amount", "label"])
    with pytest.raises(ValueError, match="Dataframe is empty"):
        validate_dataframe(df, required_columns=["revenue", "loan_amount"], target_column="label")


def test_validate_dataframe_passes_on_valid_input():
    """Verify validate_dataframe returns True when all requirements are met."""
    df = pd.DataFrame({"revenue": [50000], "loan_amount": [10000], "label": [1]})
    assert validate_dataframe(df, required_columns=["revenue", "loan_amount"], target_column="label") is True


def test_add_missing_indicator_marks_nulls():
    """Verify add_missing_indicator creates correct binary indicator columns."""
    df = pd.DataFrame({"emp_length": [5.0, np.nan, 3.0], "indebtedness": [10.0, 20.0, np.nan]})
    result = add_missing_indicator(df, columns=["emp_length", "indebtedness"])

    assert "emp_length_missing" in result.columns
    assert result["emp_length_missing"].iloc[1] == 1, "NaN row should be marked as 1"
    assert result["emp_length_missing"].iloc[0] == 0, "Non-NaN row should be marked as 0"
    assert "indebtedness_missing" in result.columns
    assert result["indebtedness_missing"].iloc[2] == 1


def test_add_missing_indicator_does_not_modify_original():
    """Verify add_missing_indicator does not modify the original dataframe."""
    df = pd.DataFrame({"emp_length": [np.nan, 2.0]})
    _ = add_missing_indicator(df, columns=["emp_length"])
    assert "emp_length_missing" not in df.columns, "Indicator column should not appear in original dataframe"


def test_map_to_nan_replaces_sentinel_value():
    """Verify map_to_nan correctly replaces a sentinel value with NaN."""
    df = pd.DataFrame({"emp_length": ["NI", "5 years", "NI", "3 years"]})
    result = map_to_nan(df, column="emp_length", value="NI")

    assert result["emp_length"].isna().sum() == 2, "Both 'NI' entries should be replaced with NaN"
    assert result["emp_length"].iloc[1] == "5 years", "Non-sentinel values should remain unchanged"


def test_map_to_nan_does_not_modify_original():
    """Verify map_to_nan does not modify the original dataframe."""
    df = pd.DataFrame({"emp_length": ["NI", "3 years"]})
    _ = map_to_nan(df, column="emp_length", value="NI")
    assert df["emp_length"].iloc[0] == "NI", "Original dataframe should still have sentinel value"


def test_norm_preprocessor_standard_scales_to_zero_mean():
    """Verify StandardScaler columns have approximately zero mean after fitting."""
    preprocessor = norm_preprocessor(standard_cols=["revenue"], minmax_cols=["fico_score"])
    X = pd.DataFrame({"revenue": [30000.0, 60000.0, 90000.0], "fico_score": [600.0, 700.0, 800.0]})
    result = preprocessor.fit_transform(X)

    assert abs(result[:, 0].mean()) < 1e-10, "StandardScaler column should have ~zero mean"


def test_norm_preprocessor_minmax_scales_to_unit_range():
    """Verify MinMaxScaler columns are scaled to the [0, 1] range after fitting."""
    preprocessor = norm_preprocessor(standard_cols=["revenue"], minmax_cols=["fico_score"])
    X = pd.DataFrame({"revenue": [30000.0, 60000.0, 90000.0], "fico_score": [600.0, 700.0, 800.0]})
    result = preprocessor.fit_transform(X)

    minmax_col = result[:, 1]
    assert minmax_col.min() >= 0.0, "MinMaxScaler output should be >= 0"
    assert minmax_col.max() <= 1.0, "MinMaxScaler output should be <= 1"
