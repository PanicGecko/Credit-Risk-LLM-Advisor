import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def select_columns(df, columns):
    """Select specific columns from a dataframe."""
    df = df.copy()
    df = df[columns]
    return df

def rename_cols(df, cols_dict):
    df = df.copy()
    df.rename(columns=cols_dict, inplace=True)
    return df

def date_features(df, date_column):
    """Extract date features from a date column."""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df["issue_year"] = df[date_column].dt.year
    df["issue_month"] = df[date_column].dt.month
    df["issue_quarter"] = df[date_column].dt.quarter
    df["post_crisis_recovery"] = ((df["issue_year"] >= 2008) & (df["issue_year"] <= 2010)).astype(int)
    df = df.drop(columns=[date_column])
    return df

def additional_loan_features(df):
    df = df.copy()
    df['loan_to_income'] = df['loan_amount'] / (df['revenue'] + 1)
    df['monthly_payment_burden'] = (df['loan_amount'] * 0.03) / (df['revenue'] / 12 + 1)
    for col in ['loan_to_income', 'monthly_payment_burden']:
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=upper)
    top_states = df['state'].value_counts().head(20).index.tolist()
    df['state'] = df['state'].where(df['state'].isin(top_states), 'Other')
    return df
    

def validate_dataframe(df, required_columns, target_column):
    """Check that a dataframe meets basic requirements."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    if len(df) == 0:
        raise ValueError("Dataframe is empty")

    return True

def map_to_nan(df, column, value):
    """Map specific values in a column to NaN."""
    df = df.copy()
    df[column] = df[column].replace(value, np.nan)
    return df
    

def add_missing_indicator(df, columns):
    """Add missing indicator columns for a list of columns."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
    return df

def clean_data(df, numeric_columns, categorical_columns):
    """Clean a dataframe by handling missing values and encoding categoricals."""
    df = df.copy()
    
    # Fill numeric missing values with median
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values with mode
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna("UnKnown")

    return df

def encode_categoricals(df, categorical_columns):
    """One-hot encode categorical columns."""
    df = df.copy()
    for col in categorical_columns:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col)
    return df


def norm_preprocessor(standard_cols, minmax_cols):
    """Create a ColumnTransformer for scaling."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('standard', StandardScaler(), standard_cols),
            ('minmax', MinMaxScaler(), minmax_cols),
        ], remainder='passthrough')
    return preprocessor