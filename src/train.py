import pandas as pd
import numpy as np
import json
import os
import sys
import pickle
from pathlib import Path
import yaml
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
sys.path.insert(0, os.path.dirname(__file__))
from data_preprocess import validate_dataframe, clean_data, encode_categoricals, check_data_quality, select_columns, decode_target, encode_target, norm_preprocessor, rename_cols, date_features, additional_loan_features, map_to_nan, add_missing_indicator
from evaluation import evaluate_model, check_thresholds
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"
CONFIG = load_config(DEFAULT_CONFIG_PATH)
MODEL_CONFIG = load_config(PROJECT_ROOT / "configs" / CONFIG["config_file"])

def load_data(relative_path):
    full_path = PROJECT_ROOT / relative_path
    print(f"Loading data from {full_path}...")
    return pd.read_csv(full_path)

def get_model(config):
    if config is None:
        print("No Config File!!")
        return None
    model = None
    if config["model"] == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=config["params"]["n_estimators"], 
            max_depth=config["params"]["max_depth"],
            random_state=config["params"]["random_state"],
            # min_samples_leaf=config["params"]["min_samples_leaf"],
            class_weight='balanced'
        )
    elif config["model"] == "Logistic Regression":
        model = LogisticRegression(
            C=config["params"]["C"],
            max_iter=config["params"]["max_iter"],
            random_state=config["params"]["random_state"],
            class_weight='balanced'
        )
    elif config["model"] == "Gradient Boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=config["params"]["learning_rate"],
            max_depth=config["params"]["max_depth"],
            random_state=config["params"]["random_state"],
            max_iter=config["params"]["max_iter"],
            class_weight='balanced',
            min_samples_leaf=config["params"]["min_samples_leaf"],
            early_stopping=config["params"]["early_stopping"],
            l2_regularization=config["params"]["l2_regularization"]
        )
    elif config["model"] == "XGBoost":
        return XGBClassifier(
            n_estimators=config["params"]["n_estimators"],
            learning_rate=config["params"]["learning_rate"],
            max_depth=config["params"]["max_depth"],
            min_child_weight=config["params"]["min_child_weight"],
            subsample=config["params"]["subsample"],
            colsample_bytree=config["params"]["colsample_bytree"],
            gamma=config["params"]["gamma"],
            reg_lambda=config["params"]["reg_lambda"],
            scale_pos_weight=config["params"]["scale_pos_weight"],
            random_state=config["params"]["random_state"],
            n_jobs=-1,
            eval_metric='auc',
            tree_method='hist',  # fast histogram-based, equivalent to LightGBM's approach
        )
    return model

def train_model(training=False):
    """Full training pipeline. Returns metrics dictionary."""
    config = CONFIG
    model_config = MODEL_CONFIG

    df = load_data(config["data_url"])

    df = df.drop(columns=['id', 'title', 'desc', 'zip_code'])

    df = rename_cols(df, {
    'loan_amnt': 'loan_amount', 'fico_n': 'fico_score', 'experience_c': 'has_experience', 'home_ownership_n': 'home_ownership', 'addr_state': 'state', 'Default': 'label', 'dti_n': 'indebtedness'
                      })

    df = additional_loan_features(df)

    df = date_features(df, date_column="issue_d")

    if validate_dataframe(df, config["required_columns"], config["target"]):
        print("Dataframe validation passed")

    df = map_to_nan(df, 'emp_length', 'NI')

    df = add_missing_indicator(df, config["missing_numeric_columns"])

    emp_map = {
        '< 1 year': 0,
        '1 year': 1,
        '2 years': 2,
        '3 years': 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        '10+ years': 10
    }
    df['emp_length'] = df['emp_length'].map(emp_map)

    df = clean_data(df, config["missing_numeric_columns"], config["missing_categorical_columns"])

    df = encode_categoricals(df, config["missing_categorical_columns"])
    df['emp_length'] = df['emp_length'].astype(int)

    preprocessor = norm_preprocessor(config["standard_columns"], config["minmax_columns"])

    X = df.drop(columns=[config["target"]])
    y = df[config["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y
    )
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    if training:
        mlflow.set_experiment("smartphone-stress-impact-prediction")

        with mlflow.start_run() as run:
            mlflow.log_param("model", model_config["model_type"])
            mlflow.log_param("random_state", model_config["params"]["random_state"])

            if model_config["model_type"] == "Random Forest":
                mlflow.log_param("n_estimators", str(model_config["params"]["n_estimators"]))
                mlflow.log_param("max_depth", str(model_config["params"]["max_depth"]))
            elif model_config["model_type"] == "Logistic Regression":
                mlflow.log_param("C", str(model_config["params"]["C"]))
                mlflow.log_param("max_iter", str(model_config["params"]["max_iter"]))
            elif model_config["model_type"] == "Gradient Boosting":
                mlflow.log_param("learning_rate", str(model_config["params"]["learning_rate"]))
                mlflow.log_param("max_depth", str(model_config["params"]["max_depth"]))
                mlflow.log_param("max_iter", str(model_config["params"]["max_iter"]))
                mlflow.log_param("min_samples_leaf", str(model_config["params"]["min_samples_leaf"]))
            elif model_config["model_type"] == "XGBoost":
                mlflow.log_param("n_estimators", str(model_config["params"]["n_estimators"]))
                mlflow.log_param("max_depth", str(model_config["params"]["max_depth"]))
                mlflow.log_param("learning_rate", str(model_config["params"]["learning_rate"]))

            model = get_model(model_config)
            if model is None:
                raise ValueError("Model not found")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)
            metrics["train_size"] = len(X_train)
            metrics["test_size"] = len(X_test)
            metrics["n_features"] = X_train.shape[1]

            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])
            mlflow.log_metric("f1_score", metrics["f1_score"])

            check_thresholds(metrics, {"accuracy": config["min_accuracy"], "f1_score": config["min_f1"]})

            # Save model
            os.makedirs("models", exist_ok=True)
            model_path = "models/model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"\nModel saved to {model_path}")

            # Save metrics
            os.makedirs("metrics", exist_ok=True)
            metrics_path = "metrics/results.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to {metrics_path}")

            return metrics

        
    model = get_model(model_config)
    if model is None:
        raise ValueError("Model not found")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    metrics["train_size"] = len(X_train)
    metrics["test_size"] = len(X_test)
    metrics["n_features"] = X_train.shape[1]
    check_thresholds(metrics, {"accuracy": config["min_accuracy"], "f1_score": config["min_f1"]})

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")

    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    metrics_path = "metrics/results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config YAML file")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    metrics = train_model(config, training=args.train)

    # Exit with error if thresholds not met
    if metrics["accuracy"] < config["min_accuracy"]:
        print(f"\nFAILED: Accuracy below threshold")
        sys.exit(1)
    if metrics["f1_score"] < config["min_f1"]:
        print(f"\nFAILED: F1 score below threshold")
        sys.exit(1)

    print("\nAll thresholds passed!")
