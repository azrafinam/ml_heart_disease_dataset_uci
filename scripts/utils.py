"""
Shared utilities for the UCI Heart Disease machine-learning project.

The helpers in this file keep the notebooks and command-line pipeline aligned:
data is split before preprocessing, preprocessing is fit on training data only,
and all models are evaluated with the same metrics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

TargetMode = Literal["multiclass", "binary"]

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency guard
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency guard
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency guard
    CatBoostClassifier = None


RANDOM_STATE = 42
TARGET_COLUMN = "num"
BINARY_TARGET_COLUMN = "num_binary"
ID_COLUMNS = ["id"]

# num supports two supervised formulations:
# - multiclass: 0=no disease, 1-4=increasing severity (five classes)
# - binary: 0=no disease, 1=disease (severity levels 1-4 collapsed to 1)
TARGET_MODE_DESCRIPTIONS = {
    "multiclass": "0=no disease, 1-4=increasing heart disease severity",
    "binary": "0=no disease, 1=disease (original num values 1-4 mapped to 1)",
}


def encode_target(y: pd.Series, mode: TargetMode = "multiclass") -> pd.Series:
    """Encode the raw num target for modeling."""
    y_encoded = pd.to_numeric(y, errors="raise").astype(int)
    unexpected = sorted(set(y_encoded.unique()) - {0, 1, 2, 3, 4})
    if unexpected:
        raise ValueError(f"Unexpected target values in '{TARGET_COLUMN}': {unexpected}")

    if mode == "multiclass":
        return y_encoded
    if mode == "binary":
        return (y_encoded > 0).astype(int)
    raise ValueError(f"Unknown target mode: {mode}")


def add_binary_target_column(df: pd.DataFrame, source_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """Add a derived binary target column alongside the original num severity labels."""
    df_out = df.copy()
    df_out[BINARY_TARGET_COLUMN] = encode_target(df_out[source_col], mode="binary")
    return df_out


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the heart disease dataset and treat '?' as missing."""
    df = pd.read_csv(path, na_values="?")
    print(f"  Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features and target."""
    missing_target = target_col not in df.columns
    if missing_target:
        raise ValueError(f"Target column '{target_col}' was not found.")

    drop_cols = [col for col in ID_COLUMNS + [target_col] if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col].copy()
    return X, y


def get_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.columns[~X.columns.isin(numeric_features)].tolist()
    return numeric_features, categorical_features


def create_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Create a leakage-safe preprocessing transformer."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot_encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    target_col: str = TARGET_COLUMN,
    target_mode: TargetMode = "multiclass",
) -> dict[str, Any]:
    """Split first, then fit preprocessing on the training data only."""
    X, y_raw = split_features_target(df, target_col=target_col)
    y = encode_target(y_raw, mode=target_mode)
    numeric_features, categorical_features = get_feature_types(X)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = create_preprocessor(numeric_features, categorical_features)
    X_train_array = preprocessor.fit_transform(X_train_raw)
    X_test_array = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out()

    X_train = pd.DataFrame(X_train_array, columns=feature_names).reset_index(drop=True)
    X_test = pd.DataFrame(X_test_array, columns=feature_names).reset_index(drop=True)

    return {
        "X": X,
        "y": y,
        "y_raw": y_raw,
        "target_mode": target_mode,
        "target_description": TARGET_MODE_DESCRIPTIONS[target_mode],
        "X_train_raw": X_train_raw.reset_index(drop=True),
        "X_test_raw": X_test_raw.reset_index(drop=True),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "preprocessor": preprocessor,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_names": feature_names.tolist(),
    }


def build_model_zoo(random_state: int = RANDOM_STATE, target_mode: TargetMode = "multiclass") -> dict[str, Any]:
    """Return a broad, beginner-friendly classifier collection."""
    is_binary = target_mode == "binary"
    models: dict[str, Any] = {
        "dummy_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=3000, random_state=random_state),
        "ridge_classifier": RidgeClassifier(),
        "sgd_classifier": SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-3, random_state=random_state),
        "perceptron": Perceptron(max_iter=2000, random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "gaussian_nb": GaussianNB(),
        "linear_svm": LinearSVC(random_state=random_state, dual="auto", max_iter=5000),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=random_state),
        "decision_tree": DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=random_state,
            n_jobs=-1,
        ),
        "bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=random_state),
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "adaboost": AdaBoostClassifier(n_estimators=100, learning_rate=0.05, random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_state),
        "linear_discriminant_analysis": LinearDiscriminantAnalysis(),
        "nearest_centroid": NearestCentroid(),
    }

    if XGBClassifier is not None:
        xgb_params: dict[str, Any] = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": random_state,
            "n_jobs": -1,
        }
        if is_binary:
            xgb_params["objective"] = "binary:logistic"
            xgb_params["eval_metric"] = "logloss"
        else:
            xgb_params["eval_metric"] = "mlogloss"
        models["xgboost"] = XGBClassifier(**xgb_params)

    if LGBMClassifier is not None:
        lgbm_params: dict[str, Any] = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        if is_binary:
            lgbm_params["objective"] = "binary"
        models["lightgbm"] = LGBMClassifier(**lgbm_params)

    if CatBoostClassifier is not None:
        catboost_params: dict[str, Any] = {
            "iterations": 200,
            "learning_rate": 0.05,
            "depth": 5,
            "loss_function": "Logloss" if is_binary else "MultiClass",
            "random_seed": random_state,
            "allow_writing_files": False,
            "verbose": False,
        }
        models["catboost"] = CatBoostClassifier(**catboost_params)

    models["random_forest_random_search"] = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_distributions={
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        n_iter=8,
        scoring="f1_weighted",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
    )

    return models


def get_model_scores(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    """Return probability or decision scores for ROC-AUC when a model supports it."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.asarray(scores)
    return None


def evaluate_classifier(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Evaluate a fitted classifier on the held-out test set."""
    y_pred = model.predict(X_test)
    y_score = get_model_scores(model, X_test)

    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    try:
        if y_score is not None:
            unique_classes = np.unique(y_test)
            if len(unique_classes) == 2:
                if getattr(y_score, "ndim", 1) == 2 and y_score.shape[1] >= 2:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_score[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_score)
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_test,
                    y_score,
                    multi_class="ovr",
                    average="weighted",
                )
        else:
            metrics["roc_auc"] = np.nan
    except Exception:
        metrics["roc_auc"] = np.nan

    return metrics


def create_model_comparison_report(evaluation_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Create a ranked model comparison table."""
    comparison_df = pd.DataFrame(evaluation_results).T
    metric_cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    comparison_df = comparison_df[[col for col in metric_cols if col in comparison_df.columns]]
    comparison_df = comparison_df.sort_values("f1_score", ascending=False)
    return comparison_df.round(4)


def get_best_model(comparison_df: pd.DataFrame, metric: str = "f1_score") -> tuple[str, float]:
    """Return the model name and score for the best row."""
    if metric not in comparison_df.columns:
        metric = "f1_score"
    best_model = comparison_df[metric].idxmax()
    best_score = float(comparison_df.loc[best_model, metric])
    return best_model, best_score


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON with parent directory creation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


def save_model(model: Any, path: str | Path) -> None:
    """Save a model artifact with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Any:
    """Load a model artifact saved with joblib or pickle-compatible data."""
    return joblib.load(path)


def ensure_output_dirs() -> None:
    """Create project output directories."""
    for path in [
        "data/processed",
        "outputs/models",
        "outputs/reports",
        "outputs/visualizations",
    ]:
        os.makedirs(path, exist_ok=True)
