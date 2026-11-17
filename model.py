"""
model.py - machine-learning core of MediScan AI.

What this file does
-------------------
1. Loads the Wisconsin Diagnostic Breast Cancer (WDBC) dataset that ships with
   scikit-learn and runs basic cleaning / validation checks.
2. Splits the data into training and test sets (stratified, fixed random seed).
3. Trains a Logistic Regression model inside a scikit-learn Pipeline
   (StandardScaler -> LogisticRegression).  Because the scaler lives *inside*
   the pipeline and is fitted on the training split only, no information from
   the test set leaks into training.
4. Evaluates on the held-out test set (accuracy, precision, recall, F1,
   confusion matrix, ROC-AUC) and prepares explainability tables.

IMPORTANT: this is an educational decision-support prototype. It has NOT been
clinically validated and must never be used to diagnose real patients.

Class convention (from scikit-learn): 0 = malignant, 1 = benign.
In this project "positive" always means MALIGNANT, because that is the class a
reviewer most needs to catch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
RANDOM_SEED = 42      # fixed seed -> reproducible results
TEST_SIZE = 0.20      # 20 % of records are held out for evaluation
MALIGNANT = 0         # scikit-learn encoding of the target column
BENIGN = 1
CLASS_LABELS = {MALIGNANT: "Malignant", BENIGN: "Benign"}

# The 30 feature names, in the order the model expects them.
FEATURE_NAMES: list[str] = list(load_breast_cancer().feature_names)


# ----------------------------------------------------------------------------
# Data loading, cleaning and validation
# ----------------------------------------------------------------------------
def load_dataset() -> tuple[pd.DataFrame, dict]:
    """Load WDBC, validate it and return (dataframe, data-quality report).

    The returned dataframe has the 30 feature columns, a numeric ``target``
    column (0 = malignant, 1 = benign) and a readable ``diagnosis`` column.
    """
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame[FEATURE_NAMES + ["target"]].copy()

    # --- validation / cleaning -------------------------------------------
    df = df.apply(pd.to_numeric, errors="coerce")        # force numeric types
    missing = int(df.isna().sum().sum())
    if missing:
        df = df.dropna()                                  # (none expected)
    duplicates = int(df.duplicated().sum())
    if duplicates:
        df = df.drop_duplicates()                         # (none expected)
    negatives = int((df[FEATURE_NAMES] < 0).sum().sum())
    if negatives:
        raise ValueError("Dataset contains negative measurements - unexpected for WDBC.")

    df["target"] = df["target"].astype(int)
    if not set(df["target"].unique()) <= {MALIGNANT, BENIGN}:
        raise ValueError("Unexpected target labels in dataset.")
    df["diagnosis"] = df["target"].map(CLASS_LABELS)

    counts = df["diagnosis"].value_counts()
    quality = {
        "n_records": int(len(df)),
        "n_features": len(FEATURE_NAMES),
        "missing_values_found": missing,
        "duplicate_rows_found": duplicates,
        "negative_values_found": negatives,
        "class_counts": {k: int(v) for k, v in counts.items()},
    }
    return df, quality


# ----------------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """Scaler + Logistic Regression.  Scaling is learned from training data only."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )


def malignant_probability(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Model-estimated probability of the MALIGNANT class for each row of X."""
    cols = [str(c) for c in pipeline.feature_names_in_]
    frame = X[cols].copy()
    frame.columns = cols                      # guarantee plain-str column names
    idx = list(pipeline.classes_).index(MALIGNANT)
    return pipeline.predict_proba(frame)[:, idx]


# ----------------------------------------------------------------------------
# Training + evaluation bundle
# ----------------------------------------------------------------------------
@dataclass
class ModelBundle:
    """Everything the web app needs, produced once by :func:`train_model`."""

    pipeline: Pipeline
    data: pd.DataFrame                 # features + target + diagnosis
    feature_names: list[str]
    train_index: pd.Index
    test_index: pd.Index
    metrics: dict                      # test-set metrics
    confusion: np.ndarray              # rows = actual, cols = predicted; order [malignant, benign]
    roc: dict                          # fpr / tpr / thresholds arrays
    test_probs: pd.DataFrame           # test records: actual label + P(malignant)
    coefficients: pd.DataFrame         # global explainability table
    cv_auc_mean: float                 # 5-fold CV ROC-AUC on the training split only
    cv_auc_std: float
    train_mean: pd.Series              # scaler statistics (training data only)
    train_scale: pd.Series
    train_min: pd.Series
    train_max: pd.Series
    quality: dict                      # data-quality report


def train_model(random_seed: int = RANDOM_SEED, test_size: float = TEST_SIZE) -> ModelBundle:
    """Load data, split, train, evaluate - fully deterministic for a given seed."""
    df, quality = load_dataset()
    X = df[FEATURE_NAMES]
    y = df["target"]

    # Stratified split keeps the benign/malignant ratio the same in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_seed
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)          # scaler + model see ONLY training data

    # --- test-set evaluation (malignant = positive class) ------------------
    p_mal = malignant_probability(pipe, X_test)
    y_pred = np.where(p_mal >= 0.5, MALIGNANT, BENIGN)
    y_true_mal = (y_test.to_numpy() == MALIGNANT).astype(int)
    pred_mal = (y_pred == MALIGNANT).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[MALIGNANT, BENIGN])
    tn_benign_correct = int(cm[1, 1])
    fp = int(cm[1, 0])                  # benign called malignant
    fn = int(cm[0, 1])                  # malignant called benign (the costly error)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_true_mal, pred_mal, zero_division=0)),
        "recall": float(recall_score(y_true_mal, pred_mal, zero_division=0)),
        "f1": float(f1_score(y_true_mal, pred_mal, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_mal, p_mal)),
        "specificity": float(tn_benign_correct / (tn_benign_correct + fp)) if (tn_benign_correct + fp) else 0.0,
        "false_negatives": fn,
        "false_positives": fp,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    fpr, tpr, thr = roc_curve(y_true_mal, p_mal)

    # Cross-validation on the TRAINING split only (fresh pipeline per fold).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    cv_scores = cross_val_score(build_pipeline(), X_train, y_train, cv=cv, scoring="roc_auc")

    # --- explainability: standardized coefficients --------------------------
    clf = pipe.named_steps["clf"]
    scaler = pipe.named_steps["scaler"]
    coef_benign = clf.coef_[0]          # coefficients describe the BENIGN class (class 1)
    coef_df = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "coefficient": coef_benign,
            "malignant_effect": -coef_benign,   # + raises P(malignant), - lowers it
        }
    )
    coef_df["abs_importance"] = coef_df["malignant_effect"].abs()
    coef_df["direction"] = np.where(
        coef_df["malignant_effect"] > 0, "Raises malignant probability", "Lowers malignant probability"
    )
    coef_df = coef_df.sort_values("abs_importance", ascending=False).reset_index(drop=True)

    test_probs = pd.DataFrame(
        {"actual": y_test.map(CLASS_LABELS), "p_malignant": p_mal}, index=X_test.index
    )

    return ModelBundle(
        pipeline=pipe,
        data=df,
        feature_names=list(FEATURE_NAMES),
        train_index=X_train.index,
        test_index=X_test.index,
        metrics=metrics,
        confusion=cm,
        roc={"fpr": fpr, "tpr": tpr, "thresholds": thr},
        test_probs=test_probs,
        coefficients=coef_df,
        cv_auc_mean=float(cv_scores.mean()),
        cv_auc_std=float(cv_scores.std()),
        train_mean=pd.Series(scaler.mean_, index=FEATURE_NAMES),
        train_scale=pd.Series(scaler.scale_, index=FEATURE_NAMES),
        train_min=X_train.min(),
        train_max=X_train.max(),
        quality=quality,
    )


# ----------------------------------------------------------------------------
# Per-case explanation
# ----------------------------------------------------------------------------
def explain_case(bundle: ModelBundle, row: pd.Series) -> pd.DataFrame:
    """Explain one record with simple, exact logistic-regression arithmetic.

    For each feature:  z = (value - training mean) / training std,
    contribution = -coefficient * z  (log-odds of the MALIGNANT class).
    Positive contribution = pushes toward malignant, negative = toward benign.
    The contributions plus the intercept term add up exactly to the model's
    log-odds, so the explanation is faithful to the model (not an approximation).
    """
    values = row[bundle.feature_names].astype(float)
    z = (values - bundle.train_mean) / bundle.train_scale
    coef = pd.Series(
        bundle.pipeline.named_steps["clf"].coef_[0], index=bundle.feature_names
    )
    out = pd.DataFrame(
        {
            "feature": bundle.feature_names,
            "value": values.to_numpy(),
            "training_mean": bundle.train_mean.to_numpy(),
            "z_score": z.to_numpy(),
            "contribution": (-coef * z).to_numpy(),
        }
    )
    out["abs_contribution"] = out["contribution"].abs()
    return out.sort_values("abs_contribution", ascending=False).reset_index(drop=True)
