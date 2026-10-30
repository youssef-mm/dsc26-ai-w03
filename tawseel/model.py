"""Failure-risk baseline for Tawseel COD parcels.

You write this file. Keep every name and signature exactly as it is: the grader imports them.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "failed"

# The raw columns of make_parcels() that the model may use at decision time (when the parcel
# is created). Nothing known only after the delivery attempt, not the label, not the row id.
FEATURE_COLUMNS: list[str] = [
    "customer_id",
    "merchant_id",
    "area",
    "created_at",
    "promised_slot",
    "cod_amount",
    "n_items",
    "address_text",
]


def _clean_text_1d(x) -> list[str]:
    """Clean text input into a 1D list of strings, safely handling None/NaN and arrays/DataFrames."""
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    elif isinstance(x, np.ndarray):
        x = x.ravel()
    return pd.Series(x).fillna("").astype(str).tolist()


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df (same index and row order) with two new columns.

    Needs customer_id, created_at and failed.
    - customer_prior_parcels (int): the same customer's parcels with created_at strictly
      earlier than this row.
    - customer_prior_fail_rate (float): share of those that failed; NaN when there are none.
    Parcels with the same timestamp are not prior to each other. The result must not depend
    on the input row order. Never modify the input.
    """
    out = df.copy()
    if len(df) == 0:
        out["customer_prior_parcels"] = pd.Series(dtype=int)
        out["customer_prior_fail_rate"] = pd.Series(dtype=float)
        return out

    # Group by customer_id and created_at to correctly handle ties
    agg = df.groupby(["customer_id", "created_at"], as_index=False).agg(
        n_parcels=("failed", "count"),
        n_fails=("failed", "sum"),
    )
    agg = agg.sort_values(["customer_id", "created_at"])

    cust_grp = agg.groupby("customer_id")
    agg["customer_prior_parcels"] = cust_grp["n_parcels"].cumsum() - agg["n_parcels"]
    agg["customer_prior_fails"] = cust_grp["n_fails"].cumsum() - agg["n_fails"]

    agg["customer_prior_fail_rate"] = np.where(
        agg["customer_prior_parcels"] > 0,
        agg["customer_prior_fails"] / agg["customer_prior_parcels"],
        np.nan,
    )

    agg_indexed = agg.set_index(["customer_id", "created_at"])[
        ["customer_prior_parcels", "customer_prior_fail_rate"]
    ]
    out_keys = pd.MultiIndex.from_arrays([out["customer_id"], out["created_at"]])
    matched = agg_indexed.loc[out_keys]

    out["customer_prior_parcels"] = matched["customer_prior_parcels"].to_numpy().astype(int)
    out["customer_prior_fail_rate"] = matched["customer_prior_fail_rate"].to_numpy().astype(float)
    return out


def time_split(df: pd.DataFrame, time_col: str = "created_at",
               test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test).

    ValueError unless 0 < test_fraction < 1. Sort by time; n_test = ceil(len(df) * test_fraction);
    the boundary is the timestamp of the first of the last n_test rows. Rows at or after the
    boundary are test, rows before it are train.
    """
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(f"test_fraction must be strictly between 0 and 1, got {test_fraction}")

    n = len(df)
    if n == 0:
        return df.copy(), df.copy()

    n_test = math.ceil(n * test_fraction)
    n_test = min(n, max(1, n_test))

    sorted_times = df[time_col].sort_values()
    boundary = sorted_times.iloc[-n_test]

    test = df[df[time_col] >= boundary]
    train = df[df[time_col] < boundary]
    return train, test


def build_pipeline(seed: int = 0) -> Pipeline:
    """Return an UNFITTED Pipeline.

    Input: a DataFrame with at least FEATURE_COLUMNS plus the two history columns; ignore any
    other column. The final step must support predict_proba. Must not fail on unseen
    categories, missing numbers, missing categories or empty address strings. Same seed and
    same data must give identical predictions.
    """
    num_cols = ["cod_amount", "n_items", "customer_prior_parcels", "customer_prior_fail_rate"]
    cat_cols = ["area", "promised_slot", "merchant_id"]
    txt_col = "address_text"

    num_pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    txt_pipe = make_pipeline(
        FunctionTransformer(_clean_text_1d),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
            ("txt", txt_pipe, txt_col),
        ],
        remainder="drop",
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    return model


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """{"roc_auc", "pr_auc", "brier", "base_rate", "n"} from model.predict_proba(test_df)[:, 1]
    against test_df["failed"] (roc_auc_score, average_precision_score, brier_score_loss)."""
    y_true = test_df[TARGET]
    y_prob = model.predict_proba(test_df)[:, 1]

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    base_rate = float(y_true.mean())
    n = int(len(test_df))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "base_rate": base_rate,
        "n": n,
    }


def compare_to_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0) -> dict:
    """Fit build_pipeline(seed) and DummyClassifier(strategy="prior") on train_df only, evaluate
    both on test_df, return {"model": {...}, "dummy": {...}, "lift_pr_auc": model - dummy PR-AUC}."""
    model = build_pipeline(seed=seed)
    model.fit(train_df, train_df[TARGET])

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(train_df, train_df[TARGET])

    model_eval = evaluate(model, test_df)
    dummy_eval = evaluate(dummy, test_df)
    lift_pr_auc = float(model_eval["pr_auc"] - dummy_eval["pr_auc"])

    return {
        "model": model_eval,
        "dummy": dummy_eval,
        "lift_pr_auc": lift_pr_auc,
    }
