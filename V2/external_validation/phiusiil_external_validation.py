"""
External validation on the PhiUSIIL Phishing URL Dataset.

This script adds an independent-dataset validation experiment after the main
project feedback.

It performs two checks:

1. Within-PhiUSIIL validation:
   Train/test split on the PhiUSIIL dataset using only URL-derived features.

2. Cross-dataset transfer:
   Train on the original project dataset using the same URL-derived features,
   then test directly on PhiUSIIL.

Run from V2/external_validation:

    python phiusiil_external_validation.py

Expected input file:

    data/PhiUSIIL_Phishing_URL_Dataset.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"

PHIUSIIL_PATH = DATA_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
ORIGINAL_DATA_PATH = V2_DIR / "5.urldata.csv"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Allow importing feature_extraction_clean.py from V2/
sys.path.append(str(V2_DIR))

import feature_extraction_clean as clean  # noqa: E402


# ---------------------------------------------------------------------
# Shared feature definition
# ---------------------------------------------------------------------

URL_ONLY_FEATURES = [
    "Have_IP",
    "Have_At",
    "URL_Length",
    "URL_Depth",
    "Redirection",
    "https_Domain",
    "TinyURL",
    "Prefix/Suffix",
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def extract_url_only_features(urls: pd.Series) -> pd.DataFrame:
    """
    Extract the 8 URL-only features used in the original project.

    This intentionally avoids webpage fetching and WHOIS lookups, so the
    external validation is deterministic and fast.
    """

    rows = []

    for url in urls.astype(str):
        rows.append({
            "Have_IP": clean.havingIP(url),
            "Have_At": clean.haveAtSign(url),
            "URL_Length": clean.getLength(url),
            "URL_Depth": clean.getDepth(url),
            "Redirection": clean.redirection(url),
            "https_Domain": clean.httpDomain(url),
            "TinyURL": clean.tinyURL(url),
            "Prefix/Suffix": clean.prefixSuffix(url),
        })

    return pd.DataFrame(rows)


def evaluate_binary_classifier(
    model_name: str,
    experiment_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
) -> dict:
    """
    Evaluate a binary phishing classifier.

    In this script:
        1 = phishing
        0 = legitimate
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    result = {
        "Experiment": experiment_name,
        "Model": model_name,
        "N Test": len(y_true),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }

    if y_score is not None:
        try:
            result["ROC-AUC"] = roc_auc_score(y_true, y_score)
        except ValueError:
            result["ROC-AUC"] = np.nan

        try:
            result["PR-AUC"] = average_precision_score(y_true, y_score)
        except ValueError:
            result["PR-AUC"] = np.nan
    else:
        result["ROC-AUC"] = np.nan
        result["PR-AUC"] = np.nan

    return result


def get_prediction_scores(model, X: pd.DataFrame) -> np.ndarray | None:
    """Return phishing probability scores if supported."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    return None


def make_xgboost_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def make_random_forest_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("External validation on PhiUSIIL")
    print("=" * 80)

    if not PHIUSIIL_PATH.exists():
        raise FileNotFoundError(
            f"Missing PhiUSIIL CSV file:\n{PHIUSIIL_PATH}\n\n"
            "Download PhiUSIIL_Phishing_URL_Dataset.csv from UCI and place it "
            "inside external_validation/data/."
        )

    if not ORIGINAL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing original project dataset:\n{ORIGINAL_DATA_PATH}"
        )

    print("Loading PhiUSIIL dataset...")
    phiusiil = pd.read_csv(PHIUSIIL_PATH)

    print("PhiUSIIL shape:", phiusiil.shape)
    print("PhiUSIIL columns:")
    print(list(phiusiil.columns))

    if "URL" not in phiusiil.columns:
        raise ValueError("Expected PhiUSIIL dataset to contain a URL column.")

    if "label" in phiusiil.columns and "Label" not in phiusiil.columns:
        phiusiil = phiusiil.rename(columns={"label": "Label"})

    if "Label" not in phiusiil.columns:
        raise ValueError("Expected PhiUSIIL dataset to contain a Label column.")

    # UCI PhiUSIIL label convention:
    # Label 1 = legitimate, Label 0 = phishing.
    # Convert to the convention used in the original project:
    # 1 = phishing, 0 = legitimate.
    phiusiil["Phishing_Label"] = 1 - phiusiil["Label"].astype(int)

    class_balance = (
        phiusiil["Phishing_Label"]
        .value_counts()
        .rename(index={0: "Legitimate", 1: "Phishing"})
        .reset_index()
    )
    class_balance.columns = ["Class", "Count"]
    class_balance["Percentage"] = (
        class_balance["Count"] / len(phiusiil) * 100
    )

    print("\nPhiUSIIL class balance after relabeling:")
    print(class_balance)

    class_balance.to_csv(
        RESULTS_DIR / "phiusiil_class_balance.csv",
        index=False,
    )

    print("\nExtracting URL-only features from PhiUSIIL...")
    X_phiusiil = extract_url_only_features(phiusiil["URL"])
    y_phiusiil = phiusiil["Phishing_Label"].astype(int)

    X_phiusiil.to_csv(
        RESULTS_DIR / "phiusiil_url_only_features.csv",
        index=False,
    )

    results = []

    # -------------------------------------------------------------
    # Experiment 1: Within-PhiUSIIL train/test validation
    # -------------------------------------------------------------

    print("\nRunning within-PhiUSIIL validation...")

    X_train_phi, X_test_phi, y_train_phi, y_test_phi = train_test_split(
        X_phiusiil,
        y_phiusiil,
        test_size=0.2,
        random_state=42,
        stratify=y_phiusiil,
    )

    within_models = {
        "XGBoost URL-only": make_xgboost_model(),
        "Random Forest URL-only": make_random_forest_model(),
    }

    for model_name, model in within_models.items():
        print(f"Training {model_name} on PhiUSIIL...")
        model.fit(X_train_phi, y_train_phi)

        y_pred = model.predict(X_test_phi)
        y_score = get_prediction_scores(model, X_test_phi)

        results.append(
            evaluate_binary_classifier(
                model_name=model_name,
                experiment_name="Within-PhiUSIIL 80/20 split",
                y_true=y_test_phi,
                y_pred=y_pred,
                y_score=y_score,
            )
        )

    # -------------------------------------------------------------
    # Experiment 2: Cross-dataset transfer
    # -------------------------------------------------------------

    print("\nRunning cross-dataset transfer validation...")

    original_data = pd.read_csv(ORIGINAL_DATA_PATH)

    missing_columns = [
        column
        for column in URL_ONLY_FEATURES + ["Label"]
        if column not in original_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The original dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    X_original = original_data[URL_ONLY_FEATURES]
    y_original = original_data["Label"].astype(int)

    transfer_models = {
        "XGBoost trained on original, tested on PhiUSIIL": make_xgboost_model(),
        "Random Forest trained on original, tested on PhiUSIIL": make_random_forest_model(),
    }

    for model_name, model in transfer_models.items():
        print(f"Training {model_name}...")
        model.fit(X_original, y_original)

        y_pred = model.predict(X_phiusiil)
        y_score = get_prediction_scores(model, X_phiusiil)

        results.append(
            evaluate_binary_classifier(
                model_name=model_name,
                experiment_name="Cross-dataset transfer: original -> PhiUSIIL",
                y_true=y_phiusiil,
                y_pred=y_pred,
                y_score=y_score,
            )
        )

    results_df = pd.DataFrame(results)

    metric_columns = [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "MCC",
        "ROC-AUC",
        "PR-AUC",
    ]

    print("\nExternal validation results:")
    print(
        results_df[
            ["Experiment", "Model", "N Test", *metric_columns, "TN", "FP", "FN", "TP"]
        ].round(4)
    )

    results_df.to_csv(
        RESULTS_DIR / "phiusiil_external_validation_results.csv",
        index=False,
    )

    # Report-friendly compact version
    compact_results = results_df[
        ["Experiment", "Model", *metric_columns]
    ].copy()

    compact_results.to_csv(
        RESULTS_DIR / "phiusiil_external_validation_results_for_report.csv",
        index=False,
    )

    # Plot MCC and balanced accuracy comparison.
    plot_df = results_df.copy()
    plot_df["Short Model"] = [
        "PhiUSIIL XGB",
        "PhiUSIIL RF",
        "Transfer XGB",
        "Transfer RF",
    ]

    for metric in ["Balanced Accuracy", "MCC"]:
        plt.figure(figsize=(9, 5))
        plt.bar(plot_df["Short Model"], plot_df[metric])
        plt.ylabel(metric)
        plt.title(f"External Validation: {metric}")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        output_path = FIGURES_DIR / f"phiusiil_external_validation_{metric.lower().replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved figure: {output_path}")

    print("\nSaved results:")
    print(f"  - {RESULTS_DIR / 'phiusiil_class_balance.csv'}")
    print(f"  - {RESULTS_DIR / 'phiusiil_url_only_features.csv'}")
    print(f"  - {RESULTS_DIR / 'phiusiil_external_validation_results.csv'}")
    print(f"  - {RESULTS_DIR / 'phiusiil_external_validation_results_for_report.csv'}")

    print("=" * 80)
    print("External validation completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()

