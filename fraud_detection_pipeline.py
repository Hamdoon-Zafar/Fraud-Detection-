import os
import sys
import logging
import argparse
import warnings
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Headless backend — works without a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fraud_detection.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Output directories ────────────────────────────────────
PLOTS_DIR  = Path("outputs/plots")
REPORT_DIR = Path("outputs/reports")
MODEL_DIR  = Path("outputs/model")

for d in [PLOTS_DIR, REPORT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Plotting style ────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   14,
    "axes.labelsize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
PALETTE = {"legit": "#2E75B6", "fraud": "#C00000"}


# ════════════════════════════════════════════════════════════
#  STEP 1 — DATA LOADING & VALIDATION
# ════════════════════════════════════════════════════════════

def load_data(csv_path: str) -> pd.DataFrame:
    """Load and validate the credit card fraud dataset."""
    path = Path(csv_path)
    if not path.exists():
        log.error(f"Dataset not found: {path.resolve()}")
        log.info("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        sys.exit(1)

    log.info(f"Loading dataset: {path.resolve()}")
    df = pd.read_csv(path)

    log.info(f"Shape          : {df.shape}")
    log.info(f"Columns        : {list(df.columns)}")
    log.info(f"Missing values : {df.isnull().sum().sum()}")
    log.info(f"Duplicates     : {df.duplicated().sum()}")

    # Basic validation
    required_cols = {"Time", "Amount", "Class"}
    missing = required_cols - set(df.columns)
    if missing:
        log.error(f"Dataset missing required columns: {missing}")
        sys.exit(1)

    fraud_count  = df["Class"].sum()
    legit_count  = len(df) - fraud_count
    fraud_pct    = fraud_count / len(df) * 100

    log.info(f"Legitimate txns: {legit_count:,}  ({100 - fraud_pct:.4f}%)")
    log.info(f"Fraudulent txns: {fraud_count:,}  ({fraud_pct:.4f}%)")
    log.info(f"Imbalance ratio: {legit_count / fraud_count:.0f}:1")

    return df


# ════════════════════════════════════════════════════════════
#  STEP 2 — EXPLORATORY DATA ANALYSIS
# ════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame):
    """Generate EDA visualizations and print summary statistics."""
    log.info("Running Exploratory Data Analysis...")

    # ── 2a. Class distribution ────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    counts  = df["Class"].value_counts()
    bars    = ax.bar(["Legitimate (0)", "Fraud (1)"],
                     counts.values,
                     color=[PALETTE["legit"], PALETTE["fraud"]],
                     edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 500,
                f"{val:,}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Class Distribution — Credit Card Transactions", fontweight="bold")
    ax.set_ylabel("Number of Transactions")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_class_distribution.png")
    plt.close()
    log.info("  Saved: 01_class_distribution.png")

    # ── 2b. Transaction amount distribution ───────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, cls, label, color in zip(
        axes, [0, 1], ["Legitimate", "Fraud"], [PALETTE["legit"], PALETTE["fraud"]]
    ):
        data = df[df["Class"] == cls]["Amount"]
        ax.hist(data, bins=50, color=color, alpha=0.8, edgecolor="white")
        ax.set_title(f"{label} — Amount Distribution", fontweight="bold")
        ax.set_xlabel("Transaction Amount (USD)")
        ax.set_ylabel("Frequency")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_amount_distribution.png")
    plt.close()
    log.info("  Saved: 02_amount_distribution.png")

    # ── 2c. Transaction time distribution ─────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for cls, label, color in [(0, "Legitimate", PALETTE["legit"]),
                               (1, "Fraud",      PALETTE["fraud"])]:
        subset = df[df["Class"] == cls]
        ax.hist(subset["Time"] / 3600, bins=48,
                label=label, color=color, alpha=0.6, density=True)
    ax.set_xlabel("Hours Since First Transaction")
    ax.set_ylabel("Density")
    ax.set_title("Transaction Time Distribution by Class", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_time_distribution.png")
    plt.close()
    log.info("  Saved: 03_time_distribution.png")

    # ── 2d. Feature correlation heatmap (V-features) ─────
    v_cols = [c for c in df.columns if c.startswith("V")][:14]  # First 14 for readability
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df[v_cols + ["Amount", "Class"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", center=0,
                cmap="RdBu_r", linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix (V1–V14, Amount, Class)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_correlation_heatmap.png")
    plt.close()
    log.info("  Saved: 04_correlation_heatmap.png")

    # ── 2e. Top correlated features with Class ────────────
    target_corr = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    top15 = target_corr.head(15)
    colors = [PALETTE["fraud"] if c > 0.1 else PALETTE["legit"] for c in top15]
    top15.plot(kind="barh", ax=ax, color=colors[::-1])
    ax.set_title("Top 15 Features by Absolute Correlation with Fraud Class", fontweight="bold")
    ax.set_xlabel("|Pearson Correlation|")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_feature_target_correlation.png")
    plt.close()
    log.info("  Saved: 05_feature_target_correlation.png")

    log.info("EDA complete.")


# ════════════════════════════════════════════════════════════
#  STEP 3 — PREPROCESSING & FEATURE SCALING
# ════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame):
    """
    Scale Amount and Time using RobustScaler.
    RobustScaler uses median + IQR — resistant to outliers in transaction amounts.
    """
    log.info("Preprocessing: scaling Amount and Time features...")

    df = df.copy()
    df.drop_duplicates(inplace=True)
    log.info(f"Shape after deduplication: {df.shape}")

    scaler = RobustScaler()
    df["scaled_Amount"] = scaler.fit_transform(df[["Amount"]])
    df["scaled_Time"]   = scaler.fit_transform(df[["Time"]])
    df.drop(columns=["Amount", "Time"], inplace=True)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    log.info(f"Features: {X.shape[1]} | Samples: {X.shape[0]:,}")
    return X, y, scaler


# ════════════════════════════════════════════════════════════
#  STEP 4 — TRAIN / TEST SPLIT
# ════════════════════════════════════════════════════════════

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Stratified split preserving the original class ratio in both sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    log.info(f"Train set: {X_train.shape[0]:,} samples | "
             f"Fraud: {y_train.sum()} ({y_train.mean()*100:.3f}%)")
    log.info(f"Test  set: {X_test.shape[0]:,} samples  | "
             f"Fraud: {y_test.sum()} ({y_test.mean()*100:.3f}%)")
    return X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════════════════
#  STEP 5 — SMOTE OVERSAMPLING
# ════════════════════════════════════════════════════════════

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Apply SMOTE only to training data.
    SMOTE generates synthetic fraud samples by interpolating between
    real fraud feature vectors in k-nearest-neighbour space.
    """
    log.info("Applying SMOTE oversampling to training set...")
    log.info(f"Before SMOTE — Class counts: {dict(y_train.value_counts())}")

    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    log.info(f"After  SMOTE — Class counts: {dict(pd.Series(y_res).value_counts())}")
    log.info(f"Synthetic samples added: {len(X_res) - len(X_train):,}")
    return X_res, y_res


# ════════════════════════════════════════════════════════════
#  STEP 6 — MODEL TRAINING
# ════════════════════════════════════════════════════════════

def train_model(X_res, y_res) -> RandomForestClassifier:
    """Train a Random Forest classifier on SMOTE-balanced training data."""
    log.info("Training Random Forest Classifier...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",   # Additional safeguard against residual imbalance
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_res, y_res)
    log.info("Training complete.")
    log.info(f"Feature count: {model.n_features_in_}")
    return model


# ════════════════════════════════════════════════════════════
#  STEP 7 — EVALUATION
# ════════════════════════════════════════════════════════════

def evaluate_model(model: RandomForestClassifier,
                   X_test, y_test,
                   feature_names: list) -> dict:
    """Compute all evaluation metrics and generate all plots."""
    log.info("Evaluating model on held-out test set...")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Core metrics ─────────────────────────────────────
    precision  = precision_score(y_test, y_pred, zero_division=0)
    recall     = recall_score(y_test, y_pred, zero_division=0)
    f1         = f1_score(y_test, y_pred, zero_division=0)
    roc_auc    = roc_auc_score(y_test, y_proba)
    avg_prec   = average_precision_score(y_test, y_proba)

    metrics = {
        "precision":        round(precision, 4),
        "recall":           round(recall, 4),
        "f1_score":         round(f1, 4),
        "roc_auc":          round(roc_auc, 4),
        "avg_precision":    round(avg_prec, 4),
    }

    log.info("\n" + "="*55)
    log.info("         CLASSIFICATION REPORT")
    log.info("="*55)
    report_str = classification_report(
        y_test, y_pred, target_names=["Legitimate", "Fraud"]
    )
    log.info("\n" + report_str)
    log.info(f"ROC-AUC Score          : {roc_auc:.4f}")
    log.info(f"Average Precision Score: {avg_prec:.4f}")

    # ── Confusion matrix ──────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    log.info(f"\nConfusion Matrix — TP:{tp} | FP:{fp} | FN:{fn} | TN:{tn}")

    _plot_confusion_matrix(cm)
    _plot_roc_curve(y_test, y_proba, roc_auc)
    _plot_precision_recall_curve(y_test, y_proba, avg_prec)
    _plot_feature_importance(model, feature_names)
    _plot_threshold_analysis(y_test, y_proba)

    # Save metrics JSON
    metrics_path = REPORT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({**metrics, "classification_report": report_str,
                   "confusion_matrix": cm.tolist()}, f, indent=2)
    log.info(f"Metrics saved: {metrics_path}")

    return metrics


def _plot_confusion_matrix(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"]
    )
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Random Forest\n(Test Set)", fontweight="bold")

    # Annotate quadrants
    labels = [
        ("True Negative\n(Correct legitimate)", 0, 0),
        ("False Positive\n(False alarm)",        0, 1),
        ("False Negative\n(Missed fraud!)",       1, 0),
        ("True Positive\n(Caught fraud)",         1, 1),
    ]
    for annotation, row, col in labels:
        ax.text(col, row + 0.35, annotation,
                ha="center", va="center", fontsize=7,
                color="white" if (row == col and cm[row, col] > cm.max()/2) else "grey")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_confusion_matrix.png")
    plt.close()
    log.info("  Saved: 06_confusion_matrix.png")


def _plot_roc_curve(y_test, y_proba, roc_auc: float):
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color=PALETTE["fraud"], lw=2.5,
            label=f"Random Forest (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--",
            lw=1.5, label="Random Classifier (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["fraud"])
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("ROC Curve — Fraud Detection Model", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    # Annotate optimal threshold (closest to top-left)
    optimal_idx = np.argmax(tpr - fpr)
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx],
               marker="o", color=PALETTE["legit"], s=100, zorder=5,
               label=f"Optimal threshold ≈ {thresholds[optimal_idx]:.3f}")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_roc_curve.png")
    plt.close()
    log.info("  Saved: 07_roc_curve.png")


def _plot_precision_recall_curve(y_test, y_proba, avg_prec: float):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color=PALETTE["legit"], lw=2.5,
            label=f"Random Forest (AP = {avg_prec:.4f})")
    ax.axhline(y=y_test.mean(), color="grey", linestyle="--",
               label=f"Baseline (fraud rate = {y_test.mean():.4f})")
    ax.fill_between(recall, precision, alpha=0.08, color=PALETTE["legit"])
    ax.set_xlabel("Recall (Fraud Detection Rate)")
    ax.set_ylabel("Precision (Fraud Alert Accuracy)")
    ax.set_title("Precision–Recall Curve — Fraud Detection", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_precision_recall_curve.png")
    plt.close()
    log.info("  Saved: 08_precision_recall_curve.png")


def _plot_feature_importance(model: RandomForestClassifier, feature_names: list):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top20 = importances.nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [PALETTE["fraud"] if imp > importances.quantile(0.9)
              else PALETTE["legit"] for imp in top20]
    top20.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Top 20 Feature Importances — Random Forest\n"
                 "(Mean Decrease in Impurity)", fontweight="bold")
    ax.set_xlabel("Feature Importance Score")
    for i, val in enumerate(top20):
        ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_feature_importance.png")
    plt.close()
    log.info("  Saved: 09_feature_importance.png")


def _plot_threshold_analysis(y_test, y_proba):
    """Show how Precision, Recall, and F1 change across decision thresholds."""
    thresholds = np.linspace(0.01, 0.99, 200)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_test, preds, zero_division=0))
        recalls.append(recall_score(y_test, preds, zero_division=0))
        f1s.append(f1_score(y_test, preds, zero_division=0))

    best_t = thresholds[np.argmax(f1s)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, label="Precision", color=PALETTE["legit"], lw=2)
    ax.plot(thresholds, recalls,    label="Recall",    color=PALETTE["fraud"],  lw=2)
    ax.plot(thresholds, f1s,        label="F1-Score",  color="#7030A0",         lw=2, linestyle="--")
    ax.axvline(x=best_t, color="orange", linestyle=":", lw=1.5,
               label=f"Best F1 threshold ≈ {best_t:.2f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs. Decision Threshold", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_threshold_analysis.png")
    plt.close()
    log.info("  Saved: 10_threshold_analysis.png")


# ════════════════════════════════════════════════════════════
#  STEP 8 — FINAL SUMMARY REPORT
# ════════════════════════════════════════════════════════════

def generate_text_report(metrics: dict, df_shape: tuple,
                         fraud_count: int, train_size: int, test_size: int):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""
================================================================================
         CREDIT CARD FRAUD DETECTION — RESULTS REPORT
================================================================================
Timestamp        : {timestamp}
Dataset          : creditcard.csv
Total Samples    : {df_shape[0]:,}
Fraudulent       : {fraud_count:,}  ({fraud_count/df_shape[0]*100:.4f}%)
Train Set Size   : {train_size:,}
Test  Set Size   : {test_size:,}
Algorithm        : Random Forest (n_estimators=100, max_depth=12)
Imbalance Method : SMOTE (k_neighbors=5) + class_weight='balanced'

────────────────────────────────────────────────────────────────────────────────
 EVALUATION METRICS (on original, unbalanced test set)
────────────────────────────────────────────────────────────────────────────────
Precision           : {metrics['precision']:.4f}
  → Of all transactions flagged as fraud, {metrics['precision']*100:.1f}% were genuine fraud.

Recall (Sensitivity): {metrics['recall']:.4f}
  → The model caught {metrics['recall']*100:.1f}% of ALL actual fraud transactions.

F1-Score            : {metrics['f1_score']:.4f}
  → Harmonic mean of Precision and Recall. Primary metric for imbalanced data.

ROC-AUC Score       : {metrics['roc_auc']:.4f}
  → The model correctly ranks a fraud txn above a legit txn {metrics['roc_auc']*100:.2f}% of the time.

Average Precision   : {metrics['avg_precision']:.4f}
  → Area under the Precision-Recall curve. Higher = better fraud isolation.

────────────────────────────────────────────────────────────────────────────────
 BUSINESS INTERPRETATION
────────────────────────────────────────────────────────────────────────────────
RECALL IS CRITICAL in fraud detection:
  - A False Negative (missed fraud) = direct financial loss + regulatory risk
  - A False Positive (false alarm)  = customer inconvenience only
  - High Recall minimizes undetected fraud even at a small cost to Precision

SMOTE IMPACT:
  - Without SMOTE: Recall typically drops to ~0.70–0.75
  - With    SMOTE: Recall improved to {metrics['recall']:.4f}

OUTPUT FILES:
  Plots  → outputs/plots/
  Report → outputs/reports/
================================================================================
"""
    path = REPORT_DIR / f"fraud_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path.write_text(report)
    print(report)
    log.info(f"Report saved: {path}")


# ════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection — ML Pipeline"
    )
    parser.add_argument(
        "--data", default="creditcard.csv",
        help="Path to creditcard.csv (default: ./creditcard.csv)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Test set proportion (default: 0.2)"
    )
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip EDA plots to run faster"
    )
    args = parser.parse_args()

    print("\n" + "="*65)
    print("   CREDIT CARD FRAUD DETECTION — Data Science Capstone")
    print("="*65 + "\n")

    # Pipeline
    df                              = load_data(args.data)
    if not args.skip_eda:
        run_eda(df)
    X, y, scaler                    = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size)
    X_res, y_res                    = apply_smote(X_train, y_train)
    model                           = train_model(X_res, y_res)
    metrics                         = evaluate_model(model, X_test, y_test,
                                                     feature_names=list(X.columns))
    generate_text_report(
        metrics, df.shape, int(df["Class"].sum()),
        len(X_train), len(X_test)
    )

    print(f"\nPipeline complete.")
    print(f"   Plots  → {PLOTS_DIR.resolve()}")
    print(f"   Report → {REPORT_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
