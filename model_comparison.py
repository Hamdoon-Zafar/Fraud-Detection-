import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score,
    roc_curve, precision_recall_curve,
)
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not installed. Run: pip install xgboost")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PLOTS_DIR = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

C = {"rf": "#2E75B6", "xgb": "#C00000", "lr": "#7030A0"}


def load_and_preprocess(path: str):
    p = Path(path)
    if not p.exists():
        log.error(f"Dataset not found: {p}")
        sys.exit(1)

    df = pd.read_csv(p)
    df.drop_duplicates(inplace=True)

    scaler = RobustScaler()
    df["scaled_Amount"] = scaler.fit_transform(df[["Amount"]])
    df["scaled_Time"]   = scaler.fit_transform(df[["Time"]])
    df.drop(columns=["Amount", "Time"], inplace=True)

    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def train_evaluate(name: str, model, X_train_res, y_train_res,
                   X_test, y_test) -> dict:
    log.info(f"Training {name}...")
    t0 = time.time()
    model.fit(X_train_res, y_train_res)
    train_time = time.time() - t0

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "name":          name,
        "model":         model,
        "y_proba":       y_proba,
        "train_time_s":  round(train_time, 2),
        "precision":     round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":        round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":            round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":       round(roc_auc_score(y_test, y_proba), 4),
        "avg_precision": round(average_precision_score(y_test, y_proba), 4),
    }


def plot_roc_comparison(results: list, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = list(C.values())

    for res, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, lw=2.5, color=color,
                label=f"{res['name']}  (AUC = {res['roc_auc']:.4f})")

    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1.5, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison — All Models", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "11_roc_comparison.png")
    plt.close()
    log.info("Saved: 11_roc_comparison.png")


def plot_pr_comparison(results: list, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = list(C.values())

    for res, color in zip(results, colors):
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        ax.plot(rec, prec, lw=2.5, color=color,
                label=f"{res['name']}  (AP = {res['avg_precision']:.4f})")

    ax.axhline(y=y_test.mean(), color="grey", linestyle="--",
               label=f"Baseline (fraud rate = {y_test.mean():.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve Comparison — All Models", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "12_pr_comparison.png")
    plt.close()
    log.info("Saved: 12_pr_comparison.png")


def plot_metric_bar_comparison(results: list):
    metrics = ["precision", "recall", "f1", "roc_auc", "avg_precision"]
    labels  = ["Precision", "Recall", "F1-Score", "ROC-AUC", "Avg Precision"]
    names   = [r["name"] for r in results]
    colors  = list(C.values())[:len(results)]

    x     = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (res, color) in enumerate(zip(results, colors)):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width,
                      label=res["name"], color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim([0, 1.12])
    ax.set_title("Model Comparison — All Evaluation Metrics", fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "13_metric_comparison.png")
    plt.close()
    log.info("Saved: 13_metric_comparison.png")


def print_comparison_table(results: list):
    df = pd.DataFrame([{
        "Model":           r["name"],
        "Precision":       r["precision"],
        "Recall":          r["recall"],
        "F1-Score":        r["f1"],
        "ROC-AUC":         r["roc_auc"],
        "Avg Precision":   r["avg_precision"],
        "Train Time (s)":  r["train_time_s"],
    } for r in results])

    df.set_index("Model", inplace=True)
    print("\n" + "="*75)
    print("         MODEL COMPARISON TABLE")
    print("="*75)
    print(df.to_string())
    print("="*75 + "\n")

    csv_path = Path("outputs/reports/model_comparison.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)
    log.info(f"Comparison table saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Model Comparison — Fraud Detection")
    parser.add_argument("--data", default="creditcard.csv")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("   MODEL COMPARISON — Credit Card Fraud Detection")
    print("="*65 + "\n")

    X, y = load_and_preprocess(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    log.info("Applying SMOTE to training set...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    log.info(f"After SMOTE: {dict(pd.Series(y_res).value_counts())}")

    # Define models
    models = [
        ("Random Forest", RandomForestClassifier(
            n_estimators=100, max_depth=12,
            class_weight="balanced", n_jobs=-1, random_state=42
        )),
        ("Logistic Regression", LogisticRegression(
            C=0.01, class_weight="balanced",
            max_iter=1000, random_state=42
        )),
    ]

    if XGBOOST_AVAILABLE:
        scale_pos = int((y_res == 0).sum() / (y_res == 1).sum())
        models.insert(1, ("XGBoost", XGBClassifier(
            n_estimators=100, max_depth=6,
            learning_rate=0.1, scale_pos_weight=scale_pos,
            use_label_encoder=False, eval_metric="logloss",
            n_jobs=-1, random_state=42
        )))
    else:
        log.warning("XGBoost skipped (not installed). Run: pip install xgboost")

    # Train & evaluate all
    results = []
    for name, model in models:
        res = train_evaluate(name, model, X_res, y_res, X_test, y_test)
        results.append(res)

    # Output
    print_comparison_table(results)
    plot_roc_comparison(results, y_test)
    plot_pr_comparison(results, y_test)
    plot_metric_bar_comparison(results)

    best = max(results, key=lambda r: r["f1"])
    print(f"Best model by F1-Score: {best['name']} (F1 = {best['f1']:.4f})")
    print(f"   Plots saved to: {PLOTS_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
