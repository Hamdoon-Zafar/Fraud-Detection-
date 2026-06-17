import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PLOTS_DIR = Path("outputs/plots/eda")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PLT_STYLE = {
    "figure.dpi": 150, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
}
plt.rcParams.update(PLT_STYLE)
C = {"legit": "#2E75B6", "fraud": "#C00000", "neutral": "#7030A0"}


def load(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        log.error(f"File not found: {p.resolve()}")
        sys.exit(1)
    df = pd.read_csv(p)
    log.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def plot_class_balance(df: pd.DataFrame):
    """Pie + bar side-by-side showing the severe class imbalance."""
    counts = df["Class"].value_counts()
    labels = ["Legitimate", "Fraud"]
    sizes  = [counts[0], counts[1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Pie
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels,
        colors=[C["legit"], C["fraud"]],
        autopct="%1.3f%%", startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    autotexts[1].set_fontsize(9)
    ax1.set_title("Proportion of Fraudulent Transactions", fontweight="bold")

    # Bar (log scale)
    bars = ax2.bar(labels, sizes, color=[C["legit"], C["fraud"]],
                   edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                 f"{val:,}", ha="center", fontweight="bold", fontsize=10)
    ax2.set_yscale("log")
    ax2.set_ylabel("Count (log scale)")
    ax2.set_title("Absolute Transaction Counts", fontweight="bold")

    plt.suptitle(f"Class Imbalance Analysis  |  Ratio ≈ {counts[0]//counts[1]:,}:1",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_01_class_imbalance.png", bbox_inches="tight")
    plt.close()
    log.info("Saved: eda_01_class_imbalance.png")


def plot_amount_boxplot(df: pd.DataFrame):
    """Boxplot comparing transaction amounts for fraud vs legitimate."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Full range
    df.boxplot(column="Amount", by="Class", ax=axes[0],
               patch_artist=True,
               boxprops=dict(facecolor=C["legit"], color=C["legit"]),
               medianprops=dict(color="white", linewidth=2))
    axes[0].set_title("Amount Distribution (Full Range)")
    axes[0].set_xlabel("Class  (0=Legitimate, 1=Fraud)")
    axes[0].set_ylabel("Amount (USD)")

    # Zoomed (cap at 99th percentile for readability)
    cap = df["Amount"].quantile(0.99)
    df_cap = df[df["Amount"] <= cap]
    for cls, color, label in [(0, C["legit"], "Legitimate"), (1, C["fraud"], "Fraud")]:
        data = df_cap[df_cap["Class"] == cls]["Amount"]
        axes[1].boxplot(data.values, positions=[cls],
                        patch_artist=True, widths=0.4,
                        boxprops=dict(facecolor=color, alpha=0.7),
                        medianprops=dict(color="white", linewidth=2))
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Legitimate", "Fraud"])
    axes[1].set_title(f"Amount Distribution (Capped at 99th Pct ≈ ${cap:.0f})")
    axes[1].set_ylabel("Amount (USD)")

    plt.suptitle("Transaction Amount: Legitimate vs. Fraud", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_02_amount_boxplot.png", bbox_inches="tight")
    plt.close()
    log.info("Saved: eda_02_amount_boxplot.png")


def plot_pca_features(df: pd.DataFrame):
    """Grid of density plots for V1–V28 features split by class."""
    v_cols = [c for c in df.columns if c.startswith("V")]
    fraud  = df[df["Class"] == 1]
    legit  = df[df["Class"] == 0].sample(n=min(5000, len(df[df["Class"]==0])),
                                          random_state=42)

    n_cols = 7
    n_rows = int(np.ceil(len(v_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 2.5))
    axes = axes.flatten()

    for i, col in enumerate(v_cols):
        ax = axes[i]
        ax.hist(legit[col], bins=40, color=C["legit"], alpha=0.6,
                density=True, label="Legit")
        ax.hist(fraud[col], bins=40, color=C["fraud"], alpha=0.7,
                density=True, label="Fraud")
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_yticks([])
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for j in range(len(v_cols), len(axes)):
        axes[j].set_visible(False)

    handles = [
        plt.Rectangle((0,0),1,1, fc=C["legit"], alpha=0.7, label="Legitimate (sample)"),
        plt.Rectangle((0,0),1,1, fc=C["fraud"], alpha=0.7, label="Fraud"),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=10, ncol=2)
    plt.suptitle("PCA Feature Distributions: Legitimate vs. Fraud (V1–V28)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_03_pca_feature_distributions.png", bbox_inches="tight")
    plt.close()
    log.info("Saved: eda_03_pca_feature_distributions.png")


def plot_fraud_by_hour(df: pd.DataFrame):
    """Fraud frequency heatmap by hour-of-day across two dataset days."""
    df = df.copy()
    df["hour"] = (df["Time"] / 3600).astype(int) % 24
    df["day"]  = ((df["Time"] / 3600) // 24).astype(int) + 1

    pivot = df[df["Class"] == 1].groupby(["day", "hour"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 3))
    sns.heatmap(pivot, cmap="Reds", ax=ax, linewidths=0.3,
                cbar_kws={"label": "Fraud Count"}, annot=True, fmt="d", annot_kws={"size": 7})
    ax.set_title("Fraud Frequency by Hour of Day", fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Dataset Day")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_04_fraud_hourly_heatmap.png", bbox_inches="tight")
    plt.close()
    log.info("Saved: eda_04_fraud_hourly_heatmap.png")


def summary_statistics(df: pd.DataFrame):
    """Print and save summary statistics."""
    log.info("\n── Summary Statistics ──")
    fraud = df[df["Class"] == 1]["Amount"]
    legit = df[df["Class"] == 0]["Amount"]

    stats = pd.DataFrame({
        "Legitimate": legit.describe(),
        "Fraud":      fraud.describe(),
    })
    log.info(f"\n{stats.to_string()}")

    stats_path = Path("outputs/reports/eda_summary_statistics.csv")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(stats_path)
    log.info(f"Stats saved: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="EDA Module — Credit Card Fraud")
    parser.add_argument("--data", default="creditcard.csv")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("   EDA MODULE — Credit Card Fraud Detection")
    print("="*60 + "\n")

    df = load(args.data)
    summary_statistics(df)
    plot_class_balance(df)
    plot_amount_boxplot(df)
    plot_pca_features(df)
    plot_fraud_by_hour(df)

    print(f"\n EDA complete. All plots saved to: {PLOTS_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
