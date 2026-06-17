# Credit Card Fraud Detection — Task 2

## Project Overview

This project builds a production-ready machine learning pipeline to classify credit card transactions as **legitimate (0)** or **fraudulent (1)**. The core challenge is the severe class imbalance — only **0.172%** of transactions are fraudulent — which requires careful handling to avoid models that simply predict "legitimate" for everything.

### Techniques Demonstrated
- **RobustScaler** for outlier-resistant feature normalization
- **Stratified Train/Test Split** to preserve class ratios
- **SMOTE** (Synthetic Minority Over-sampling Technique) for imbalance correction
- **Random Forest** classification with `class_weight='balanced'`
- **Proper evaluation** via Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC
- **Threshold analysis** to find the optimal decision boundary
- **Multi-model comparison** (RF vs XGBoost vs Logistic Regression)

---

## Repository Structure

```
task2_fraud_detection/
│
├── fraud_detection_pipeline.py   # Complete end-to-end ML pipeline (main script)
├── eda.py                        # Standalone EDA module with 4 detailed plots
├── model_comparison.py           # RF vs XGBoost vs Logistic Regression comparison
├── requirements.txt              # All Python dependencies
├── .gitignore                    # Excludes dataset and outputs
└── README.md                     # This file
```

### Generated Outputs (after running)
```
outputs/
├── plots/
│   ├── 01_class_distribution.png
│   ├── 02_amount_distribution.png
│   ├── 03_time_distribution.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_feature_target_correlation.png
│   ├── 06_confusion_matrix.png
│   ├── 07_roc_curve.png
│   ├── 08_precision_recall_curve.png
│   ├── 09_feature_importance.png
│   ├── 10_threshold_analysis.png
│   ├── 11_roc_comparison.png       ← model_comparison.py
│   ├── 12_pr_comparison.png        ← model_comparison.py
│   └── 13_metric_comparison.png    ← model_comparison.py
├── reports/
│   ├── fraud_detection_report_<timestamp>.txt
│   ├── metrics.json
│   ├── model_comparison.csv
│   └── eda_summary_statistics.csv  ← eda.py
```

---


### Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

###  Download the dataset

**Option A — Manual:**
1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click **Download** → extract `creditcard.csv` to this directory

**Option B — Kaggle CLI:**
```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip
```

---

##  Running the Scripts

### Script 1: Full ML Pipeline (recommended starting point)
```bash
python fraud_detection_pipeline.py
```

Optional flags:
```bash
python fraud_detection_pipeline.py --data /path/to/creditcard.csv
python fraud_detection_pipeline.py --test-size 0.25
python fraud_detection_pipeline.py --skip-eda     # Faster, skips EDA plots
```

**What it does:**
1. Loads and validates `creditcard.csv`
2. Runs EDA → saves 5 visualisation plots
3. Scales `Amount` and `Time` with `RobustScaler`
4. Splits data (80% train / 20% test, stratified)
5. Applies SMOTE to training set only
6. Trains Random Forest (100 trees, max_depth=12)
7. Evaluates on original imbalanced test set
8. Saves 5 evaluation plots + JSON metrics + text report

---

### Script 2: Standalone EDA
```bash
python eda.py
```
Generates 4 deep-dive EDA visualisations without training a model.

---

### Script 3: Model Comparison
```bash
pip install xgboost   # Optional but recommended
python model_comparison.py
```
Trains Random Forest, XGBoost, and Logistic Regression on the same data and produces side-by-side metric comparison plots.

---

## Expected Results

| Metric | Value (approx.) |
|--------|----------------|
| Precision (Fraud) | ~0.92 |
| Recall (Fraud) | ~0.87 |
| F1-Score (Fraud) | ~0.89 |
| ROC-AUC | ~0.98 |
| Average Precision | ~0.82 |

> *Exact values vary slightly by environment due to SMOTE's synthetic sample generation.*

---

##  Why These Metrics Matter

```
CONFUSION MATRIX
                  Predicted Legitimate  Predicted Fraud
Actual Legitimate       TN                FP
Actual Fraud            FN                 TP

FN (False Negative) = Missed fraud → FINANCIAL LOSS
FP (False Positive) = False alarm  → CUSTOMER FRICTION only

Therefore: RECALL must be maximised even at some cost to Precision.
```

**Why not just use Accuracy?**  
A model predicting "Legitimate" for every single transaction scores **99.83% accuracy** but catches **0% of fraud**. Accuracy is meaningless for imbalanced classification.

---

## Pipeline Architecture

```
creditcard.csv
      │
      ▼
┌─────────────────────────┐
│   Data Validation       │  Check shape, columns, nulls, duplicates
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   EDA & Visualization   │  5 plots: class balance, amounts, time, correlations
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   RobustScaler          │  Scale Amount & Time (robust to outliers)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Stratified Split      │  80% train / 20% test (preserves class ratio)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   SMOTE (train only)    │  Synthetic fraud samples → balanced 50:50 train set
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Random Forest         │  n_estimators=100, max_depth=12, class_weight=balanced
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Evaluation            │  Test set remains ORIGINAL (imbalanced — realistic)
│   (original test set)   │  Confusion Matrix | ROC-AUC | F1 | PR Curve | Threshold
└─────────────────────────┘
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.5.0 | Data loading and manipulation |
| numpy | ≥1.23.0 | Numerical operations |
| scikit-learn | ≥1.2.0 | ML models, metrics, preprocessing |
| imbalanced-learn | ≥0.10.0 | SMOTE implementation |
| matplotlib | ≥3.6.0 | Visualisations |
| seaborn | ≥0.12.0 | Statistical plots |
| xgboost | ≥1.7.0 | Optional: model comparison |

---

##  Dataset Notice

The `creditcard.csv` file is **not included** in this repository (284 MB). It must be downloaded separately from Kaggle:  
 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

