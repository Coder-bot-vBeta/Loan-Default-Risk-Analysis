# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import sqlite3
from ucimlrepo import fetch_ucirepo

# -----------------------------------------
# 1) Load UCI dataset (Python)
# -----------------------------------------
ucr = fetch_ucirepo(id=350)  # Default of Credit Card Clients
X = ucr.data.features.copy()     # DataFrame
y = ucr.data.targets.copy()      # single-col DataFrame

TARGET_OLD = 'default payment next month'
TARGET_NEW = 'default_payment_next_month'
if TARGET_OLD in y.columns:
    y = y.rename(columns={TARGET_OLD: TARGET_NEW})
elif TARGET_NEW not in y.columns:
    y.columns = [TARGET_NEW]

df = pd.concat([X, y], axis=1)

# -----------------------------------------
# 2) Persist to SQLite and engineer features in SQL
# -----------------------------------------
conn = sqlite3.connect(":memory:")  # use "credit.db" if you want a file
df.to_sql("credit_raw", conn, index=False, if_exists="replace")

# SQL feature engineering:
# - late_cnt_6m: count of months with delay (PAY_* >= 1)
# - avg_pay_delay: average of PAY_* (higher => more delayed)
# - total_bill_6m, total_pay_6m
# - payment_ratio_6m: total_pay_6m / total_bill_6m
# - avg_utilization_6m: average(BILL_AMT*/LIMIT_BAL)
# - recent_overdue_flag: 1 if most recent PAY_0 >= 1 else 0
# - age_bucket: simple bucketing for downstream viz (optional)
sql = """
WITH base AS (
    SELECT
        *,
        -- Count months with any delay (1+ months overdue)
        (CASE WHEN PAY_0 >= 1 THEN 1 ELSE 0 END +
         CASE WHEN PAY_2 >= 1 THEN 1 ELSE 0 END +
         CASE WHEN PAY_3 >= 1 THEN 1 ELSE 0 END +
         CASE WHEN PAY_4 >= 1 THEN 1 ELSE 0 END +
         CASE WHEN PAY_5 >= 1 THEN 1 ELSE 0 END +
         CASE WHEN PAY_6 >= 1 THEN 1 ELSE 0 END) AS late_cnt_6m,

        -- Average delay score across 6 months (PAY_*: -2..8; higher => worse)
        (PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6) * 1.0 / 6.0 AS avg_pay_delay,

        -- Sums across 6 months
        (BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6) AS total_bill_6m,
        (PAY_AMT1  + PAY_AMT2  + PAY_AMT3  + PAY_AMT4  + PAY_AMT5  + PAY_AMT6) AS total_pay_6m,

        -- Most recent overdue flag (PAY_0)
        CASE WHEN PAY_0 >= 1 THEN 1 ELSE 0 END AS recent_overdue_flag
    FROM credit_raw
),
util AS (
    SELECT
        *,
        -- Average utilization across 6 months: BILL / LIMIT
        (
            (BILL_AMT1 / NULLIF(LIMIT_BAL, 0.0)) +
            (BILL_AMT2 / NULLIF(LIMIT_BAL, 0.0)) +
            (BILL_AMT3 / NULLIF(LIMIT_BAL, 0.0)) +
            (BILL_AMT4 / NULLIF(LIMIT_BAL, 0.0)) +
            (BILL_AMT5 / NULLIF(LIMIT_BAL, 0.0)) +
            (BILL_AMT6 / NULLIF(LIMIT_BAL, 0.0))
        ) / 6.0 AS avg_utilization_6m
    FROM base
),
final AS (
    SELECT
        LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
        PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
        BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
        PAY_AMT1,  PAY_AMT2,  PAY_AMT3,  PAY_AMT4,  PAY_AMT5,  PAY_AMT6,

        late_cnt_6m,
        avg_pay_delay,
        total_bill_6m,
        total_pay_6m,
        CASE WHEN total_bill_6m = 0 THEN 0.0 ELSE total_pay_6m * 1.0 / total_bill_6m END AS payment_ratio_6m,
        avg_utilization_6m,
        recent_overdue_flag,

        -- Optional categorical bucketing for viz
        CASE
            WHEN AGE < 25 THEN '18-24'
            WHEN AGE BETWEEN 25 AND 34 THEN '25-34'
            WHEN AGE BETWEEN 35 AND 44 THEN '35-44'
            WHEN AGE BETWEEN 45 AND 54 THEN '45-54'
            WHEN AGE >= 55 THEN '55+'
            ELSE 'Unknown'
        END AS age_bucket,

        -- Target
        "default payment next month" AS default_payment_next_month
    FROM util
)
SELECT * FROM final;
"""

features_sql = pd.read_sql_query(sql, conn)

# -----------------------------------------
# 3) Quick EDA (optional)
# -----------------------------------------
plt.figure()
features_sql["default_payment_next_month"].value_counts().sort_index().plot(kind='bar')
plt.xticks([0,1], ['No Default (0)', 'Default (1)'])
plt.title("Loan Default Distribution")
plt.xlabel("Class"); plt.ylabel("Count")
plt.tight_layout(); plt.show()

# -----------------------------------------
# 4) Train / Test + Modeling (Python)
# -----------------------------------------
TARGET = "default_payment_next_month"
X_features = features_sql.drop(columns=[TARGET])
y_target = features_sql[TARGET].astype(int).values.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_target, test_size=0.30, random_state=42, stratify=y_target
)

# Scale numeric columns (good for linear models)
num_cols = X_features.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])

# Logistic Regression (baseline)
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train_scaled, y_train)
y_proba_lr = logreg.predict_proba(X_test_scaled)[:, 1]
y_pred_lr  = (y_proba_lr >= 0.5).astype(int)

print("\n=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr, digits=4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba_lr), 4))

# Try XGBoost; if unavailable, fall back to RandomForest
advanced_name = None
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", n_jobs=-1, random_state=42, tree_method="hist"
    )
    xgb.fit(X_train, y_train)  # tree models don't require scaling
    y_proba_adv = xgb.predict_proba(X_test)[:, 1]
    y_pred_adv  = (y_proba_adv >= 0.5).astype(int)
    advanced_name = "XGBoost"

    print("\n=== XGBoost ===")
    print(classification_report(y_test, y_pred_adv, digits=4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba_adv), 4))

    importances = pd.Series(xgb.feature_importances_, index=X_features.columns).sort_values(ascending=True)
except Exception as e:
    print("\n[Info] XGBoost not available or failed. Falling back to RandomForest. Reason:", e)
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_proba_adv = rf.predict_proba(X_test)[:, 1]
    y_pred_adv  = (y_proba_adv >= 0.5).astype(int)
    advanced_name = "RandomForest"

    print("\n=== Random Forest ===")
    print(classification_report(y_test, y_pred_adv, digits=4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba_adv), 4))

    importances = pd.Series(rf.feature_importances_, index=X_features.columns).sort_values(ascending=True)

# Feature importance (top 10)
plt.figure(figsize=(8,6))
importances.tail(10).plot(kind="barh")
plt.title(f"Top 10 Features Affecting Default ({advanced_name})")
plt.tight_layout(); plt.show()

# ROC Curve
plt.figure()
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
plt.plot(fpr_lr, tpr_lr, label="LogReg")
fpr_adv, tpr_adv, _ = roc_curve(y_test, y_proba_adv)
plt.plot(fpr_adv, tpr_adv, label=advanced_name)
plt.plot([0,1], [0,1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(); plt.tight_layout(); plt.show()

# -----------------------------------------
# 5) (Optional) Export for BI tools
# -----------------------------------------
features_sql.to_csv("loan_default_features.csv", index=False)
print("\nExported engineered dataset for BI:", "loan_default_features.csv")
