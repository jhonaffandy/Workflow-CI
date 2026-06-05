"""
modelling.py  (MLProject version)
Accepts CLI arguments so it can be launched via `mlflow run`.
"""

import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_path",    type=str, default="titanic_preprocessing/titanic_preprocessing.csv")
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth",    type=int, default=6)
args = parser.parse_args()

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(args.data_path)
X  = df.drop(columns=["Survived"])
y  = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── MLflow ────────────────────────────────────────────────────────────────────
mlflow.set_experiment("Titanic_RandomForest_CI")
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="ci_run"):

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds    = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"\nAccuracy : {accuracy:.4f}")
    print(classification_report(y_test, preds))

print("\n[modelling] Run complete. Artifacts saved to mlruns/")
