import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",    type=str, default="titanic_preprocessing/titanic_preprocessing.csv")
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth",    type=int, default=6)
args = parser.parse_args()

df = pd.read_csv(args.data_path)
X  = df.drop(columns=["Survived"])
y  = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

mlflow.sklearn.autolog()

model = RandomForestClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    random_state=42
)
model.fit(X_train, y_train)

preds    = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy:.4f}")
