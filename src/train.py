import os, json, joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

os.makedirs("model_building", exist_ok=True)
os.makedirs("deployment", exist_ok=True)

df = pd.read_csv("data/tourism.csv")
df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
X = df.drop(columns=["ProdTaken", "CustomerID"])
y = df["ProdTaken"]

categorical = X.select_dtypes(include="object").columns.tolist()
numeric = X.select_dtypes(exclude="object").columns.tolist()

def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

if os.path.exists("artifacts/train.csv") and os.path.exists("artifacts/test.csv"):
    train_df = pd.read_csv("artifacts/train.csv")
    test_df = pd.read_csv("artifacts/test.csv")
    X_train = train_df.drop(columns=["ProdTaken", "CustomerID"])
    y_train = train_df["ProdTaken"]
    X_test = test_df.drop(columns=["ProdTaken", "CustomerID"])
    y_test = test_df["ProdTaken"]
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

models = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    )
}

if MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("tourism_package_purchase")

results = []
trained = {}

for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            prob = pipe.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, prob)
            }
            mlflow.log_params(estimator.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, "model")
    else:
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob)
        }
    results.append({"model": name, **metrics})
    trained[name] = pipe

results_df = pd.DataFrame(results).sort_values(["roc_auc", "f1"], ascending=False)
results_df.to_csv("model_building/experiment_results.csv", index=False)

best_name = results_df.iloc[0]["model"]
best_model = trained[best_name]
joblib.dump(best_model, "deployment/model.joblib")
joblib.dump(best_model, "model_building/best_model.joblib")

metadata = {
    "model_name": best_name,
    "target": "ProdTaken",
    "features": X.columns.tolist(),
    "categorical_features": categorical,
    "numeric_features": numeric,
    "random_state": 42
}
json.dump(metadata, open("deployment/model_metadata.json", "w"), indent=2)

print(results_df.to_string(index=False))
print(f"\nSelected model: {best_name}")
