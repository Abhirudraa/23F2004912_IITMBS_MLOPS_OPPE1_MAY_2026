import os
import argparse
import itertools

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument(
    "--experiment_name",
    default="stock_prediction_rf"
)
args = parser.parse_args()


# -------------------------------
# MLflow setup
# -------------------------------
mlflow.set_experiment(args.experiment_name)


# -------------------------------
# Load data
# -------------------------------
all_data = []

for file in os.listdir(args.data_dir):

    if not file.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(args.data_dir, file))

    df["rolling_avg_10"] = (
        df["close"]
        .rolling(10, min_periods=1)
        .mean()
    )

    df["volume_sum_10"] = (
        df["volume"]
        .rolling(10, min_periods=1)
        .sum()
    )

    df["target"] = (
        df["close"].shift(-5) > df["close"]
    ).astype(int)

    df = df.iloc[:-5]

    all_data.append(
        df[
            [
                "rolling_avg_10",
                "volume_sum_10",
                "target",
            ]
        ]
    )

dataset = pd.concat(all_data, ignore_index=True)

# ---------------------------------------
# SAMPLE ONLY 20k ROWS (FAST)
# ---------------------------------------
dataset = dataset.sample(
    n=min(20000, len(dataset)),
    random_state=42,
)

X = dataset[
    [
        "rolling_avg_10",
        "volume_sum_10",
    ]
]

y = dataset["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# ---------------------------------------
# Small search space
# ---------------------------------------
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [4, 6],
}


best_model = None
best_f1 = -1
best_run_id = None


for n_estimators, max_depth in itertools.product(
    param_grid["n_estimators"],
    param_grid["max_depth"],
):

    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        mlflow.log_param(
            "n_estimators",
            n_estimators,
        )

        mlflow.log_param(
            "max_depth",
            max_depth,
        )

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "precision",
            precision,
        )

        mlflow.log_metric(
            "recall",
            recall,
        )

        mlflow.log_metric(
            "f1",
            f1,
        )

        mlflow.sklearn.log_model(
            model,
            name="model",
        )

        print(
            f"n_estimators={n_estimators}, "
            f"max_depth={max_depth}, "
            f"F1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_run_id = mlflow.active_run().info.run_id


print("\n==========================")
print("Best F1 :", best_f1)
print("Run ID  :", best_run_id)
print("==========================")

os.makedirs("models", exist_ok=True)

best_path = "models/best_model.joblib"

joblib.dump(best_model, best_path)

with mlflow.start_run(run_name="best_model_registration"):

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="best_model",
        registered_model_name="stock_rf",
    )

print("\nRegistered model as: stock_rf")
