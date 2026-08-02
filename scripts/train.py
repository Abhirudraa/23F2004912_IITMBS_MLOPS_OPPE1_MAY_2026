import os
import argparse
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--model_out", required=True)
args = parser.parse_args()

all_data = []

for file in os.listdir(args.data_dir):
    if not file.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(args.data_dir, file))

    df["rolling_avg_10"] = (
        df["close"]
        .rolling(window=10, min_periods=1)
        .mean()
    )

    df["volume_sum_10"] = (
        df["volume"]
        .rolling(window=10, min_periods=1)
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

X = dataset[["rolling_avg_10", "volume_sum_10"]]
y = dataset["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nEvaluation Metrics")
print("------------------")
print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall   :", recall_score(y_test, pred))
print("F1 Score :", f1_score(y_test, pred))

os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
joblib.dump(model, args.model_out)

print(f"\nModel saved to {args.model_out}")
