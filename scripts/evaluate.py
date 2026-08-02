import os
import mlflow
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ------------------------------------
# Load registered model
# ------------------------------------
model = mlflow.pyfunc.load_model(
    model_uri="models:/stock_rf/latest"
)

# ------------------------------------
# Load DVC pulled test data
# ------------------------------------
all_data = []

for file in os.listdir("data/merged"):

    if file.endswith(".csv"):

        df = pd.read_csv(os.path.join("data/merged", file))

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
            df["close"].shift(-5)
            > df["close"]
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

X = dataset[
    [
        "rolling_avg_10",
        "volume_sum_10",
    ]
]

y = dataset["target"]

pred = model.predict(X)

accuracy = accuracy_score(y, pred)
precision = precision_score(y, pred)
recall = recall_score(y, pred)
f1 = f1_score(y, pred)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1       :", f1)

with open("metrics.txt", "w") as f:

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
