import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(ROOT, "StockAnalyticaData", "v0")
OUT_DIR = os.path.join(ROOT, "feature_repo", "data")

os.makedirs(OUT_DIR, exist_ok=True)

frames = []

for file in os.listdir(DATA_DIR):

    if not file.endswith(".csv"):
        continue

    stock = file.replace("__EQ__NSE__NSE__MINUTE.csv", "")

    df = pd.read_csv(os.path.join(DATA_DIR, file))

    # --------- USE SMALL SUBSET ----------
    df = df.head(500)
    # -------------------------------------

    df["event_timestamp"] = pd.to_datetime(df["timestamp"])
    df["stock_name"] = stock

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

    frames.append(
        df[
            [
                "stock_name",
                "event_timestamp",
                "rolling_avg_10",
                "volume_sum_10",
                "target",
            ]
        ]
    )

features = pd.concat(frames, ignore_index=True)

outfile = os.path.join(OUT_DIR, "features.parquet")

features.to_parquet(outfile, index=False)

print(features.head())
print(features.shape)
print(outfile)
