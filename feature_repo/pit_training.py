import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path=".")

entity_df = (
    pd.read_parquet("data/features.parquet")
    [["stock_name", "event_timestamp", "target"]]
    .head(50)
)

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "stock_features:rolling_avg_10",
        "stock_features:volume_sum_10",
    ],
).to_df()

print(training_df.head())
print(training_df.shape)
