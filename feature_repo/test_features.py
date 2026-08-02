from feast import FeatureStore

store = FeatureStore(repo_path=".")

response = store.get_online_features(
    features=[
        "stock_features:rolling_avg_10",
        "stock_features:volume_sum_10",
    ],
    entity_rows=[
        {"stock_name": "AARTIIND"},
    ],
)

print(response.to_dict())
