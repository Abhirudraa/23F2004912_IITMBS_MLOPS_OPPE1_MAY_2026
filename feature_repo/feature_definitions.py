from datetime import timedelta

from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32
from feast.value_type import ValueType


# Entity
stock = Entity(
    name="stock",
    join_keys=["stock_name"],
    value_type=ValueType.STRING,
)


# Offline source
stock_source = FileSource(
    name="stock_source",
    path="data/features.parquet",
    timestamp_field="event_timestamp",
)


# Feature View
stock_features = FeatureView(
    name="stock_features",
    entities=[stock],
    ttl=timedelta(days=1),
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Float32),
    ],
    source=stock_source,
)
