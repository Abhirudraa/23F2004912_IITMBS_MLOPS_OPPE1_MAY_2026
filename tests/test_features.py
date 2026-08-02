import os
import pandas as pd


def test_rolling_avg_feature():

    file = os.path.join("data/merged", os.listdir("data/merged")[0])

    df = pd.read_csv(file)

    rolling = df["close"].rolling(10, min_periods=1).mean()

    assert rolling.notna().all()


def test_volume_sum_feature():

    file = os.path.join("data/merged", os.listdir("data/merged")[0])

    df = pd.read_csv(file)

    volume = df["volume"].rolling(10, min_periods=1).sum()

    assert volume.notna().all()
