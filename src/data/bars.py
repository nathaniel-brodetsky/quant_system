import numpy as np
import pandas as pd


def create_volume_bars(ticks: pd.DataFrame, volume_threshold: float) -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame()

    df = ticks.copy()
    df["timestamp"] = df.index
    df["cum_vol"] = df["volume"].cumsum()
    df["bar_id"] = (df["cum_vol"] // volume_threshold).astype(int)

    bars = df.groupby("bar_id").agg(
        timestamp=("timestamp", "last"),
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume", "sum"),
    )
    bars.set_index("timestamp", inplace=True)
    return bars


def create_dollar_bars(ticks: pd.DataFrame, dollar_threshold: float) -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame()

    df = ticks.copy()
    df["timestamp"] = df.index
    df["dollar_vol"] = df["price"] * df["volume"]
    df["cum_dollar"] = df["dollar_vol"].cumsum()
    df["bar_id"] = (df["cum_dollar"] // dollar_threshold).astype(int)

    bars = df.groupby("bar_id").agg(
        timestamp=("timestamp", "last"),
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume", "sum"),
        dollar_volume=("dollar_vol", "sum"),
    )
    bars.set_index("timestamp", inplace=True)
    return bars


def create_tick_bars(ticks: pd.DataFrame, ticks_per_bar: int) -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame()

    df = ticks.copy()
    df["timestamp"] = df.index
    df["bar_id"] = np.arange(len(df)) // ticks_per_bar

    bars = df.groupby("bar_id").agg(
        timestamp=("timestamp", "last"),
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume", "sum"),
    )
    bars.set_index("timestamp", inplace=True)
    return bars
