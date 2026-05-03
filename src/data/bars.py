import pandas as pd
import numpy as np


def create_volume_bars(ticks: pd.DataFrame, volume_threshold: float) -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame()

    df = ticks.copy()

    df['timestamp'] = df.index

    df['cum_vol'] = df['volume'].cumsum()

    df['bar_id'] = (df['cum_vol'] // volume_threshold).astype(int)

    bars = df.groupby('bar_id').agg(
        timestamp=('timestamp', 'last'),
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('volume', 'sum')
    )

    bars.set_index('timestamp', inplace=True)
    return bars