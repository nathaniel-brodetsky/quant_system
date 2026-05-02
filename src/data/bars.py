import pandas as pd
import numpy as np


def create_volume_bars(ticks: pd.DataFrame, volume_threshold: float) -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame()

    ticks['cum_vol'] = ticks['volume'].cumsum()

    ticks['bar_id'] = (ticks['cum_vol'] // volume_threshold).astype(int)

    bars = ticks.groupby('bar_id').agg(
        timestamp=('index', 'last'),
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('volume', 'sum')
    )

    bars.set_index('timestamp', inplace=True)
    return bars
