import pandas as pd
import numpy as np
from typing import Dict, Any
from src.core.interfaces import FeatureEngine


class MicrostructureEngine(FeatureEngine):

    def __init__(self, window: int = 10):
        self.window = window

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

        df['dp'] = df['close'].diff()
        cov = df['dp'].rolling(window=self.window).cov(df['dp'].shift(1))
        df['roll_measure'] = 2 * np.sqrt(np.clip(-cov, 0, None))

        price_change_sign = np.sign(df['dp'])
        price_change_sign = price_change_sign.ffill()
        df['signed_volume'] = price_change_sign * df['volume']

        df['ofi_proxy'] = df['signed_volume'].rolling(window=self.window).mean()

        const = 1.0 / (4.0 * np.log(2.0))
        hl_log_sq = (np.log(df['high'] / df['low'])) ** 2
        df['parkinson_vol'] = np.sqrt(const * hl_log_sq.rolling(window=self.window).mean())

        df.drop(columns=['dp', 'signed_volume'], inplace=True)
        df.dropna(inplace=True)

        return df

    def update_online(self, new_data: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError("Online update will be implemented in the execution phase.")
