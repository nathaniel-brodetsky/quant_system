from typing import Any, Dict

import numpy as np
import pandas as pd

from src.core.interfaces import FeatureEngine


class MicrostructureEngine(FeatureEngine):
    def __init__(self, window: int = 10):
        self.window = window

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

        df["dp"] = df["close"].diff()
        cov = df["dp"].rolling(window=self.window).cov(df["dp"].shift(1))
        df["roll_measure"] = 2 * np.sqrt(np.clip(-cov, 0, None))

        if "taker_buy_base_vol" in df.columns:
            df["market_sell_vol"] = df["volume"] - df["taker_buy_base_vol"]
            raw_ofi = df["taker_buy_base_vol"] - df["market_sell_vol"]
            df["ofi_proxy"] = raw_ofi.rolling(window=self.window).mean()
        else:
            price_change_sign = np.sign(df["dp"]).ffill()
            df["ofi_proxy"] = (
                (price_change_sign * df["volume"]).rolling(window=self.window).mean()
            )

        const = 1.0 / (4.0 * np.log(2.0))
        hl_log_sq = (np.log(df["high"] / df["low"])) ** 2
        df["parkinson_vol"] = np.sqrt(
            const * hl_log_sq.rolling(window=self.window).mean()
        )

        cols_to_drop = ["dp"]
        if "market_sell_vol" in df.columns:
            cols_to_drop.append("market_sell_vol")
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        df.dropna(inplace=True)

        return df

    def update_online(self, new_data: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError()
