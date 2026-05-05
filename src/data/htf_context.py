from __future__ import annotations

import numpy as np
import pandas as pd


class HTFContext:
    def __init__(self, k: int = 20) -> None:
        if k < 2:
            raise ValueError(f"k должен быть >= 2, получено {k}")
        self.k = k

    def build(self, ltf_bars: pd.DataFrame) -> pd.DataFrame:
        if ltf_bars.empty:
            return pd.DataFrame()

        df = ltf_bars.copy()
        n = len(df)
        df["_htf_id"] = np.arange(n) // self.k

        agg: dict[str, tuple] = {}

        for col, func in [
            ("open", "first"),
            ("high", "max"),
            ("low", "min"),
            ("close", "last"),
            ("volume", "sum"),
        ]:
            if col in df.columns:
                agg[col] = (col, func)

        if "log_ret" in df.columns:
            agg["log_ret"] = ("log_ret", "sum")
        if "parkinson_vol" in df.columns:
            agg["parkinson_vol"] = ("parkinson_vol", "mean")
        if "ofi_proxy" in df.columns:
            agg["ofi_proxy"] = ("ofi_proxy", "sum")
        if "roll_measure" in df.columns:
            agg["roll_measure"] = ("roll_measure", "mean")

        htf = df.groupby("_htf_id").agg(**agg)

        last_ts = df.index.to_series().groupby(df["_htf_id"]).last()
        htf.index = pd.DatetimeIndex(last_ts.values)
        htf.index.name = "timestamp"

        return htf

    def compute_htf_features(
            self,
            htf_bars: pd.DataFrame,
            rolling_window: int = 3,
    ) -> pd.DataFrame:
        df = htf_bars.copy()

        if "log_ret" not in df.columns:
            df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

        if "parkinson_vol" not in df.columns:
            const = 1.0 / (4.0 * np.log(2.0))
            hl_sq = np.log(df["high"] / df["low"]) ** 2
            df["parkinson_vol"] = np.sqrt(const * hl_sq)

        df["parkinson_vol_smooth"] = (
            df["parkinson_vol"].rolling(rolling_window, min_periods=1).mean()
        )

        df.dropna(inplace=True)
        return df

    def align_to_ltf(
            self,
            htf_values: pd.Series,
            ltf_index: pd.DatetimeIndex,
            shift: bool = True,
    ) -> pd.Series:
        values = htf_values.shift(1) if shift else htf_values

        combined_index = values.index.union(ltf_index)
        aligned = values.reindex(combined_index).ffill()
        return aligned.reindex(ltf_index).fillna(0.0)

    def htf_bars_count(self, n_ltf: int) -> int:
        return (n_ltf + self.k - 1) // self.k

    def __repr__(self) -> str:
        return f"HTFContext(k={self.k})"
