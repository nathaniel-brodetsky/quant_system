from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.core.interfaces import StrategyEvaluator

_BARS_PER_YEAR_1MIN = 525_600


class WalkForwardEvaluator(StrategyEvaluator):
    def __init__(
        self,
        train_window: int = 1000,
        test_window: int = 100,
        commission: float = 0.0002,
        slippage: float = 0.0001,
        bars_per_year: Optional[int] = None,
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.commission = commission
        self.slippage = slippage
        self.bars_per_year = bars_per_year

    def _estimate_bars_per_year(self, data: pd.DataFrame) -> int:
        if not isinstance(data.index, pd.DatetimeIndex) or len(data) < 2:
            return _BARS_PER_YEAR_1MIN
        total_seconds = (data.index[-1] - data.index[0]).total_seconds()
        if total_seconds <= 0:
            return _BARS_PER_YEAR_1MIN
        bars_per_second = (len(data) - 1) / total_seconds
        return max(1, int(bars_per_second * 365.25 * 24 * 3600))

    def walk_forward_eval(
        self,
        data: pd.DataFrame,
        alpha_logic: Callable,
        risk_manager=None,
    ) -> pd.DataFrame:
        total_bars = len(data)
        out_of_sample_results = []
        bpy = self.bars_per_year or self._estimate_bars_per_year(data)

        for start_idx in range(
            0, total_bars - self.train_window - self.test_window, self.test_window
        ):
            train_end = start_idx + self.train_window
            test_end = train_end + self.test_window

            train_data = data.iloc[start_idx:train_end].copy()
            test_data = data.iloc[train_end:test_end].copy()

            raw_signals = alpha_logic(train_data, test_data)

            test_data["raw_signal"] = raw_signals.values

            if risk_manager is not None:
                vol_series = test_data["parkinson_vol"]
                final_positions = risk_manager.apply_risk_to_signals(raw_signals, vol_series)
            else:
                final_positions = raw_signals

            test_data["signal"] = final_positions.values
            test_data["_bpy"] = bpy
            out_of_sample_results.append(test_data)

        if not out_of_sample_results:
            raise ValueError("Недостаточно данных для бэктеста с заданными окнами.")

        full_oos_data = pd.concat(out_of_sample_results)
        return self._calculate_metrics(full_oos_data)

    def _calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df["position"] = df["signal"].shift(1).fillna(0)
        df["market_return"] = df["close"].pct_change()
        df["strategy_return"] = df["position"] * df["market_return"]

        if "raw_signal" in df.columns:
            raw_pos = df["raw_signal"].shift(1).fillna(0)
            df["trade_happened"] = raw_pos.diff().abs() > 0
        else:
            df["trade_happened"] = df["position"].diff().abs() > 0.05

        costs = df["trade_happened"] * (self.commission + self.slippage)
        df["strategy_return_net"] = df["strategy_return"] - costs
        df["equity_curve"] = (1 + df["strategy_return_net"]).cumprod()

        return df

    @staticmethod
    def get_summary(df: pd.DataFrame, bars_per_year: Optional[int] = None) -> dict:
        returns = df["strategy_return_net"].dropna()

        if len(returns) == 0 or returns.std() == 0:
            return {"Sharpe Ratio": 0.0, "Max Drawdown": 0.0, "Total Return": 0.0}

        bpy = bars_per_year
        if bpy is None and "_bpy" in df.columns:
            bpy = int(df["_bpy"].iloc[0])
        if bpy is None:
            bpy = _BARS_PER_YEAR_1MIN

        sharpe = np.sqrt(bpy) * (returns.mean() / returns.std())
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        return {
            "Sharpe Ratio": round(sharpe, 2),
            "Max Drawdown": round(drawdown.min() * 100, 2),
            "Total Return": round((cumulative.iloc[-1] - 1) * 100, 2),
        }