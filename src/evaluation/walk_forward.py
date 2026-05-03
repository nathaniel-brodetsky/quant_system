import pandas as pd
import numpy as np
from typing import Callable, Tuple
from src.core.interfaces import StrategyEvaluator


class WalkForwardEvaluator(StrategyEvaluator):

    def __init__(self, train_window: int = 1000, test_window: int = 100,
                 commission: float = 0.0002, slippage: float = 0.0001):

        self.train_window = train_window
        self.test_window = test_window
        self.commission = commission
        self.slippage = slippage

    def walk_forward_eval(self, data: pd.DataFrame, alpha_logic: Callable) -> pd.DataFrame:

        total_bars = len(data)
        out_of_sample_results = []

        for start_idx in range(0, total_bars - self.train_window - self.test_window, self.test_window):
            train_end = start_idx + self.train_window
            test_end = train_end + self.test_window

            train_data = data.iloc[start_idx:train_end].copy()
            test_data = data.iloc[train_end:test_end].copy()

            signals = alpha_logic(train_data, test_data)

            test_data['signal'] = signals
            out_of_sample_results.append(test_data)

        if not out_of_sample_results:
            raise ValueError("Недостаточно данных для бэктеста с заданными окнами.")

        full_oos_data = pd.concat(out_of_sample_results)

        return self._calculate_metrics(full_oos_data)

    def _calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df['position'] = df['signal'].shift(1).fillna(0)

        df['market_return'] = df['close'].pct_change()

        df['strategy_return'] = df['position'] * df['market_return']

        df['trade_happened'] = df['position'].diff().abs() > 0

        costs = df['trade_happened'] * (self.commission + self.slippage)
        df['strategy_return_net'] = df['strategy_return'] - costs

        df['equity_curve'] = (1 + df['strategy_return_net']).cumprod()

        return df

    @staticmethod
    def get_summary(df: pd.DataFrame) -> dict:
        returns = df['strategy_return_net'].dropna()

        if len(returns) == 0 or returns.std() == 0:
            return {"Sharpe Ratio": 0.0, "Max Drawdown": 0.0, "Total Return": 0.0}

        sharpe = np.sqrt(len(returns)) * (returns.mean() / returns.std())

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            "Sharpe Ratio": round(sharpe, 2),
            "Max Drawdown": round(max_drawdown * 100, 2),
            "Total Return": round((cumulative.iloc[-1] - 1) * 100, 2)
        }
