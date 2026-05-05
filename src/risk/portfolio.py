import numpy as np
import pandas as pd

from src.core.interfaces import RiskManager


class TargetVolatilityRiskManager(RiskManager):
    def __init__(self, target_annual_vol: float = 0.40, max_leverage: float = 1.0):
        self.target_annual_vol = target_annual_vol
        self.max_leverage = max_leverage
        self.bars_per_year = 525_600

    def calculate_position_size(self, signal: float, current_volatility: float) -> float:
        if signal == 0.0 or current_volatility == 0.0 or pd.isna(current_volatility):
            return 0.0

        annualized_current_vol = current_volatility * np.sqrt(self.bars_per_year)
        weight = self.target_annual_vol / annualized_current_vol
        weight = min(weight, self.max_leverage)
        return weight * np.sign(signal)

    def apply_risk_to_signals(
            self, signals: pd.Series, volatility_series: pd.Series
    ) -> pd.Series:
        positions = [
            self.calculate_position_size(sig, vol)
            for sig, vol in zip(signals, volatility_series)
        ]
        return pd.Series(positions, index=signals.index)
