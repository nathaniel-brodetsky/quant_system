from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


class DataLoader(ABC):
    @abstractmethod
    def fetch_historical(
            self,
            symbol: str,
            start: str,
            end: Optional[str] = None,
            interval: str = "1m",
    ) -> pd.DataFrame: ...


class FeatureEngine(ABC):
    @abstractmethod
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def update_online(self, new_bar: Dict[str, Any]) -> np.ndarray: ...


class KoopmanModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray, horizon: int = 1) -> None: ...

    @abstractmethod
    def predict(self, x_t: np.ndarray, steps: int = 1) -> np.ndarray: ...

    @abstractmethod
    def get_eigenvalues(self) -> np.ndarray: ...


class TDARegimeDetector(ABC):
    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def detect_regime(
            self,
            current_amplitude: float,
            historical_amplitudes: np.ndarray,
            threshold_percentile: float,
    ) -> int: ...


class AlphaEngine(ABC):
    @abstractmethod
    def generate_signals(
            self, train: pd.DataFrame, test: pd.DataFrame
    ) -> pd.Series: ...

    @abstractmethod
    def evaluate_and_evolve(
            self, features: np.ndarray, returns: np.ndarray
    ) -> None: ...


class StrategyEvaluator(ABC):
    @abstractmethod
    def walk_forward_eval(
            self,
            data: pd.DataFrame,
            alpha_logic: Callable,
            risk_manager: Optional["RiskManager"] = None,
    ) -> pd.DataFrame: ...


class RiskManager(ABC):
    @abstractmethod
    def calculate_position_size(
            self, signal: float, current_volatility: float
    ) -> float: ...

    @abstractmethod
    def apply_risk_to_signals(
            self, signals: pd.Series, volatility: pd.Series
    ) -> pd.Series: ...


class MTFEngine(ABC):
    @abstractmethod
    def build_htf(self, ltf_bars: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def auto_select_k(
            self, ltf_bars: pd.DataFrame, candidates: List[int]
    ) -> int: ...

    @abstractmethod
    def align_to_ltf(
            self, htf_values: pd.Series, ltf_index: pd.DatetimeIndex
    ) -> pd.Series: ...


class ExperimentLogger(ABC):
    @abstractmethod
    def log_run(self, params: Dict, metrics: Dict, artifacts: Dict) -> str: ...

    @abstractmethod
    def load_run(self, run_id: str) -> Dict: ...
