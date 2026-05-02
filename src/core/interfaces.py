from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any


class DataLoader(ABC):
    @abstractmethod
    def fetch_historical(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        pass


class FeatureEngine(ABC):
    @abstractmethod
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def update_online(self, new_data: Dict[str, Any]) -> np.ndarray:
        pass


class KoopmanModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, x_t: np.ndarray, steps: int = 1) -> np.ndarray:
        pass


class TDARegimeDetector(ABC):
    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def detect_regime(self, current_topology: np.ndarray) -> int:
        pass
