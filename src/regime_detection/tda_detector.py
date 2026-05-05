import numpy as np
from gtda.diagrams import Amplitude
from gtda.homology import VietorisRipsPersistence

from src.core.interfaces import TDARegimeDetector


class PersistenceLandscapeDetector(TDARegimeDetector):
    def __init__(self, max_edge_length: float = 10.0, metric: str = "wasserstein"):
        self.vr_persistence = VietorisRipsPersistence(
            homology_dimensions=[0, 1],
            max_edge_length=max_edge_length,
            n_jobs=-1,
        )
        self.amplitude = Amplitude(metric=metric)
        self._amplitude_fitted = False

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            return X.reshape(1, X.shape[0], X.shape[1])
        elif X.ndim == 3:
            return X
        else:
            raise ValueError("X должен иметь размерность 2 (одно окно) или 3 (батч окон).")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X_reshaped = self._reshape(X)
        diagrams = self.vr_persistence.fit_transform(X_reshaped)
        amplitudes = self.amplitude.fit_transform(diagrams)
        self._amplitude_fitted = True
        return amplitudes[:, 0:1]

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._amplitude_fitted:
            raise ValueError(
                "Необходимо сначала вызвать fit_transform() на обучающих данных, "
                "чтобы откалибровать нормализатор амплитуд."
            )
        X_reshaped = self._reshape(X)
        diagrams = self.vr_persistence.fit_transform(X_reshaped)
        amplitudes = self.amplitude.transform(diagrams)
        return amplitudes[:, 0:1]

    def detect_regime(
            self,
            current_amplitude: float,
            historical_amplitudes: np.ndarray,
            threshold_percentile: float = 85.0,
    ) -> int:
        threshold = np.percentile(historical_amplitudes, threshold_percentile)
        return 1 if current_amplitude > threshold else 0
