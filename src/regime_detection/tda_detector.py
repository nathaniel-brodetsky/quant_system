import numpy as np
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude
from src.core.interfaces import TDARegimeDetector


class PersistenceLandscapeDetector(TDARegimeDetector):

    def __init__(self, max_edge_length: float = 10.0, metric: str = 'wasserstein'):
        self.vr_persistence = VietorisRipsPersistence(
            homology_dimensions=[0, 1],
            max_edge_length=max_edge_length,
            n_jobs=-1
        )

        self.amplitude = Amplitude(metric=metric)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            X_reshaped = X.reshape(1, X.shape[0], X.shape[1])
        elif X.ndim == 3:
            X_reshaped = X
        else:
            raise ValueError("X должен иметь размерность 2 (одно окно) или 3 (батч окон).")

        diagrams = self.vr_persistence.fit_transform(X_reshaped)

        amplitudes = self.amplitude.fit_transform(diagrams)

        return amplitudes

    def detect_regime(self, current_topology_amplitude: float, historical_amplitudes: np.ndarray,
                      threshold_percentile: float = 85.0) -> int:
        threshold = np.percentile(historical_amplitudes, threshold_percentile)

        if current_topology_amplitude > threshold:
            return 1
        else:
            return 0
