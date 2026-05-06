import numpy as np
import pykoopman as pk
from pykoopman.observables import TimeDelay

from src.core.interfaces import KoopmanModel


class EDMDKoopmanModel(KoopmanModel):
    def __init__(self, delay_shifts: int = 3, rbf_centers: int = 10):
        self.delay_shifts = delay_shifts
        self.rbf_centers = rbf_centers

        self.delay_obs = TimeDelay(delay=1, n_delays=self.delay_shifts)
        self.observables = self.delay_obs
        self.regressor = pk.regression.EDMD(svd_rank=0.95)
        self.model = pk.Koopman(observables=self.observables, regressor=self.regressor)
        self.is_fitted = False
        self._horizon = 1

    @property
    def n_min_rows(self) -> int:
        return self.delay_shifts + 1

    def fit(self, X: np.ndarray, Y: np.ndarray, horizon: int = 1) -> None:
        X_train = X[::horizon] if horizon > 1 else X
        self.model.fit(X_train)
        self._horizon = horizon
        self.is_fitted = True

    def predict(self, x_trajectory: np.ndarray, steps: int = 1) -> np.ndarray:

        if not self.is_fitted:
            raise ValueError("Модель Купмана еще не обучена.")

        if x_trajectory.ndim == 1:
            x_trajectory = x_trajectory.reshape(1, -1)

        n_min = self.n_min_rows
        if len(x_trajectory) < n_min:
            pad = np.zeros((n_min - len(x_trajectory), x_trajectory.shape[1]))
            x_trajectory = np.vstack([pad, x_trajectory])

        try:
            trajectory = self.model.simulate(x_trajectory, n_steps=steps)
            return trajectory[-1]
        except Exception:
            trajectory = self.model.simulate(x_trajectory[-n_min:], n_steps=steps)
            return trajectory[-1]

    def get_eigenvalues(self) -> np.ndarray:
        if not self.is_fitted:
            return np.array([])
        K = self.model.koopman_matrix
        return np.linalg.eigvals(K)