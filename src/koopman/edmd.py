import numpy as np
import pykoopman as pk
from pykoopman.observables import Polynomial
from src.core.interfaces import KoopmanModel
from typing import Optional


class EDMDKoopmanModel(KoopmanModel):

    def __init__(self, polynomial_degree: int = 2):
        self.polynomial_degree = polynomial_degree

        self.observables = Polynomial(degree=self.polynomial_degree)

        self.model = pk.Koopman(observables=self.observables)
        self.is_fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.model.fit(x=X, y=Y)
        self.is_fitted = True

    def predict(self, x_t: np.ndarray, steps: int = 1) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Модель Купмана еще не обучена. Вызовите fit().")

        if x_t.ndim == 1:
            x_t = x_t.reshape(1, -1)

        trajectory = self.model.simulate(x_t, n_steps=steps)
        return trajectory[-1]

    def get_eigenvalues(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Модель Купмана еще не обучена.")

        K = self.model.koopman_matrix

        eigenvalues = np.linalg.eigvals(K)
        return eigenvalues

    def get_instability_index(self) -> float:
        evals = self.get_eigenvalues()
        max_modulus = np.max(np.abs(evals))
        return float(max_modulus)
