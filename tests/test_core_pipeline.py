import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


def _install_mocks():
    pk = types.ModuleType("pykoopman")
    pk_obs = types.ModuleType("pykoopman.observables")
    pk_reg = types.ModuleType("pykoopman.regression")

    class FakeTimeDelay:
        def __init__(self, delay=1, n_delays=3):
            pass

    class FakeEDMD:
        def __init__(self, svd_rank=0.95):
            pass

    class FakeKoopman:
        def __init__(self, observables=None, regressor=None):
            pass

        def fit(self, X):
            pass

        def simulate(self, x0, n_steps=1):
            n_features = x0.shape[-1] if hasattr(x0, "shape") else 4
            return np.zeros((n_steps, n_features))

    pk_obs.TimeDelay = FakeTimeDelay
    pk_obs.RadialBasisFunction = MagicMock()
    pk_obs.CustomObservables = MagicMock()
    pk_reg.EDMD = FakeEDMD
    pk.Koopman = FakeKoopman
    pk.observables = pk_obs
    pk.regression = pk_reg

    gtda = types.ModuleType("gtda")
    gtda_hom = types.ModuleType("gtda.homology")
    gtda_diag = types.ModuleType("gtda.diagrams")

    class FakeVR:
        def __init__(self, **kw):
            pass

        def fit_transform(self, X):
            return np.zeros((X.shape[0], 4, 3))

    class FakeAmplitude:
        def __init__(self, **kw):
            pass

        def fit_transform(self, d):
            return np.random.rand(d.shape[0], 2)

        def transform(self, d):
            return np.random.rand(d.shape[0], 2)

    gtda_hom.VietorisRipsPersistence = FakeVR
    gtda_diag.Amplitude = FakeAmplitude
    gtda.homology = gtda_hom
    gtda.diagrams = gtda_diag

    src = types.ModuleType("src")
    src_core = types.ModuleType("src.core")
    src_iface = types.ModuleType("src.core.interfaces")

    class _ABC:
        pass

    for name in [
        "DataLoader", "FeatureEngine", "KoopmanModel",
        "TDARegimeDetector", "AlphaEngine", "StrategyEvaluator", "RiskManager",
    ]:
        setattr(src_iface, name, _ABC)
    src_core.interfaces = src_iface

    class FakeScaler:
        def fit_transform(self, X):
            return X.astype(float)

        def transform(self, X):
            return X.astype(float)

    sk = types.ModuleType("sklearn")
    sk_pre = types.ModuleType("sklearn.preprocessing")
    sk_pre.StandardScaler = FakeScaler

    for mod_name, mod in [
        ("pykoopman", pk),
        ("pykoopman.observables", pk_obs),
        ("pykoopman.regression", pk_reg),
        ("gtda", gtda),
        ("gtda.homology", gtda_hom),
        ("gtda.diagrams", gtda_diag),
        ("binance", types.ModuleType("binance")),
        ("binance.client", types.ModuleType("binance.client")),
        ("sklearn", sk),
        ("sklearn.preprocessing", sk_pre),
        ("src", src),
        ("src.core", src_core),
        ("src.core.interfaces", src_iface),
    ]:
        sys.modules[mod_name] = mod

    for stub in [
        "src.data", "src.features", "src.evaluation",
        "src.risk", "src.regime_detection", "src.alpha_engine",
    ]:
        sys.modules[stub] = types.ModuleType(stub)

    fake_gen = types.ModuleType("src.alpha_engine.generator")
    fake_gen.HybridAlphaEngine = MagicMock
    sys.modules["src.alpha_engine.generator"] = fake_gen

    fake_wf = types.ModuleType("src.evaluation.walk_forward")
    fake_wf.WalkForwardEvaluator = MagicMock
    sys.modules["src.evaluation.walk_forward"] = fake_wf

    fake_tda = types.ModuleType("src.regime_detection.tda_detector")
    fake_tda.PersistenceLandscapeDetector = MagicMock
    sys.modules["src.regime_detection.tda_detector"] = fake_tda

    fake_edmd = types.ModuleType("src.koopman.edmd")
    sys.modules["src.koopman"] = types.ModuleType("src.koopman")
    sys.modules["src.koopman.edmd"] = fake_edmd


_install_mocks()
sys.path.insert(0, "/home/claude")

edmd_mod = importlib.import_module("edmd")
optimizer_mod = importlib.import_module("optimizer")

EDMDKoopmanModel = edmd_mod.EDMDKoopmanModel
AlphaOptimizer = optimizer_mod.AlphaOptimizer
TrialResult = optimizer_mod.TrialResult


def _arr(n=200, f=4):
    return np.random.default_rng(42).standard_normal((n, f))


def _opt_with_results(grid):
    opt = AlphaOptimizer(grid, train_window=100, test_window=50)
    opt._results = [
        TrialResult(
            params={k: v[0] for k, v in grid.items()},
            sharpe=1.2,
            total_return=5.0,
            max_drawdown=-3.0,
            n_trades=10,
        ),
        TrialResult(
            params={k: v[-1] for k, v in grid.items()},
            sharpe=0.8,
            total_return=3.0,
            max_drawdown=-2.0,
            n_trades=8,
        ),
    ]
    return opt


GRID = {
    "tda_window": [20, 30],
    "tda_percentile": [70.0],
    "z_threshold": [0.05],
    "z_quantile": [0.6, 0.7],
}


class TestEDMDInstantiation:
    def test_valid_params_ok(self):
        assert EDMDKoopmanModel(delay_shifts=3, rbf_centers=10) is not None

    def test_polynomial_degree_raises(self):
        with pytest.raises(TypeError):
            EDMDKoopmanModel(polynomial_degree=2)

    def test_defaults(self):
        m = EDMDKoopmanModel()
        assert m.delay_shifts == 3
        assert not m.is_fitted


class TestOptimizerBestParams:
    def test_all_keys_present(self):
        opt = _opt_with_results(GRID)
        result = opt.best_params(top_n=1)[0]
        missing = set(GRID) - set(result)
        assert not missing, f"Потерянные ключи: {missing}"

    def test_z_quantile_not_lost(self):
        opt = _opt_with_results(GRID)
        assert "z_quantile" in opt.best_params(top_n=1)[0]

    def test_no_metric_columns(self):
        opt = _opt_with_results(GRID)
        result = opt.best_params(top_n=1)[0]
        for m in ("sharpe", "total_return", "max_drawdown", "n_trades"):
            assert m not in result

    def test_sorted_by_sharpe(self):
        opt = _opt_with_results(GRID)
        assert opt.best_params(top_n=2)[0]["tda_window"] == 20

    def test_top_n_count(self):
        opt = _opt_with_results(GRID)
        assert len(opt.best_params(top_n=2)) == 2

    def test_missing_required_key_raises(self):
        with pytest.raises(ValueError):
            AlphaOptimizer({"tda_window": [20], "z_threshold": [0.05]})


class TestEDMDHorizon:
    def test_fit_h1(self):
        m = EDMDKoopmanModel(delay_shifts=2)
        m.fit(_arr(), _arr(), horizon=1)
        assert m.is_fitted

    def test_fit_h5(self):
        m = EDMDKoopmanModel(delay_shifts=2)
        m.fit(_arr(), _arr(), horizon=5)
        assert m.is_fitted

    def test_horizon_stored(self):
        m = EDMDKoopmanModel(delay_shifts=2)
        m.fit(_arr(), _arr(), horizon=7)
        assert m._horizon == 7

    def test_predict_returns_ndarray(self):
        m = EDMDKoopmanModel(delay_shifts=2)
        X = _arr()
        m.fit(X, X, horizon=5)
        pred = m.predict(X[0], steps=5)
        assert isinstance(pred, np.ndarray)

    def test_predict_without_fit_raises(self):
        with pytest.raises(ValueError, match="не обучена"):
            EDMDKoopmanModel().predict(_arr(10)[0])

    def test_small_array_with_subsampling(self):
        m = EDMDKoopmanModel(delay_shifts=1)
        m.fit(_arr(40), _arr(40), horizon=5)
        assert m.is_fitted
