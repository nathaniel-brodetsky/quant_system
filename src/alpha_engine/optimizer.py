import itertools
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from src.alpha_engine.generator import HybridAlphaEngine
from src.evaluation.walk_forward import WalkForwardEvaluator

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    params: Dict[str, Any]
    sharpe: float
    total_return: float
    max_drawdown: float
    n_trades: int
    error: str = ""


class AlphaOptimizer:
    REQUIRED_KEYS = {"tda_window", "tda_percentile", "z_threshold"}
    OPTIONAL_KEYS = {"z_quantile", "horizon", "min_hold_bars", "direction_mode", "rsi_oversold", "rsi_overbought", "htf_k"}

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        train_window: int = 1000,
        test_window: int = 100,
        commission: float = 0.0002,
        slippage: float = 0.0001,
        risk_manager=None,
        verbose: bool = True,
    ):
        missing = self.REQUIRED_KEYS - set(param_grid.keys())
        if missing:
            raise ValueError(f"param_grid не содержит обязательных ключей: {missing}")

        self.param_grid = param_grid
        self.train_window = train_window
        self.test_window = test_window
        self.commission = commission
        self.slippage = slippage
        self.risk_manager = risk_manager
        self.verbose = verbose
        self._results: List[TrialResult] = []

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        self._results.clear()
        combinations = list(self._generate_combinations())
        total = len(combinations)
        logger.info("AlphaOptimizer: старт. Всего комбинаций: %d", total)

        for idx, params in enumerate(combinations, start=1):
            if self.verbose:
                print(f"[{idx}/{total}] Тест параметров: {params}", flush=True)

            result = self._evaluate_single(data, params)
            self._results.append(result)

            if self.verbose and not result.error:
                print(
                    f"         → Sharpe={result.sharpe:.3f}  "
                    f"Return={result.total_return:.2f}%  "
                    f"MaxDD={result.max_drawdown:.2f}%  "
                    f"Trades={result.n_trades}"
                )
            elif result.error:
                print(f"         ✗ Ошибка: {result.error}")

        return self._to_dataframe()

    def _generate_combinations(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    def _evaluate_single(self, data: pd.DataFrame, params: Dict[str, Any]) -> TrialResult:
        try:
            engine_kwargs = {
                "tda_window": params["tda_window"],
                "tda_percentile": params["tda_percentile"],
                "z_threshold": params["z_threshold"],
            }
            for key in self.OPTIONAL_KEYS:
                if key in params:
                    engine_kwargs[key] = params[key]
            engine = HybridAlphaEngine(**engine_kwargs)

            evaluator = WalkForwardEvaluator(
                train_window=self.train_window,
                test_window=self.test_window,
                commission=self.commission,
                slippage=self.slippage,
            )

            def alpha_logic(train_df, test_df):
                return engine.generate_signals(train_df, test_df)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oos_df = evaluator.walk_forward_eval(
                    data, alpha_logic, risk_manager=self.risk_manager
                )

            summary = WalkForwardEvaluator.get_summary(oos_df)
            n_trades = int(oos_df["trade_happened"].sum()) if "trade_happened" in oos_df.columns else 0

            return TrialResult(
                params=params,
                sharpe=summary["Sharpe Ratio"],
                total_return=summary["Total Return"],
                max_drawdown=summary["Max Drawdown"],
                n_trades=n_trades,
            )

        except Exception as exc:
            logger.warning("Ошибка при params=%s: %s", params, exc, exc_info=True)
            return TrialResult(
                params=params, sharpe=-999.0,
                total_return=0.0, max_drawdown=0.0,
                n_trades=0, error=str(exc),
            )

    def _to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self._results:
            row = {**r.params, "sharpe": r.sharpe, "total_return": r.total_return,
                   "max_drawdown": r.max_drawdown, "n_trades": r.n_trades}
            if r.error:
                row["error"] = r.error
            rows.append(row)

        df = pd.DataFrame(rows)
        if "error" in df.columns:
            df_ok = df[df["error"] == ""]
            df_err = df[df["error"] != ""]
        else:
            df_ok, df_err = df, pd.DataFrame()

        return pd.concat(
            [df_ok.sort_values("sharpe", ascending=False), df_err],
            ignore_index=True,
        )

    def best_params(self, top_n: int = 1) -> List[Dict[str, Any]]:
        df = self._to_dataframe()
        param_cols = list(self.param_grid.keys())
        available = [c for c in param_cols if c in df.columns]
        return df.head(top_n)[available].to_dict(orient="records")