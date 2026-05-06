import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.htf_context import HTFContext
from src.koopman.edmd import EDMDKoopmanModel
from src.regime_detection.tda_detector import PersistenceLandscapeDetector


class HybridAlphaEngine:
    """
    v5: RSI mean-reversion + HTF TDA regime + Koopman amplitude filter.

    Signal logic per bar:
      1. HTF TDA amplitude > threshold  (active regime)
      2. RSI in extreme zone            (direction signal)
      3. |Koopman pred| > threshold     (amplitude confirmation — direction ignored)
      → enter position in RSI direction

    Koopman direction confirmation removed: corr~3% = noise, randomly rejects
    good RSI signals. Koopman kept only as amplitude filter (unusual move expected).
    """

    def __init__(
        self,
        horizon: int = 5,
        tda_window: int = 15,
        tda_percentile: float = 90.0,
        z_threshold: float = 0.0,
        z_quantile: float = 0.75,
        min_hold_bars: int = 0,
        direction_mode: str = "rsi",
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        htf_k: int = 10,
    ):
        self.horizon = horizon
        self.tda_window = tda_window
        self.tda_percentile = tda_percentile
        self.z_threshold = z_threshold
        self.z_quantile = z_quantile
        self.min_hold_bars = min_hold_bars if min_hold_bars > 0 else horizon
        self.direction_mode = direction_mode
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.htf_k = htf_k

        self.htf_ctx = HTFContext(k=htf_k)
        self.tda_detector = PersistenceLandscapeDetector()
        self.koopman_model = EDMDKoopmanModel(delay_shifts=3)
        self._adaptive_threshold: float = 0.0
        self._train_pred_corr: float = 0.0

    def _get_sliding_windows(self, data: np.ndarray, window_size: int) -> np.ndarray:
        shape = (data.shape[0] - window_size + 1, window_size, data.shape[1])
        strides = (data.strides[0], data.strides[0], data.strides[1])
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    def _predict_with_window(self, data: np.ndarray, i: int) -> float:
        n_min = self.koopman_model.n_min_rows
        window = data[max(0, i - n_min + 1): i + 1]
        return float(self.koopman_model.predict(window, steps=self.horizon).flatten()[0])

    def _compute_adaptive_threshold(self, train_scaled: np.ndarray) -> float:
        preds = [abs(self._predict_with_window(train_scaled, i))
                 for i in range(len(train_scaled) - self.horizon)]
        return float(np.percentile(preds, self.z_quantile * 100)) if preds else 0.0

    def _compute_train_corr_diag(self, train_scaled: np.ndarray) -> None:
        preds, actuals = [], []
        for i in range(min(200, len(train_scaled) - self.horizon)):
            preds.append(self._predict_with_window(train_scaled, i))
            actuals.append(float(train_scaled[i + self.horizon, 0]))
        if len(preds) > 5:
            c = np.corrcoef(preds, actuals)[0, 1]
            self._train_pred_corr = 0.0 if np.isnan(c) else float(c)

    def _bar_direction(self, test_data: pd.DataFrame, i: int) -> int:
        if self.direction_mode == "rsi":
            if "rsi_14" not in test_data.columns:
                return 0
            rsi = test_data["rsi_14"].iloc[i]
            if np.isnan(rsi):
                return 0
            if rsi < self.rsi_oversold:
                return 1
            if rsi > self.rsi_overbought:
                return -1
            return 0

        if "momentum_20" not in test_data.columns:
            return 0
        m = test_data["momentum_20"].iloc[i]
        if np.isnan(m) or m == 0.0:
            return 0
        s = int(np.sign(m))
        return -s if self.direction_mode == "contrarian" else s

    @staticmethod
    def _apply_hold_period(signals: list, min_hold: int) -> list:
        filtered, current_pos, hold_counter = [], 0, 0
        for sig in signals:
            if hold_counter > 0:
                filtered.append(current_pos)
                hold_counter -= 1
            else:
                if sig != current_pos:
                    current_pos = sig
                    hold_counter = min_hold - 1
                filtered.append(current_pos)
        return filtered

    def _build_htf_amplitude(self, train_data, test_data, feature_cols):
        try:
            ohlcv = ["open", "high", "low", "close", "volume"]
            extra = [c for c in ohlcv if c in train_data.columns]
            cols = list(dict.fromkeys(feature_cols + extra))

            train_htf = self.htf_ctx.build(train_data[cols])
            test_htf  = self.htf_ctx.build(test_data[cols])

            if train_htf.empty or test_htf.empty:
                return None, None

            htf_feat = [c for c in feature_cols if c in train_htf.columns]
            if len(htf_feat) < 2:
                return None, None

            from sklearn.preprocessing import StandardScaler as _SS
            sc = _SS()
            tr_sc = sc.fit_transform(train_htf[htf_feat].fillna(0).values)

            if len(tr_sc) < self.tda_window + 1:
                return None, None

            tr_wins = self._get_sliding_windows(tr_sc, self.tda_window)
            tr_amps  = self.tda_detector.fit_transform(tr_wins)
            threshold = float(np.percentile(tr_amps, self.tda_percentile))

            te_sc  = sc.transform(test_htf[htf_feat].fillna(0).values)
            ctx_sc = np.vstack([tr_sc[-(self.tda_window - 1):], te_sc])
            if len(ctx_sc) < self.tda_window:
                return None, None

            te_wins = self._get_sliding_windows(ctx_sc, self.tda_window)
            te_amps = self.tda_detector.transform(te_wins)

            htf_series = pd.Series(te_amps[:len(test_htf), 0], index=test_htf.index[:len(te_amps)])
            ltf_amps   = self.htf_ctx.align_to_ltf(htf_series, test_data.index, shift=True)
            return threshold, ltf_amps

        except Exception:
            return None, None

    def generate_signals(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.Series:
        candidate_cols = ["log_ret", "roll_measure", "ofi_proxy", "parkinson_vol",
                          "momentum_5", "momentum_20", "rsi_14"]
        feature_cols = [c for c in candidate_cols if c in train_data.columns]

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data[feature_cols].values)

        self.koopman_model.fit(train_scaled[:-self.horizon], train_scaled[self.horizon:],
                               horizon=self.horizon)

        effective_threshold = (self._compute_adaptive_threshold(train_scaled)
                               if self.z_threshold <= 0 else self.z_threshold)
        self._adaptive_threshold = effective_threshold
        self._compute_train_corr_diag(train_scaled)

        tda_threshold, ltf_amps = self._build_htf_amplitude(train_data, test_data, feature_cols)

        if tda_threshold is None:
            tr_wins = self._get_sliding_windows(train_scaled, self.tda_window)
            tr_amps = self.tda_detector.fit_transform(tr_wins)
            tda_threshold = float(np.percentile(tr_amps, self.tda_percentile))
            hist = train_data[feature_cols].iloc[-(self.tda_window - 1):]
            ctx  = scaler.transform(pd.concat([hist, test_data[feature_cols]]).values)
            te_wins = self._get_sliding_windows(ctx, self.tda_window)
            te_amps = self.tda_detector.transform(te_wins)
            ltf_amps = pd.Series(te_amps[:, 0], index=test_data.index)

        n_min    = self.koopman_model.n_min_rows
        test_sc  = scaler.transform(test_data[feature_cols].values)
        pred_ctx = np.vstack([train_scaled[-n_min:], test_sc])

        raw_signals = []
        for i in range(len(test_data)):
            amp = float(ltf_amps.iloc[i]) if isinstance(ltf_amps, pd.Series) else float(ltf_amps[i])
            if amp <= tda_threshold:
                raw_signals.append(0)
                continue

            direction = self._bar_direction(test_data, i)
            if direction == 0:
                raw_signals.append(0)
                continue

            pred_z = float(self.koopman_model.predict(
                pred_ctx[i: i + n_min], steps=self.horizon).flatten()[0])
            if abs(pred_z) <= effective_threshold:
                raw_signals.append(0)
                continue

            raw_signals.append(direction)

        return pd.Series(self._apply_hold_period(raw_signals, self.min_hold_bars),
                         index=test_data.index)