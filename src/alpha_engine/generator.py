import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.koopman.edmd import EDMDKoopmanModel
from src.regime_detection.tda_detector import PersistenceLandscapeDetector


class HybridAlphaEngine:
    def __init__(
            self,
            horizon: int = 5,
            tda_window: int = 50,
            tda_percentile: float = 85.0,
            z_threshold: float = 0.0,
            z_quantile: float = 0.6,
            min_corr_threshold: float = 0.02,
    ):
        self.horizon = horizon
        self.tda_window = tda_window
        self.tda_percentile = tda_percentile
        self.z_threshold = z_threshold
        self.z_quantile = z_quantile
        self.min_corr_threshold = min_corr_threshold

        self.tda_detector = PersistenceLandscapeDetector()
        self.koopman_model = EDMDKoopmanModel(delay_shifts=3)

        self._adaptive_threshold: float = 0.0
        self._train_pred_std: float = 0.0
        self._signal_direction: float = 0.0
        self._train_pred_corr: float = 0.0

    def _get_sliding_windows(self, data: np.ndarray, window_size: int) -> np.ndarray:
        shape = (data.shape[0] - window_size + 1, window_size, data.shape[1])
        strides = (data.strides[0], data.strides[0], data.strides[1])
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    def _compute_adaptive_threshold(self, train_features_scaled: np.ndarray) -> float:
        preds = []
        for i in range(len(train_features_scaled) - self.horizon):
            p = self.koopman_model.predict(train_features_scaled[i], steps=self.horizon)
            preds.append(abs(float(p.flatten()[0])))
        preds = np.array(preds)
        self._train_pred_std = float(np.std(preds))
        return float(np.percentile(preds, self.z_quantile * 100))

    def _compute_signal_direction(self, train_features_scaled: np.ndarray) -> float:
        preds, actuals = [], []
        for i in range(len(train_features_scaled) - self.horizon):
            p = self.koopman_model.predict(train_features_scaled[i], steps=self.horizon)
            preds.append(float(p.flatten()[0]))
            actuals.append(float(train_features_scaled[i + self.horizon, 0]))

        corr = float(np.corrcoef(preds, actuals)[0, 1])
        self._train_pred_corr = corr

        if abs(corr) < self.min_corr_threshold:
            return 0.0
        return float(np.sign(corr))

    def generate_signals(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.Series:
        candidate_cols = [
            "log_ret", "roll_measure", "ofi_proxy", "parkinson_vol",
            "momentum_5", "momentum_20", "rsi_14",
        ]
        feature_cols = [c for c in candidate_cols if c in train_data.columns]

        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_data[feature_cols].values)

        X_train = train_features_scaled[: -self.horizon]
        Y_train = train_features_scaled[self.horizon:]
        self.koopman_model.fit(X_train, Y_train, horizon=self.horizon)

        if self.z_threshold <= 0:
            effective_threshold = self._compute_adaptive_threshold(train_features_scaled)
        else:
            effective_threshold = self.z_threshold
        self._adaptive_threshold = effective_threshold

        signal_direction = self._compute_signal_direction(train_features_scaled)
        self._signal_direction = signal_direction

        train_windows = self._get_sliding_windows(train_features_scaled, self.tda_window)
        train_amplitudes = self.tda_detector.fit_transform(train_windows)
        tda_threshold = np.percentile(train_amplitudes, self.tda_percentile)

        history_for_test = train_data[feature_cols].iloc[-(self.tda_window - 1):]
        full_test_context = pd.concat([history_for_test, test_data[feature_cols]])
        test_features_scaled = scaler.transform(full_test_context.values)
        test_windows = self._get_sliding_windows(test_features_scaled, self.tda_window)
        test_amplitudes = self.tda_detector.transform(test_windows)

        test_x_scaled = scaler.transform(test_data[feature_cols].values)

        signals = []
        for i in range(len(test_data)):
            if signal_direction == 0.0:
                signals.append(0)
                continue

            current_x = test_x_scaled[i]
            current_amp = float(np.ravel(test_amplitudes[i])[0])

            predicted_next_state = self.koopman_model.predict(current_x, steps=self.horizon)
            predicted_return_z = float(predicted_next_state.flatten()[0])

            is_active_regime = current_amp > tda_threshold

            if is_active_regime:
                if predicted_return_z > effective_threshold:
                    signal = int(signal_direction)
                elif predicted_return_z < -effective_threshold:
                    signal = int(-signal_direction)
                else:
                    signal = 0
            else:
                signal = 0

            signals.append(signal)

        return pd.Series(signals, index=test_data.index)
