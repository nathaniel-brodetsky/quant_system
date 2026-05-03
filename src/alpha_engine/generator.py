import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.koopman.edmd import EDMDKoopmanModel
from src.regime_detection.tda_detector import PersistenceLandscapeDetector


class HybridAlphaEngine:
    """
    Оркестратор сигналов. Объединяет предсказательную силу оператора Купмана
    и фильтрацию режимов через Топологический Анализ Данных.
    """

    def __init__(self, tda_window: int = 50, tda_percentile: float = 85.0):
        self.tda_window = tda_window
        self.tda_percentile = tda_percentile

        self.tda_detector = PersistenceLandscapeDetector()
        self.koopman_model = EDMDKoopmanModel(polynomial_degree=2)

    def _get_sliding_windows(self, data: np.ndarray, window_size: int) -> np.ndarray:
        shape = (data.shape[0] - window_size + 1, window_size, data.shape[1])
        strides = (data.strides[0], data.strides[0], data.strides[1])
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    def generate_signals(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.Series:
        feature_cols = ['log_ret', 'roll_measure', 'ofi_proxy', 'parkinson_vol']

        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_data[feature_cols].values)

        X_train = train_features_scaled[:-1]
        Y_train = train_features_scaled[1:]
        self.koopman_model.fit(X_train, Y_train)

        train_windows = self._get_sliding_windows(train_features_scaled, self.tda_window)
        train_amplitudes = self.tda_detector.fit_transform(train_windows)
        tda_threshold = np.percentile(train_amplitudes, self.tda_percentile)

        signals = []

        history_for_test = train_data.iloc[-(self.tda_window - 1):]
        full_test_context = pd.concat([history_for_test, test_data])

        test_features_scaled = scaler.transform(full_test_context[feature_cols].values)
        test_windows = self._get_sliding_windows(test_features_scaled, self.tda_window)

        test_amplitudes = self.tda_detector.fit_transform(test_windows)

        test_x_scaled = scaler.transform(test_data[feature_cols].values)

        for i in range(len(test_data)):
            current_x = test_x_scaled[i]
            current_amp = np.ravel(test_amplitudes[i])[0]

            predicted_next_state = self.koopman_model.predict(current_x, steps=1)
            predicted_return_z = predicted_next_state.flatten()[0]

            is_trend_regime = current_amp > tda_threshold

            if is_trend_regime:
                if predicted_return_z > 0.1:
                    signal = 1
                elif predicted_return_z < -0.1:
                    signal = -1
                else:
                    signal = 0
            else:
                signal = 0

            signals.append(signal)

        return pd.Series(signals, index=test_data.index)