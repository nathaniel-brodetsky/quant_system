import pandas as pd
from binance.client import Client
from src.core.interfaces import DataLoader


class BinanceLoader(DataLoader):
    def __init__(self):
        self.client = Client()

    def fetch_historical(
            self,
            symbol: str,
            start: str,
            end: str = None,
            interval: str = Client.KLINE_INTERVAL_1MINUTE,
    ) -> pd.DataFrame:
        klines = self.client.get_historical_klines(
            symbol, interval, start_str=start, end_str=end
        )

        columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "n_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
        ]

        df = pd.DataFrame(klines, columns=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        numeric_cols = [
            "open", "high", "low", "close", "volume",
            "quote_asset_volume", "n_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol",
        ]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df.drop(columns=["close_time", "ignore"], inplace=True)

        return df
