from datamodel import Order, OrderDepth, TradingState
from typing import List
import jsonpickle
import joblib
import numpy as np
from Logger import Logger

class Trader:
    def __init__(self):
        self.model = joblib.load("/Users/difa/Desktop/IMC_Prosperity3/Pippo/rainforest_return_model.pkl")
        self.features = [
            "spread_rainforest_resin", "imbalance_rainforest_resin", "return_1_rainforest_resin", "volatility_5_rainforest_resin",
            "spread_kelp", "imbalance_kelp", "return_1_kelp", "volatility_5_kelp",
            "spread_squid_ink", "imbalance_squid_ink", "return_1_squid_ink", "volatility_5_squid_ink"
        ]
        self.last_mid_prices = {p: [] for p in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]}
        self.position_limits = {"RAINFOREST_RESIN": 50}
        self.threshold = 0.3  # Only trade if predicted return > cost
        self.logger = Logger()

    def calculate_features(self, state: TradingState) -> dict:
        result = {}
        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            depth = state.order_depths.get(product)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                return None

            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)
            spread = best_ask - best_bid
            mid = (best_bid + best_ask) / 2
            result[f"spread_{product.lower()}"] = spread

            bid_vol = depth.buy_orders[best_bid]
            ask_vol = depth.sell_orders[best_ask]
            result[f"imbalance_{product.lower()}"] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)

            self.last_mid_prices[product].append(mid)
            if len(self.last_mid_prices[product]) > 5:
                self.last_mid_prices[product] = self.last_mid_prices[product][-5:]

            prices = self.last_mid_prices[product]
            result[f"return_1_{product.lower()}"] = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0
            result[f"volatility_5_{product.lower()}"] = np.std(prices) if len(prices) >= 2 else 0

        return result

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        traderData = jsonpickle.encode({})
        product = "RAINFOREST_RESIN"

        features_dict = self.calculate_features(state)
        if not features_dict:
            return result, conversions, traderData

        feature_array = np.array([features_dict[f] for f in self.features]).reshape(1, -1)
        predicted_return = self.model.predict(feature_array)[0]

        depth = state.order_depths[product]
        position = state.position.get(product, 0)
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        if predicted_return > self.threshold and position < self.position_limits[product]:
            result[product] = [Order(product, best_ask, 10)]
        elif predicted_return < -self.threshold and position > -self.position_limits[product]:
            result[product] = [Order(product, best_bid, -10)]

        self.logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
