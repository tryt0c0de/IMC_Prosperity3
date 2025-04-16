from datamodel import OrderDepth, TradingState, Order
from typing import List
from collections import deque
import numpy as np
#from Logger import Logger

#logger = Logger()

class Trader:

    def __init__(self, parameters=[]):
        # Coefficients from Lasso or ElasticNet
        self.coefs = [
            6.95e-06,      # spread_1
            -2.747e-05,    # total_bid_vol
            3.043e-05,     # total_ask_vol
            0.0,           # volume_imbalance (dropped)
            -7.831e-05,    # momentum
            0.0,           # spread_change (dropped)
            -1.9372e-04,   # momentum_lag1
            -2.3225e-04,   # price_change
            3.157e-05,     # volatility_10
            2.4886e-04     # bid_ask_ratio
        ]
        self.intercept = -0.00036157

        # Rolling buffers for momentum and volatility
        self.midprice_buffer = deque(maxlen=30)
        self.momentum_buffer = deque(maxlen=2)

        self.buy_threshold = 0.0005
        self.sell_threshold = -0.0005
        self.order_size = 10

    def run(self, state: TradingState):
        result = {}

        for product, order_depth in state.order_depths.items():
            if product != "KELP":
                continue

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2

            self.midprice_buffer.append(mid_price)
            if len(self.midprice_buffer) < 10:
                result[product] = []
                continue

            # Compute features
            spread_1 = best_ask - best_bid
            total_bid_vol = sum(abs(v) for v in order_depth.buy_orders.values())
            total_ask_vol = sum(abs(v) for v in order_depth.sell_orders.values())
            volume_imbalance = total_bid_vol / (total_bid_vol + total_ask_vol + 1e-9)
            rolling_mean_mid = np.mean(list(self.midprice_buffer)[-5:])
            momentum = mid_price - rolling_mean_mid
            self.momentum_buffer.append(momentum)
            momentum_lag1 = self.momentum_buffer[0] if len(self.momentum_buffer) == 2 else 0.0
            price_change = self.midprice_buffer[-1] - self.midprice_buffer[-2]
            volatility_10 = np.std(list(self.midprice_buffer)[-10:])
            bid_ask_ratio = total_bid_vol / (total_ask_vol + 1e-9)

            # We don’t have spread history in buffer → set spread_change = 0.0
            spread_change = 0.0

            # Feature vector in same order as training
            features = [
                spread_1,
                total_bid_vol,
                total_ask_vol,
                volume_imbalance,
                momentum,
                spread_change,
                momentum_lag1,
                price_change,
                volatility_10,
                bid_ask_ratio
            ]

            predicted_return = self.intercept + np.dot(self.coefs, features)

            orders: List[Order] = []
            if predicted_return > self.buy_threshold:
                orders.append(Order(product, best_ask, self.order_size))
            elif predicted_return < self.sell_threshold:
                orders.append(Order(product, best_bid, -self.order_size))

            result[product] = orders

        conversions = 1
        traderData = "v3_lasso_full_features"
        #logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

