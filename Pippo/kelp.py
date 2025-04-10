from datamodel import Order, OrderDepth, TradingState
from typing import List
import numpy as np
import jsonpickle
from collections import deque
from Logger import Logger

class Trader:
    def __init__(self):
        self.logger = Logger()
        self.kelp_prices = []
        self.kelp_returns = []
        self.lag = 5  # AR(5)
        self.cooldown_ticks = 10
        self.last_trade_tick = -999
        self.entry_threshold = 0.2  # minimum predicted return to act
        self.max_position = 50
        self.position = 0
        self.last_direction = None
        self.entry_price = 0
        self.realized_pnl = 0

    def get_mid_price(self, depth: OrderDepth):
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        return (best_bid + best_ask) / 2

    def predict_return(self):
        if len(self.kelp_returns) < self.lag:
            return 0
        X = np.array(self.kelp_returns[-self.lag:]).reshape(1, -1)
        beta = np.linalg.pinv(X.T) @ np.array(self.kelp_returns[-self.lag:])
        prediction = float(X @ beta)
        return prediction

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        traderData = jsonpickle.encode({})
        timestamp = state.timestamp

        product = "KELP"
        depth = state.order_depths.get(product)
        if not depth:
            return result, conversions, traderData

        mid_price = self.get_mid_price(depth)
        self.kelp_prices.append(mid_price)

        if len(self.kelp_prices) > 1:
            ret = mid_price - self.kelp_prices[-2]
            self.kelp_returns.append(ret)

        predicted_ret = self.predict_return()

        orders = []
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        if timestamp - self.last_trade_tick > self.cooldown_ticks:
            if predicted_ret > self.entry_threshold and self.position < self.max_position:
                size = min(10, self.max_position - self.position)
                orders.append(Order(product, best_ask, size))
                self.position += size
                self.entry_price = best_ask
                self.last_trade_tick = timestamp
                self.last_direction = "long"

            elif predicted_ret < -self.entry_threshold and self.position > -self.max_position:
                size = min(10, self.max_position + self.position)
                orders.append(Order(product, best_bid, -size))
                self.position -= size
                self.entry_price = best_bid
                self.last_trade_tick = timestamp
                self.last_direction = "short"

        elif self.last_direction == "long" and predicted_ret < 0:
            orders.append(Order(product, best_bid, -self.position))
            self.realized_pnl += (best_bid - self.entry_price) * self.position
            self.position = 0
            self.last_trade_tick = timestamp

        elif self.last_direction == "short" and predicted_ret > 0:
            orders.append(Order(product, best_ask, -self.position))
            self.realized_pnl += (self.entry_price - best_ask) * (-self.position)
            self.position = 0
            self.last_trade_tick = timestamp

        result[product] = orders
        self.logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
