from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import numpy as np
# from Logger import Logger
from collections import deque

# logger = Logger()

class Trader:
    def __init__(self):
        self.p1 = "VOLCANIC_ROCK_VOUCHER_10500"
        self.p2 = "VOLCANIC_ROCK_VOUCHER_10000"

        self.max_position = 1000
        self.base_spread = 2.0
        self.skew_param = 0.2
        self.base_size = 1

        self.spread_window = 1000
        self.entry_threshold = 1.5
        self.exit_threshold = 0.2
        self.position_size = 1
        self.spread_buffer = 1.0  # Execution cost buffer

        self.p1_history = deque(maxlen=self.spread_window)
        self.p2_history = deque(maxlen=self.spread_window)

        self.tick = 0

    def run(self, state: TradingState):
        self.tick += 1
        result = {self.p1: [], self.p2: []}

        if self.p1 not in state.order_depths or self.p2 not in state.order_depths:
            return result, 0, ""

        p1_depth = state.order_depths[self.p1]
        p2_depth = state.order_depths[self.p2]

        mid_10500 = self.get_mid_price(p1_depth)
        mid_10000 = self.get_mid_price(p2_depth)

        if mid_10500 is None or mid_10000 is None:
            return result, 0, ""

        self.p1_history.append(mid_10500)
        self.p2_history.append(mid_10000)

        spread = mid_10500 - mid_10000
        mean = np.mean(np.array(self.p1_history) - np.array(self.p2_history))
        std = np.std(np.array(self.p1_history) - np.array(self.p2_history))

        p1_pos = state.position.get(self.p1, 0)
        p2_pos = state.position.get(self.p2, 0)

        size_factor = min(max(1, abs((spread - mean) / std)), 5) if std > 0 else 1
        trade_size = int(self.base_size * size_factor)

        if std > 0:
            if spread > mean + self.entry_threshold * std + self.spread_buffer:
                if p1_pos > -self.max_position and p2_pos < self.max_position:
                    result[self.p1].append(Order(self.p1, int(round(mid_10500 - 1)), -trade_size))
                    result[self.p2].append(Order(self.p2, int(round(mid_10000 + 1)), trade_size))

            elif spread < mean - self.entry_threshold * std - self.spread_buffer:
                if p1_pos < self.max_position and p2_pos > -self.max_position:
                    result[self.p1].append(Order(self.p1, int(round(mid_10500 + 1)), trade_size))
                    result[self.p2].append(Order(self.p2, int(round(mid_10000 - 1)), -trade_size))

            elif abs(spread - mean) < self.exit_threshold * std:
                if p1_pos > 0:
                    result[self.p1].append(Order(self.p1, int(round(mid_10500 - 1)), -p1_pos))
                if p1_pos < 0:
                    result[self.p1].append(Order(self.p1, int(round(mid_10500 + 1)), -p1_pos))
                if p2_pos > 0:
                    result[self.p2].append(Order(self.p2, int(round(mid_10000 - 1)), -p2_pos))
                if p2_pos < 0:
                    result[self.p2].append(Order(self.p2, int(round(mid_10000 + 1)), -p2_pos))

        inv_skew = self.skew_param * p1_pos
        adjusted_mid = mid_10500 - inv_skew
        half_spread = self.base_spread / 2
        bid_price = int(round(adjusted_mid - half_spread))
        ask_price = int(round(adjusted_mid + half_spread))

        buy_size = self.base_size if p1_pos < self.max_position else 1
        sell_size = self.base_size if p1_pos > -self.max_position else 1

        if bid_price > 0:
            result[self.p1].append(Order(self.p1, bid_price, buy_size))
        if ask_price > 0:
            result[self.p1].append(Order(self.p1, ask_price, -sell_size))

        conversions = 0
        trader_data = jsonpickle.encode({})
        # logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def get_mid_price(self, order_depth: OrderDepth):
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return None