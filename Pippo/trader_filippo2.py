from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import numpy as np
#from Logger import Logger
from collections import deque

#logger = Logger()

class Trader:
    def __init__(self):
        self.position_limits = {"RAINFOREST_RESIN": 50}
        self.max_order_size = 100
        self.base_edge = 1.0
        self.min_spread = {"RAINFOREST_RESIN": 1.5}

        self.edge = self.base_edge
        self.fill_window = deque(maxlen=50)
        self.pnl_history = deque(maxlen=2)
        self.last_update_ts = 0

    def adapt_edge(self, timestamp, current_pnl):
        if timestamp - self.last_update_ts < 50:
            return
        self.last_update_ts = timestamp

        recent_fills = sum(self.fill_window)
        fill_rate = recent_fills / len(self.fill_window) if self.fill_window else 0

        # Record pnl change
        if len(self.pnl_history) == 2:
            pnl_change = current_pnl - self.pnl_history[0]
            self.pnl_history.popleft()
            self.pnl_history.append(current_pnl)
        else:
            self.pnl_history.append(current_pnl)
            return

        # Adaptive logic with tight constraints
        if fill_rate < 0.1 or pnl_change < 0:
            self.edge = min(self.edge + 0.05, 1.2)
        elif fill_rate > 0.5 and pnl_change > 0:
            self.edge = max(self.edge - 0.05, 0.9)

    def track_pnl(self, state, product):
        pnl = 0
        if product in state.own_trades:
            for trade in state.own_trades[product]:
                pnl += -trade.quantity * trade.price
        return pnl

    def market_make(self, product, order_depth, position, mid_price, spread):
        orders = []

        inventory_skew = position / self.position_limits[product]
        edge = self.edge + abs(inventory_skew * 2) + spread * 0.1

        bid_price = int(mid_price - edge)
        ask_price = int(mid_price + edge)

        buy_limit = self.position_limits[product] - position
        sell_limit = self.position_limits[product] + position

        buy_qty = min(self.max_order_size, max(1, int((1 - inventory_skew) * self.max_order_size)), buy_limit)
        sell_qty = min(self.max_order_size, max(1, int((1 + inventory_skew) * self.max_order_size)), sell_limit)

        if buy_qty > 0:
            orders.append(Order(product, bid_price, buy_qty))
            self.fill_window.append(1)
        else:
            self.fill_window.append(0)

        if sell_qty > 0:
            orders.append(Order(product, ask_price, -sell_qty))
            self.fill_window.append(1)
        else:
            self.fill_window.append(0)

        return orders

    def run(self, state: TradingState):
        result = {}
        timestamp = state.timestamp

        for product in ["RAINFOREST_RESIN"]:
            if product not in state.order_depths:
                continue

            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)

            if not order_depth.sell_orders or not order_depth.buy_orders:
                continue

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            spread = best_ask - best_bid

            if spread < self.min_spread[product]:
                continue

            mid_price = (best_ask + best_bid) / 2
            current_pnl = self.track_pnl(state, product)

            self.adapt_edge(timestamp, current_pnl)

            result[product] = self.market_make(product, order_depth, position, mid_price, spread)

        traderData = jsonpickle.encode({})
        conversions = 0
        #logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
