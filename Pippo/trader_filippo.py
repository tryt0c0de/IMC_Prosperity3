from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import numpy as np
#from Logger import Logger
import collections

#logger = Logger()

class Trader:
    def __init__(self):
        self.position_limits = {"KELP": 50, "RAINFOREST_RESIN": 50}
        self.base_edge = 1
        self.min_spread = {"KELP": 2, "RAINFOREST_RESIN": 1.5}
        self.max_order_size = 50
        self.price_history = collections.defaultdict(lambda: collections.deque(maxlen=10))

    def market_make_refined(self, product: str, order_depth: OrderDepth, position: int) -> List[Order]:
        orders = []
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders

        best_ask = min(order_depth.sell_orders)
        best_bid = max(order_depth.buy_orders)
        spread = best_ask - best_bid

        if spread < self.min_spread[product]:
            return orders

        mid_price = (best_ask + best_bid) / 2
        self.price_history[product].append(mid_price)

        # Volatility based on recent mid-price std
        if len(self.price_history[product]) >= 5:
            volatility = np.std(self.price_history[product])
        else:
            volatility = 0

        # Inventory skew and edge
        inventory_skew = position / self.position_limits[product]
        edge = self.base_edge + abs(inventory_skew * 2) + spread * 0.1 + volatility * (-0.25)
        if volatility > 2:
            edge += 0.1 * volatility


        bid_price = int(mid_price - edge)
        ask_price = int(mid_price + edge)

        # Dynamic sizing
        buy_limit = self.position_limits[product] - position
        sell_limit = self.position_limits[product] + position

        buy_qty = min(self.max_order_size, max(1, int((1 - inventory_skew) * self.max_order_size)), buy_limit)
        sell_qty = min(self.max_order_size, max(1, int((1 + inventory_skew) * self.max_order_size)), sell_limit)

        if buy_qty > 0:
            orders.append(Order(product, bid_price, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, ask_price, -sell_qty))

        return orders

    def run(self, state: TradingState):
        result = {}

        for product in ["RAINFOREST_RESIN"]:
            if product in state.order_depths:
                pos = state.position.get(product, 0)
                result[product] = self.market_make_refined(product, state.order_depths[product], pos)

        traderData = jsonpickle.encode({})
        conversions = 0
        #logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
