from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import numpy as np
# from Logger import Logger

# logger = Logger()

class Trader:
    def __init__(self):
        self.vouchers = [9500, 9750, 10000, 10250, 10500]
        self.products = [f"VOLCANIC_ROCK_VOUCHER_{s}" for s in self.vouchers]
        self.max_position = 100  # Hedge-capacity per leg
        self.base_size = 1       # Per-leg execution size
        self.min_spread = 0      # Lowered Minimum profit threshold for safer fills

    def run(self, state: TradingState):
        result = {product: [] for product in self.products}

        bids = {}
        asks = {}
        bid_vols = {}
        ask_vols = {}
        positions = state.position or {}

        for product in self.products:
            if product not in state.order_depths:
                continue

            depth = state.order_depths[product]
            if depth.buy_orders:
                bids[product] = max(depth.buy_orders.keys())
                bid_vols[product] = depth.buy_orders[bids[product]]
            if depth.sell_orders:
                asks[product] = min(depth.sell_orders.keys())
                ask_vols[product] = depth.sell_orders[asks[product]]

        best_arb = None
        best_profit = 0

        for i in range(len(self.vouchers) - 1):
            lower = f"VOLCANIC_ROCK_VOUCHER_{self.vouchers[i]}"
            upper = f"VOLCANIC_ROCK_VOUCHER_{self.vouchers[i+1]}"

            if lower in bids and upper in asks:
                spread = bids[lower] - asks[upper]
                if spread > best_profit:
                    best_arb = (lower, upper, bids[lower], asks[upper], bid_vols[lower], ask_vols[upper])
                    best_profit = spread

        if best_arb and best_profit >= self.min_spread:
            low_prod, high_prod, bid_low, ask_high, vol_low, vol_high = best_arb
            pos_low = positions.get(low_prod, 0)
            pos_high = positions.get(high_prod, 0)

            size = min(
                self.base_size,
                vol_low,
                vol_high,
                self.max_position - abs(pos_low),
                self.max_position - abs(pos_high)
            )

            if size > 0:
                result[low_prod].append(Order(low_prod, int(bid_low), -size))  # Sell at best bid
                result[high_prod].append(Order(high_prod, int(ask_high), size))  # Buy at best ask

        conversions = 0
        trader_data = jsonpickle.encode({})
        # logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data