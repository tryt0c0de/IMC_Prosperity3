from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import numpy as np
from Logger import Logger
from collections import deque

logger = Logger()

class Trader:
    def __init__(self):
        # ----- Single-Product: JAMS -----
        self.product = "JAMS"
        
        # ----- Market Making Base Parameters -----
        self.max_position = 30        # Maximum inventory: ±30
        self.base_spread = 2.0        # Base spread around reference price
        self.skew_param = 0.2         # Inventory-based skew factor
        self.base_size = 1          # Base order size for passive quotes

        # We track mid-prices to compute momentum
        self.price_history = deque(maxlen=50)

        # ----- Momentum Overlay -----
        self.short_window = 5         # short-term MA
        self.long_window  = 15        # long-term MA
        self.momentum_band = 0.0      # if short_ma - long_ma is within +/- band, do nothing
        self.momentum_factor = 0.5    # how strongly we shift our *passive* quotes if we do
        self.take_threshold = 2.0     # if |short_ma - long_ma| is beyond this, we'll "take"

        # We also set a "take size" – how many we aggressively buy or sell
        self.take_size = 5

        self.tick = 0

    def run(self, state: TradingState):
        """
        Hybrid strategy that passively market-makes on JAMS and aggressively 'takes'
        if momentum is strong enough.
        """
        self.tick += 1
        
        # We'll store only JAMS orders in the dict
        result = {self.product: []}

        # 1) If we have no order book data for JAMS, skip
        if self.product not in state.order_depths:
            return result, 0, ""

        order_depth = state.order_depths[self.product]
        mid_price = self.get_mid_price(order_depth)
        if mid_price is None:
            return result, 0, ""

        # 2) Update rolling mid-price, compute momentum signals if enough data
        self.price_history.append(mid_price)
        prices = list(self.price_history)

        # Initialize momentum booleans & diff
        bullish_momentum = False
        bearish_momentum = False
        momentum_diff = 0.0

        if len(prices) >= self.long_window:
            short_ma = np.mean(prices[-self.short_window:])
            long_ma = np.mean(prices[-self.long_window:])
            momentum_diff = short_ma - long_ma

            if momentum_diff > self.momentum_band:
                bullish_momentum = True
            elif momentum_diff < -self.momentum_band:
                bearish_momentum = True

        # 3) Inventory-based logic for market making
        current_pos = state.position.get(self.product, 0)

        # We incorporate momentum *slightly* into the adjusted mid
        # If momentum is positive => shift mid up, negative => shift mid down
        # The shift is "momentum_factor * momentum_diff"
        # If momentum_diff is small or within band, shift is 0
        momentum_shift = 0.0
        if bullish_momentum:
            momentum_shift = self.momentum_factor * momentum_diff
        elif bearish_momentum:
            momentum_shift = self.momentum_factor * momentum_diff

        inv_skew = self.skew_param * current_pos
        adjusted_mid = mid_price - inv_skew + momentum_shift

        # 4) Build final quotes for passive market-making
        half_spread = self.base_spread / 2.0
        bid_price = round(adjusted_mid - half_spread, 2)
        ask_price = round(adjusted_mid + half_spread, 2)

        buy_size = self.base_size
        sell_size = self.base_size

        # If near max_position, reduce buy_size
        if current_pos >= (self.max_position - self.base_size):
            buy_size = 1
        if current_pos <= -(self.max_position - self.base_size):
            sell_size = 1

        # Passive Orders
        mm_orders = []
        if current_pos < self.max_position and bid_price > 0:
            mm_orders.append(Order(self.product, bid_price, buy_size))

        if current_pos > -self.max_position and ask_price > 0:
            mm_orders.append(Order(self.product, ask_price, -sell_size))

        # 5) "Taking" – if momentum is strong enough, we cross the spread.
        # e.g., if momentum_diff > take_threshold => aggressively buy from the best ask
        # or if momentum_diff < -take_threshold => aggressively sell at best bid
        take_orders = []
        if len(prices) >= self.long_window:
            # We have a valid momentum reading
            best_bid, best_ask = self.get_best_bid_ask(order_depth)

            # If momentum is strongly bullish and we have room to buy
            if momentum_diff > self.take_threshold and current_pos < self.max_position:
                # Aggressive buy => take from the best_ask
                # We'll place an order with price >= best_ask to ensure immediate fill
                # e.g. price = best_ask + 0.01
                if best_ask is not None and best_ask > 0:
                    qty = min(self.take_size, self.max_position - current_pos)
                    # We cross the ask with a limit slightly above best_ask
                    take_price = best_ask + 0.01
                    take_orders.append(Order(self.product, take_price, qty))

            # If momentum is strongly bearish and we have room to sell
            if momentum_diff < -self.take_threshold and current_pos > -self.max_position:
                # Aggressive sell => take from the best_bid
                if best_bid is not None and best_bid > 0:
                    qty = min(self.take_size, current_pos - (-self.max_position))
                    # Cross the bid with a limit slightly below best_bid
                    take_price = best_bid - 0.01
                    take_orders.append(Order(self.product, take_price, -qty))

        # Combine passive + taking orders
        all_orders = mm_orders + take_orders
        result[self.product] = all_orders

        conversions = 0
        trader_data = jsonpickle.encode({})
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def get_mid_price(self, order_depth: OrderDepth):
        """Compute midpoint of best bid/ask if both exist."""
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return None

    def get_best_bid_ask(self, order_depth: OrderDepth):
        """Return (best_bid, best_ask)."""
        best_bid = None
        best_ask = None
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
        return best_bid, best_ask
