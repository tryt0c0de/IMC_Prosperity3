from datamodel import Order, OrderDepth, TradingState
from typing import List
import numpy as np
import jsonpickle
from collections import deque
from Logger import Logger

class Trader:
    def __init__(self):
        self.logger = Logger()
        self.kelp_prices = deque(maxlen=500)
        self.squid_prices = deque(maxlen=500)
        self.residuals = deque(maxlen=300)
        self.position_limits = {"KELP": 50, "SQUID_INK": 50}
        self.entry_z = 1.5
        self.exit_z = 0.5
        self.min_volume = 10
        self.min_corr = 0.6
        self.cooldown_ticks = 15
        self.last_trade_tick = -999
        self.open_trades = []
        self.realized_pnl = 0

        # Kalman filter state
        self.beta = 1.0
        self.P = 1.0
        self.Q = 0.001
        self.R = 0.2

    def kalman_update(self, x, y):
        pred_y = self.beta * x
        e = y - pred_y
        K = self.P * x / (self.R + x * self.P * x)
        self.beta += K * e
        self.P = (1 - K * x) * self.P + self.Q

    def vwap_mid(self, depth):
        bids = sorted(depth.buy_orders.items(), reverse=True)[:3]
        asks = sorted(depth.sell_orders.items())[:3]
        bid_vwap = np.sum([p * v for p, v in bids]) / (np.sum([v for _, v in bids]) + 1e-6)
        ask_vwap = np.sum([p * v for p, v in asks]) / (np.sum([v for _, v in asks]) + 1e-6)
        return (bid_vwap + ask_vwap) / 2

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        traderData = jsonpickle.encode({})
        timestamp = state.timestamp

        kelp = "KELP"
        squid = "SQUID_INK"
        kd = state.order_depths.get(kelp)
        sd = state.order_depths.get(squid)
        if not kd or not sd:
            return result, conversions, traderData

        km = self.vwap_mid(kd)
        sm = self.vwap_mid(sd)
        self.kelp_prices.append(km)
        self.squid_prices.append(sm)

        self.kalman_update(sm, km)
        hedge_ratio = self.beta
        residual = km - hedge_ratio * sm
        self.residuals.append(residual)

        kelp_pos = state.position.get(kelp, 0)
        squid_pos = state.position.get(squid, 0)

        if len(self.kelp_prices) < 100:
            return result, conversions, traderData
        corr = np.corrcoef(self.kelp_prices, self.squid_prices)[0, 1]
        if abs(corr) < self.min_corr:
            return result, conversions, traderData

        z = (residual - np.mean(self.residuals)) / (np.std(self.residuals) + 1e-6)
        size = int(min(25, max(5, abs(z) * 10)))

        kelp_bid = max(kd.buy_orders)
        kelp_ask = min(kd.sell_orders)
        squid_bid = max(sd.buy_orders)
        squid_ask = min(sd.sell_orders)

        kelp_orders, squid_orders = [], []

        # === ENTRY ===
        if timestamp - self.last_trade_tick > self.cooldown_ticks:
            if z > self.entry_z and kelp_pos > -self.position_limits[kelp]:
                kelp_orders.append(Order(kelp, kelp_bid, -size))        # SELL KELP at bid
                squid_orders.append(Order(squid, squid_ask, size))      # BUY SQUID at ask
                self.open_trades.append(("short_kelp", kelp_bid, squid_ask, size, timestamp))
                self.last_trade_tick = timestamp

            elif z < -self.entry_z and kelp_pos < self.position_limits[kelp]:
                kelp_orders.append(Order(kelp, kelp_ask, size))         # BUY KELP at ask
                squid_orders.append(Order(squid, squid_bid, -size))     # SELL SQUID at bid
                self.open_trades.append(("long_kelp", kelp_ask, squid_bid, size, timestamp))
                self.last_trade_tick = timestamp

        # === EXIT ===
        elif abs(z) < self.exit_z:
            if kelp_pos != 0:
                exit_price_kelp = kelp_bid if kelp_pos > 0 else kelp_ask
                kelp_orders.append(Order(kelp, exit_price_kelp, -kelp_pos))

            if squid_pos != 0:
                exit_price_squid = squid_bid if squid_pos > 0 else squid_ask
                squid_orders.append(Order(squid, exit_price_squid, -squid_pos))

            for side, entry_kelp_px, entry_squid_px, qty, t0 in self.open_trades:
                if side == "long_kelp":
                    exit_kelp_px = kelp_bid
                    exit_squid_px = squid_ask
                    pnl = (exit_kelp_px - entry_kelp_px) * qty + (entry_squid_px - exit_squid_px) * qty
                else:
                    exit_kelp_px = kelp_ask
                    exit_squid_px = squid_bid
                    pnl = (entry_kelp_px - exit_kelp_px) * qty + (exit_squid_px - entry_squid_px) * qty
                self.realized_pnl += pnl

            self.open_trades.clear()
            self.last_trade_tick = timestamp

        result[kelp] = kelp_orders
        result[squid] = squid_orders
        self.logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    