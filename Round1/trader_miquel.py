from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import numpy as np
from Logger import Logger

logger = Logger()

class Trader:
    def __init__(self,parameters = []):
        self.position_limits = {"SQUID_INK": 50}
        self.max_order_size = 25
        self.recent_mid_prices = []
        self.rolling_window = int(parameters[0])
        self.cluster_labels = []  # 0 = insignificant, 1 = neg autocorr, 2 = pos autocorr

    def compute_mid_price(self, order_depth: OrderDepth) -> float:
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders)
            best_bid = max(order_depth.buy_orders)
            return (best_ask + best_bid) / 2
        return None

    def update_rolling_autocorr(self):
        if len(self.recent_mid_prices) < self.rolling_window:
            return 0, 0, 0  # autocorr, t-stat, cluster_label = insignificant

        returns = np.diff(self.recent_mid_prices[-self.rolling_window:])
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]

        # Compute t-statistic for autocorrelation
        n = len(returns) - 1
        t_stat = autocorr * np.sqrt(n) / np.sqrt(1 - autocorr**2) if abs(autocorr) < 1 else 0

        # Cluster logic (simple thresholds based on previous clustering analysis)
        if t_stat < -1.96:
            cluster = 1  # negative autocorr (mean-reverting)
        elif t_stat > 1.96:
            cluster = 2  # positive autocorr (momentum)
        else:
            cluster = 0  # insignificant

        self.cluster_labels.append(cluster)
        return autocorr, t_stat, cluster

    def cluster_strategy(self, product: str, order_depth: OrderDepth, position: int) -> List[Order]:
        orders = []
        mid_price = self.compute_mid_price(order_depth)
        if mid_price is None:
            return orders

        self.recent_mid_prices.append(mid_price)
        if len(self.recent_mid_prices) > 1000:
            self.recent_mid_prices = self.recent_mid_prices[-1000:]

        autocorr, t_stat, cluster = self.update_rolling_autocorr()

        best_ask = min(order_depth.sell_orders)
        best_bid = max(order_depth.buy_orders)

        recent_return = 0
        if len(self.recent_mid_prices) >= 2:
            recent_return = self.recent_mid_prices[-1] - self.recent_mid_prices[-2]

        buy_limit = self.position_limits[product] - position
        sell_limit = self.position_limits[product] + position

        qty = min(self.max_order_size, 10)  # Use fixed trade size or adapt later

        if cluster == 1:  # Mean-reverting regime
            if recent_return > 0 and buy_limit > 0:
                # Price went up -> fade -> short
                orders.append(Order(product, best_bid, -qty))
            elif recent_return < 0 and sell_limit > 0:
                # Price went down -> fade -> buy
                orders.append(Order(product, best_ask, qty))

        elif cluster == 2:  # Momentum regime
            if recent_return > 0 and sell_limit > 0:
                # Price going up -> go with trend -> buy
                orders.append(Order(product, best_ask, qty))
            elif recent_return < 0 and buy_limit > 0:
                # Price going down -> short
                orders.append(Order(product, best_bid, -qty))

        # In insignificant regime, optionally do nothing or do neutral MM
        return orders

    def run(self, state: TradingState):
        result = {}

        for product in ["SQUID_INK"]:
            if product in state.order_depths:
                pos = state.position.get(product, 0)
                result[product] = self.cluster_strategy(product, state.order_depths[product], pos)

        traderData = "SAMPLE"
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
