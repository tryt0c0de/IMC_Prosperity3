from datamodel import OrderDepth, TradingState, Order
from typing import List
from collections import deque
import numpy as np
from Logger import Logger

logger = Logger()

class Trader:
    
    def __init__(self,parameters = []):
        # -- 1) Embed the learned coefficients properly (5 features, 5 coefficients)
        # Make sure these match exactly what you got from your training
        self.coefs = [
            3.75157646e-06,
            -3.15786104e-05,
            3.44843719e-05,
            1.33571738e-03,
            -3.54871450e-04
        ]
        self.intercept = -0.0007536618536580788
        
        # -- 2) Track the exact feature order
        self.feature_names = [
            'spread_1',
            'total_bid_vol',
            'total_ask_vol',
            'volume_imbalance',
            'momentum'
        ]
        
        # -- 3) To replicate momentum = mid_price - rolling_mean(mid_price, 5)
        # we need to store recent mid_prices in a buffer
        self.midprice_buffer = deque(maxlen=30)

    def run(self, state: TradingState):
        """
        Called at each 'snapshot' of market data.
        Returns: (dict_of_orders, conversions, traderData)
        """
        result = {}

        for product, order_depth in state.order_depths.items():
            if product == "KELP":  #Only run for KELP
                
                # Only run if we have both buy and sell orders
                """if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
                    # No meaningful quote => skip
                    result[product] = []
                    continue"""

                # -- (A) Compute best bid and ask
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                
                # -- (B) Compute the 5 features in the same way as training
                
                # 1) spread_1
                spread_1 = best_ask - best_bid

                # 2) total_bid_vol (levels 1..3 in your training, but here we only see "all buy_orders"?)
                total_bid_vol = sum(abs(v) for v in order_depth.buy_orders.values())
                
                # 3) total_ask_vol
                total_ask_vol = sum(abs(v) for v in order_depth.sell_orders.values())
                
                # 4) volume_imbalance
                volume_imbalance = total_bid_vol / (total_bid_vol + total_ask_vol + 1e-9)
                
                # 5) momentum = mid_price - rolling_mean(mid_price, window=5)
                mid_price = (best_ask + best_bid) / 2
                self.midprice_buffer.append(mid_price)
                
                if len(self.midprice_buffer) < 30:
                    # Not enough history, fallback: momentum = 0 or skip
                    break
                else:
                    rolling_mean_mid = np.mean(self.midprice_buffer)
                    momentum = mid_price - rolling_mean_mid
                
                # -- (C) Put features in a list matching self.coefs
                features = [
                    spread_1,
                    total_bid_vol,
                    total_ask_vol,
                    volume_imbalance,
                    momentum
                ]
                
                # -- (D) Predict future return
                predicted_return = self.intercept + sum(w * x for w, x in zip(self.coefs, features))
                
                # -- (E) Decide trades using threshold
                orders: List[Order] = []
                buy_threshold = 0.0008
                sell_threshold = -0.0008

                if predicted_return > buy_threshold:
                    # Bullish => place a buy. Example: buy 10 at best_ask
                    orders.append(Order(product, best_ask, +10))
                
                elif predicted_return < sell_threshold:
                    # Bearish => place a sell. Example: sell 10 at best_bid
                    orders.append(Order(product, best_bid, -10))
                
                result[product] = orders
            
        conversions = 1
        traderData = "Any info you want to persist" 
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
