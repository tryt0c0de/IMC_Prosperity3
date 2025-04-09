import pandas as pd
import sys

from numpy.ma.core import product

from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Listing, Trade, Observation, ProsperityEncoder
from typing import List
import string

from typing import List
import string
import json
from typing import Any
from Logger import Logger

logger = Logger()



class Trader:
    def __init__(self, ub):
        self.df_kelp = pd.DataFrame({col: [] for col in ['timestamp', 'w_price_KELP', 'w_price_SQUID_INK']})
        self.span = 30
        self.ub = 1/(ub[0])
        self.lb = -self.ub
        self.max_holdings = {"SQUID_INK": 50, "KELP": 50, "RAINFOREST_RESIN": 50}
        self.current_holdings = {product: 0 for product in self.max_holdings}
        self.max_order_size = 10
        self.w_price = {}

    def run(self, state: TradingState):
        timestamp = state.timestamp
        result = {}

        for product in ['KELP', 'SQUID_INK']:
            self.current_holdings[product] = state.position.get(product, 0)
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Get best bid and ask
            best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0] if order_depth.sell_orders else (0, 0)
            best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0] if order_depth.buy_orders else (0, 0)

            if best_ask_amount + best_bid_amount == 0:
                self.w_price[product] = self.df_kelp[f'w_price_{product}'].iloc[-1] if not self.df_kelp.empty else None
                continue

            self.w_price[product] = (best_bid + best_ask) / 2

            if timestamp >= self.span * 1000:
                recent_prices = self.df_kelp[f'w_price_{product}'].iloc[-self.span + 1:].tolist() + [self.w_price[product]]
                ewm = pd.Series(recent_prices).ewm(span=self.span, adjust=False).mean().iloc[-1]
                spread = (self.w_price[product] - ewm) / ewm

                holding_ratio = self.current_holdings[product] / self.max_holdings[product]
                ub = self.ub - holding_ratio * self.ub
                lb = self.lb - holding_ratio * self.lb

                # logger.print(f"{product} spread: {spread}, UB: {ub}, LB: {lb}")

                if spread < lb:
                    buy_limit = self.max_holdings[product] - self.current_holdings[product]
                    desired_qty = max(1, int((1 - holding_ratio) * self.max_order_size))
                    q = min(self.max_order_size + min(0, self.current_holdings[product]),
                            desired_qty + min(0, self.current_holdings[product]),
                            buy_limit)
                    self.current_holdings[product] += q
                    orders.append(Order(product, best_ask, q))

                elif spread > ub:
                    sell_limit = self.max_holdings[product] + self.current_holdings[product]
                    desired_qty = max(1, int((1 + holding_ratio) * self.max_order_size))
                    q = min(self.max_order_size + max(0, self.current_holdings[product]),
                            desired_qty + max(0, self.current_holdings[product]),
                            sell_limit)
                    self.current_holdings[product] -= q
                    orders.append(Order(product, best_bid, -q))

            result[product] = orders
        for product in ["RAINFOREST_RESIN"]:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            if product == "RAINFOREST_RESIN":
                acceptable_price = 10000
                total_add_pos = 0
            # elif product == "KELP":
            #     acceptable_price = 2015

                if len(order_depth.sell_orders) != 0:
                    for i in range (0, len(list(order_depth.sell_orders.items()))):
                        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[i]
                        if int(best_ask) < acceptable_price:
                            #print("BUY", str(-best_ask_amount) + "x", best_ask)
                            if self.current_holdings[product] < self.max_holdings[product]:
                                buy_amount = min(self.max_holdings[product] - self.current_holdings[product], -best_ask_amount)
                                orders.append(Order(product, best_ask, buy_amount))
                                self.current_holdings[product] += buy_amount
                            #orders.append(Order(product, best_ask, -best_ask_amount))

                if len(order_depth.buy_orders) != 0:
                    for j in range (0, len(list(order_depth.buy_orders.items()))):
                        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[j]
                        if int(best_bid) > acceptable_price:
                            #print("SELL", str(best_bid_amount) + "x", best_bid)
                            if self.current_holdings[product] > -1 * self.max_holdings[product]:
                                sell_amount = min(self.current_holdings[product] + self.max_holdings[product], best_bid_amount)
                                orders.append(Order(product, best_bid, -sell_amount))
                                self.current_holdings[product] -= sell_amount
                
                # if we have huge position in RAINFOREST_RESIN, we need to try to sell it off by acceptable_price
                if self.current_holdings[product]/self.max_holdings[product] > 0.9:
                    orders.append(Order(product, acceptable_price, -5))

                elif self.current_holdings[product]/self.max_holdings[product] < -0.9:
                    orders.append(Order(product, acceptable_price, 5))

                # for i in range (max_holdings[product]):
                #     orders.append(Order(product, acceptable_price + 1, -1))
                #     orders.append(Order(product, acceptable_price - 1, 1))
                result[product] = orders

        # Update dataframe with new weighted prices
        self.df_kelp.loc[len(self.df_kelp)] = [timestamp] + [self.w_price[product] for product in ['KELP', 'SQUID_INK']]

        traderData = "SAMPLE"
        conversions = 1

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
