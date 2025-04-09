import pandas as pd
import sys
import math
from numpy.ma.core import product

from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Listing, Trade, Observation, ProsperityEncoder
from typing import List
import string

from typing import List
import string
import json
import matplotlib.pyplot as plt
from typing import Any
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()



class Trader:
    def __init__(self, params=None):
        if not params:
            params = [100, 1000, 1]
        self.products = ['KELP', 'SQUID_INK']
        self.products = ['SQUID_INK']
        self.df = pd.DataFrame({col: [] for col in ['timestamp'] + [f'{c}_{prod}' for c in ['std', 'ewm_fast', 'ewm_slow', 'w_price', 'ub', 'spread'] for prod in self.products]})
        self.span_fast = int(params[0])
        self.span_slow = int(params[1])
        self.coef1 = 1+ (0.005 * params[2])
        self.coef2 = 1 + (0.01 * params[2])
        self.coef3 = 1 + (0.025 * params[2])
        self.max_holdings = {prod: 50 for prod in self.products}
        self.current_holdings = {prod: 0 for prod in self.products}
        #self.max_order_size = 10
        self.w_price = {}
        self.ewm_fast = {}
        self.ewm_slow = {}
        self.ub = {}
        self.spread = {}
        self.holding_ratio = {}
        self.std = {}

        self.signal_df = pd.DataFrame({col:[] for col in ['timestamp', 'buy', 'sell', 'neutral']})



    def plot_ewm(self, product):
        plt.plot(self.df[f'w_price_{product}'].iloc[200:])
        plt.plot(self.df[f'ewm_{product}'].iloc[200:], ls='--')
        plt.show()


    def run(self, state: TradingState):

        traderData = "SAMPLE"
        conversions = 1

        timestamp = state.timestamp
        self.signal_df.loc[len(self.signal_df)] = [timestamp, 0, 0, 0]
        result = {}

        for product in self.products:
            self.current_holdings[product] = state.position.get(product, 0)
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Get best bid and ask
            best_ask, best_ask_amount = zip(*list(order_depth.sell_orders.items())) if order_depth.sell_orders else ([], [])
            best_bid, best_bid_amount = zip(*list(order_depth.buy_orders.items())) if order_depth.buy_orders else ([], [])

            best_ask = list(best_ask)
            best_ask_amount = list(best_ask_amount)
            best_bid = list(best_bid)
            best_bid_amount = list(best_bid_amount)

            if not (best_ask_amount + best_bid_amount):
                if not self.df.empty:
                    for prod in self.products:
                        self.std[prod] = self.df[f'std_{prod}'].iloc[-1]
                        self.ewm_fast[prod] = self.df[f'ewm_fast_{prod}'].iloc[-1]
                        self.ewm_slow[prod] = self.df[f'ewm_slow_{prod}'].iloc[-1]
                        self.w_price[prod] = self.df[f'w_price_{prod}'].iloc[-1]
                        self.ub[prod] = self.df[f'ub_{prod}'].iloc[-1]
                        self.spread[prod] = self.df[f'spread_{prod}'].iloc[-1]
                    timestamp = self.df['timestamp'].iloc[-1] + 100
                else:
                    logger.flush(state, result, conversions, traderData)
                    return result, conversions, traderData
                break

            self.w_price[product] = (best_bid[0] + best_ask[0]) / 2


            self.std[product], self.ewm_fast[product], self.ewm_slow[product], self.ub[product], self.spread[product] = 0,0,0,0,0

            if timestamp >= self.span_slow * 100:
                # ADJUST ?????
                def moving(span):
                    return pd.Series(self.df[f'w_price_{product}'].tolist() + [self.w_price[product]]).ewm(span=span, adjust=False)

                self.ewm_fast[product] = moving(self.span_fast).mean().iloc[-1]
                self.ewm_slow[product] = moving(self.span_slow).mean().iloc[-1]

                curr_long = (self.current_holdings[product] > 0)
                curr_short = (self.current_holdings[product] < 0)

                q = 0
                neutral = True
                ratio = self.ewm_slow[product]/self.ewm_fast[product]
                self.std[product] = ratio



                if ratio > self.coef1:
                    if curr_long:
                        q = -self.current_holdings[product]
                    if ratio > self.coef2:
                        neutral = False
                        q -= self.max_holdings[product]
                    if ratio > self.coef3:
                        q = self.max_holdings[product] - self.current_holdings[product]

                elif ratio < 2-self.coef1:
                    if curr_short:
                        q = -self.current_holdings[product]
                    if ratio < 2-self.coef2:
                        neutral = False
                        q -= -self.max_holdings[product]
                    if ratio < 2-self.coef3:
                        q = -self.max_holdings[product] + self.current_holdings[product]



                if q != 0:
                    self.signal_df.loc[len(self.signal_df)-1] = [timestamp, 0,0,1] if neutral else ([timestamp, 1,0,0] if q>0 else [timestamp, 0,1,0])






                '''if q!= 0:
                    with open('/Users/maximesolere/desktop/log.txt', "a") as file:
                        file.write(f"{best_bid}, {q}\n")
                        file.write(f"{best_bid_amount}, {q}\n")
                        file.write(f"{best_ask}, {q}\n")
                        file.write(f"{best_ask_amount}, {q}\n")
                        file.write('\n')'''


                if q < 0:
                    orders.append(Order(product, best_bid[0], max(q, -best_bid_amount[0])))

                elif q > 0:
                    orders.append(Order(product, best_ask[0], min(q, -best_ask_amount[0])))


            result[product] = orders

        # Update dataframe with new weighted prices
        self.df.loc[len(self.df)] = [timestamp] + [col[prod] for col in [self.std, self.ewm_fast, self.ewm_slow, self.w_price, self.ub, self.spread] for prod in self.products]


        if timestamp==990000:
            with open('/Users/maximesolere/desktop/log.txt', "a") as file:
                file.write(f"{self.df['w_price_SQUID_INK'].max()}")
            self.df.to_csv('/Users/maximesolere/desktop/df.csv')
            self.signal_df.to_csv('/Users/maximesolere/desktop/signal_df.csv')
            #return 1



        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
