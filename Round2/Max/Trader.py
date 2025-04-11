import pandas as pd
import statsmodels.api as sm
import sys
import math
from numpy.ma.core import product

from Round2.Max.max_tester_max import multiple
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

        self.products = ['KELP', 'SQUID_INK', 'CROISSANTS', 'DJEMBES', 'JAMS', 'PICNIC_BASKET1', 'PICNIC_BASKET2', 'RAINFOREST_RESIN']
        self.products = ['PICNIC_BASKET1', 'JAMS']
        self.df = pd.DataFrame({col: [] for col in ['timestamp'] + [f'{c}_{prod}' for c in ['mid'] for prod in self.products]})

        self.max_holdings = {prod: 50 for prod in self.products}
        self.current_holdings = {prod: 0 for prod in self.products}

        self.mid = {}

        self.signal_df = pd.DataFrame({col:[] for col in ['timestamp', 'sig']})



    def run(self, state: TradingState):

        traderData = "SAMPLE"
        conversions = 1

        timestamp = state.timestamp
        self.signal_df.loc[len(self.signal_df)] = [timestamp, 0]
        result = {}


        if True: #for product in self.products:
            asset1 = 'PICNIC_BASKET1'
            asset2 = 'JAMS'

            self.current_holdings[asset1] = state.position.get(asset1, 0)
            self.current_holdings[asset2] = state.position.get(asset2, 0)


            order_depth_1: OrderDepth = state.order_depths[asset1]
            order_depth_2: OrderDepth = state.order_depths[asset2]

            orders1: List[Order] = []
            orders2: List[Order] = []

            # Get best bid and ask
            best_ask1, best_ask_amount1 = zip(*list(order_depth_1.sell_orders.items())) if order_depth_1.sell_orders else (
            [], [])
            best_bid1, best_bid_amount1 = zip(*list(order_depth_1.buy_orders.items())) if order_depth_1.buy_orders else (
            [], [])

            best_ask2, best_ask_amount2 = zip(
                *list(order_depth_2.sell_orders.items())) if order_depth_2.sell_orders else (
                [], [])
            best_bid2, best_bid_amount2 = zip(
                *list(order_depth_2.buy_orders.items())) if order_depth_2.buy_orders else (
                [], [])

            best_ask1 = list(best_ask1)
            best_ask_amount1 = list(best_ask_amount1)
            best_bid1 = list(best_bid1)
            best_bid_amount1 = list(best_bid_amount1)

            best_ask2= list(best_ask2)
            best_ask_amount2 = list(best_ask_amount2)
            best_bid2 = list(best_bid2)
            best_bid_amount2 = list(best_bid_amount2)



            if not (best_ask_amount1 + best_bid_amount1):
                if not self.df.empty:
                    for prod in self.products:
                        self.mid[prod] = self.df[f'mid_{prod}'].iloc[-1]
                    timestamp = self.df['timestamp'].iloc[-1] + 100
                else:
                    logger.flush(state, result, conversions, traderData)
                    with open('/Users/maximesolere/desktop/log.txt', "a") as file:
                        file.write(f"{timestamp}\n")
                    return result, conversions, traderData


            self.mid[asset1] = (best_bid1[0] + best_ask1[0]) / 2
            self.mid[asset2] = (best_bid2[0] + best_ask2[0]) / 2


            b1 = 5.156039
            ratio = 5

            mean = 25087.379096616663
            std = 89.71798351096007

            multiples = list(range(1, 50//5+1))[::-1]


            if True: #timestamp >= (look + self.span_slow) * 100:


                curr_long = (self.current_holdings[asset1] > 0)
                curr_short = (self.current_holdings[asset1] < 0)





                #prices1 = pd.Series(self.df[asset1].tolist() + [self.mid[asset1]])
                #prices2 = pd.Series(self.df[asset2].tolist() + [self.mid[asset2]])



                spread = (self.mid[asset1] - b1 * self.mid[asset2] - mean)/std
                '''with open('/Users/maximesolere/desktop/log.txt', "a") as file:
                    file.write(f"{spread}\n")'''

                q = 0
                neutral = False
                threshold = 1

                if (curr_long and spread > 0) or (curr_short and spread < 0):
                    neutral = True

                elif spread > threshold and not curr_short:
                    q = -1
                elif spread < -threshold and not curr_long:
                    q = 1





                q1,q2 = 0,0

                if q > 0 :
                    n = len(best_bid2)

                    for mul in multiples:
                        desired_quant = ratio*mul
                        q0 = min(desired_quant, best_bid_amount2[0])

                        if q0 < desired_quant and n>1:
                            q1 = min(desired_quant-q0, best_bid_amount2[1])

                            if q0+q1 < ratio and n>2:
                                q2 = min(desired_quant-q0-q1, best_bid_amount2[2])

                        if q0+q1+q2 == desired_quant:
                            orders1.append(Order(asset1, best_ask1[0], mul))
                            orders2.append(Order(asset2, best_bid2[0], -q0))
                            if q1:
                                orders2.append(Order(asset2, best_bid2[1], -q1))
                            if q2:
                                orders2.append(Order(asset2, best_bid2[2], -q2))
                            break

                if q < 0:  # and abs(best_bid_amount2[0]) >= 24:
                    n = len(best_ask2)

                    for mul in multiples:
                        desired_quant = ratio*mul
                        q0 = max(-desired_quant, best_ask_amount2[0])

                        if q0 > -desired_quant and n>1:
                            q1 = max(-desired_quant - q0, best_ask_amount2[1])

                            if q0 + q1 > -desired_quant and n>2:
                                q2 = max(-desired_quant - q0 - q1, best_ask_amount2[2])

                        if q0+q1+q2 == -desired_quant:
                            orders1.append(Order(asset1, best_bid1[0], -desired_quant))
                            orders2.append(Order(asset2, best_ask2[0], -q0))
                            if q1:
                                orders2.append(Order(asset2, best_ask2[1], -q1))
                            if q2:
                                orders2.append(Order(asset2, best_ask2[2], -q2))
                            break

                if neutral:
                    if curr_long:
                        orders1.append(Order(asset1, best_bid1[0], -self.current_holdings[asset1]))
                        orders2.append(Order(asset2, best_ask2[0], -self.current_holdings[asset2]))
                    if curr_short:
                        orders1.append(Order(asset1, best_ask1[0], -self.current_holdings[asset1]))
                        orders2.append(Order(asset2, best_bid2[0], -self.current_holdings[asset2]))


            result[asset2] = orders2
            result[asset1] = orders1

        # Update dataframe with new weighted prices
        self.df.loc[len(self.df)] = [timestamp] + [col[prod] for col in [self.mid] for prod in self.products]


        if timestamp==999900:
            self.df.to_csv('/Users/maximesolere/desktop/df.csv')
            #with open('/Users/maximesolere/desktop/log.txt', "a") as file:
                #file.write(f"{self.df['mid_KELP'].max()}")
            self.signal_df.to_csv('/Users/maximesolere/desktop/signal_df.csv')
            #return 1



        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
