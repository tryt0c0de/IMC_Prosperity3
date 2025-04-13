import pandas as pd
import statsmodels.api as sm
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

        self.products = ['KELP', 'SQUID_INK', 'CROISSANTS', 'DJEMBES', 'JAMS', 'PICNIC_BASKET1', 'PICNIC_BASKET2', 'RAINFOREST_RESIN']
        #self.products = ['PICNIC_BASKET1', 'JAMS']
        self.df = pd.DataFrame({col: [] for col in ['timestamp'] + [f'{c}_{prod}' for c in ['mid'] for prod in self.products]})

        self.max_holdings = {prod: 50 for prod in self.products}
        self.current_holdings = {prod: 0 for prod in self.products}

        self.mid = {}

        self.to_liquidate = {prod: False for prod in self.products}

        span_fast = ['DJEMBES', 'JAMS', 'JAMS']
        span_slow = ['CROISSANTS', 'CROISSANTS', 'DJEMBES']

        if not params:
            self.asset1 = 'DJEMBES'
            self.asset2 = ['CROISSANTS', 'JAMS']
            self.threshold = 1.4375
            self.prop = 0

        else:
            #self.asset1 = span_fast[int(params[0])]
            #self.asset2 = [span_slow[int(params[0])]]
            self.asset1 = span_fast[int(params[0])]
            self.asset2 = [span_slow[int(params[0])]]
            self.threshold = params[1]
            self.prop = params[2]


        self.limits = {'CROISSANTS': 250,
                       'DJEMBES': 60,
                       'JAMS': 350,
                       'KELP': 50,
                       'PICNIC_BASKET1': 60,
                       'PICNIC_BASKET2': 100,
                       'RAINFOREST_RESIN': 50,
                       'SQUID_INK': 50}

        self.regression = {
            'PICNIC_BASKET1': {
                'Intercept': 9247.413503566233,
                'CROISSANTS': 8.425433348419688,
                'DJEMBES': -0.5108856493110068,
                'JAMS': 3.1028057117157886,
                'std': 77.37545333537989
        }}

        self.regression = {
            'PICNIC_BASKET1': {
                'Intercept': 9247.413503566233,
                'CROISSANTS': 8.425433348419688,
                'DJEMBES': -0.5108856493110068,
                'JAMS': 3.1028057117157886,
                'std': 77.37545333537989
            }}

        self.regression = {
            'PICNIC_BASKET1': {
                'Intercept': 13249.25698105729,
                'PICNIC_BASKET2': 0.9970625254932948,
                'JAMS': 2.3484119937592993,
                'std': 98.05545015362041
            }}

        self.regression = {
                'DJEMBES': {
                'Intercept': 7659.088066393866,
                'CROISSANTS': 0.9543371164975056,
                'JAMS': 0.2541291462546411,
                'std': 22.194071205844857
            }}




        pairs = pd.read_csv('/Users/maximesolere/Desktop/pairs.csv')
        self.regression = {}
        for i,row in pairs.iterrows():
            if row['x'] == self.asset2[0]:
                self.regression[row['y']] = {
                    row['x']: row['beta'],
                    'Intercept': row['intercept'],
                    'std': row['std']
                }

        self.multiple = 0
        for product in self.asset2:
            mul = int(self.limits[product]//abs(self.regression[self.asset1][product]))
            self.multiple = max(self.multiple, mul)



    def take_bids(self, best_bid, best_bid_amount, q, asset, liquidate=False):
        orders = []
        n = len(best_bid)
        q0 = min(q, best_bid_amount[0])
        q1,q2 = 0,0

        if q0 < q and n > 1:
            q1 = min(q - q0, best_bid_amount[1])

            if q0 + q1 < q and n > 2:
                q2 = min(q - q0 - q1, best_bid_amount[2])

        if liquidate or q0 + q1 + q2 == q:
            orders.append(Order(asset, best_bid[0], -q0))
            if q1:
                orders.append(Order(asset, best_bid[1], -q1))
            if q2:
                orders.append(Order(asset, best_bid[2], -q2))

        return orders

    def take_asks(self, best_ask, best_ask_amount, q, asset, liquidate=False):
        orders = []
        n = len(best_ask)
        q0 = max(-q, best_ask_amount[0])
        q1, q2 = 0, 0

        if q0 > -q and n > 1:
            q1 = max(-q - q0, best_ask_amount[1])

            if q0 + q1 > -q and n > 2:
                q2 = max(-q - q0 - q1, best_ask_amount[2])

        if liquidate or q0 + q1 + q2 == -q:
            orders.append(Order(asset, best_ask[0], -q0))
            if q1:
                orders.append(Order(asset, best_ask[1], -q1))
            if q2:
                orders.append(Order(asset, best_ask[2], -q2))

        return orders

    def liquidate(self, best_bid, best_bid_amount, best_ask, best_ask_amount, asset):
        q = self.current_holdings[asset]
        if q<0:
            return self.take_asks(best_ask, best_ask_amount, abs(q), asset, liquidate=True)
        elif q>0:
            return self.take_bids(best_bid, best_bid_amount, abs(q), asset, liquidate=True)
        return []



    def run(self, state: TradingState):

        traderData = "SAMPLE"
        conversions = 1


        timestamp = state.timestamp
        result = {}
        result[self.asset1] = []
        for product in self.asset2:
            result[product] = []




        if True: #for product in self.products:


            self.current_holdings[self.asset1] = state.position.get(self.asset1, 0)
            for x in self.asset2:
                self.current_holdings[x] = state.position.get(x, 0)


            order_depth_1: OrderDepth = state.order_depths[self.asset1]
            order_depth_2= {x: state.order_depths[x] for x in self.asset2}

            orders1: List[Order] = []
            orders2: List[Order] = []

            # Get best bid and ask
            best_ask1, best_ask_amount1 = zip(*list(order_depth_1.sell_orders.items())) if order_depth_1.sell_orders else (
            [], [])
            best_bid1, best_bid_amount1 = zip(*list(order_depth_1.buy_orders.items())) if order_depth_1.buy_orders else (
            [], [])

            best_ask2, best_ask_amount2 = {},{}
            best_bid2, best_bid_amount2 = {},{}
            for x in self.asset2:
                best_ask2[x], best_ask_amount2[x] = zip(
                    *list(order_depth_2[x].sell_orders.items())) if order_depth_2[x].sell_orders else (
                    [], [])
                best_bid2[x], best_bid_amount2[x] = zip(
                    *list(order_depth_2[x].buy_orders.items())) if order_depth_2[x].buy_orders else (
                    [], [])

                best_ask2[x], best_ask_amount2[x] = list(best_ask2[x]), list(best_ask_amount2[x])
                best_bid2[x], best_bid_amount2[x] = list(best_bid2[x]), list(best_bid_amount2[x])


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


            self.mid[self.asset1] = (best_bid1[0] + best_ask1[0]) / 2
            for x in self.asset2:
                self.mid[x] = (best_bid2[x][0] + best_ask2[x][0]) / 2



            multiples = list(range(1, 1+self.multiple))[::-1]


            if True:


                curr_long = (self.current_holdings[self.asset1] > 0)
                curr_short = (self.current_holdings[self.asset1] < 0)





                #prices1 = pd.Series(self.df[self.asset1].tolist() + [self.mid[self.asset1]])
                #prices2 = pd.Series(self.df[self.asset2].tolist() + [self.mid[self.asset2]])



                spread = self.mid[self.asset1] - self.regression[self.asset1]['Intercept']
                for product in self.asset2:
                    spread -= self.regression[self.asset1][product] * self.mid[product]
                spread /= self.regression[self.asset1]['std']

                with open('/Users/maximesolere/desktop/log.txt', "a") as file:
                    file.write(f"{timestamp} {spread}\n")

                q = 0
                neutral = False

                if not (curr_short or curr_long):
                    self.to_liquidate[self.asset1] = False
                    for x in self.asset2:
                        self.to_liquidate[x] = False

                if (curr_long and spread > self.threshold*self.prop) or (curr_short and spread < -self.threshold*self.prop) or self.to_liquidate[self.asset1] or True in [self.to_liquidate[x] for x in self.asset2]:
                    neutral = True

                elif spread > self.threshold and not curr_short:
                    q = 1
                elif spread < -self.threshold and not curr_long:
                    q = -1



                if q > 0 :
                    for mul in multiples:
                        bid = self.take_bids(best_bid1, best_bid_amount1, mul, self.asset1)
                        if not bid:
                            continue
                        asks = {}
                        for product in self.asset2:
                            desired_quant = round(self.regression[self.asset1][product]*mul)
                            ask = self.take_asks(best_ask2[product], best_ask_amount2[product], desired_quant, product)
                            if not (ask and desired_quant):
                                asks = {}
                                break
                            asks[product] = ask
                        if not asks:
                            continue

                        result[self.asset1] = bid
                        for product in self.asset2:
                            result[product] = asks[product]
                        break

                if q < 0:
                    for mul in multiples:
                        ask = self.take_asks(best_ask1, best_ask_amount1, mul, self.asset1)
                        if not ask:
                            continue
                        bids = {}
                        for product in self.asset2:
                            desired_quant = round(self.regression[self.asset1][product] * mul)
                            bid = self.take_asks(best_bid2[product], best_bid_amount2[product], desired_quant, product)
                            if not (bid and desired_quant):
                                bids = {}
                                break
                            bids[product] = bid
                        if not bids:
                            continue

                        result[self.asset1] = ask
                        for product in self.asset2:
                            result[product] = bids[product]
                        break


                if neutral:
                    self.to_liquidate[self.asset1] = True
                    for x in self.asset2:
                        self.to_liquidate[x] = True

                    result[self.asset1] = self.liquidate(best_bid1, best_bid_amount1, best_ask1, best_ask_amount1, self.asset1)
                    for product in self.asset2:
                        result[product] = self.liquidate(best_bid2[product], best_bid_amount2[product], best_ask2[product], best_ask_amount2[product], product)


        # Update dataframe with new weighted prices
        #self.df.loc[len(self.df)] = [timestamp] + [col[prod] for col in [self.mid] for prod in self.products]


        if timestamp==999900:
            self.df.to_csv('/Users/maximesolere/desktop/df.csv')
            #with open('/Users/maximesolere/desktop/log.txt', "a") as file:
                #file.write(f"{self.df['mid_KELP'].max()}")
            #return 1

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
