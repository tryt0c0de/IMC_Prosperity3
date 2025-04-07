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
    def __init__(self):
        self.df_kelp = pd.DataFrame({col: [] for col in ['timestamp', 'w_price_KELP', 'w_price_SQUID_INK']})
        self.span = 30
        self.ub = 1 / 1000
        self.lb = -self.ub
        self.max_holdings = {"SQUID_INK": 50, "KELP": 50}
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

                logger.print(f"{product} spread: {spread}, UB: {ub}, LB: {lb}")

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

        # Update dataframe with new weighted prices
        self.df_kelp.loc[len(self.df_kelp)] = [timestamp] + [self.w_price[product] for product in ['KELP', 'SQUID_INK']]

        traderData = "SAMPLE"
        conversions = 1

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
