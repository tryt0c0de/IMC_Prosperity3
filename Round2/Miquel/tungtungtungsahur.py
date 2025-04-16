from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import jsonpickle
import numpy as np
import math
# from Logger import Logger
# logger = Logger()

# Define the product names.
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

# Global PARAMS definitions for the three products.
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,       # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 30,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 25,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,       # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 50,
    }
}

class Trader:
    def __init__(self, params=None):
        # Initialize the parameters for the original products.
        self.params = PARAMS
        # Position limits for the original products.
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50
        }
        # Adjust KELP's mean reversion parameter.
        self.params[Product.KELP]["reversion_beta"] = -0.6

        # ----- New basket (trader_max) related initialization -----
        # Basket trading assets for DJEMBES (asset1) and CROISSANTS (asset2)
        self.products = ['CROISSANTS', 'DJEMBES']
        self.mid = {}  # To store mid-prices of basket assets
        self.to_liquidate = {prod: False for prod in self.products}
        self.asset1 = 'DJEMBES'
        self.asset2 = ['CROISSANTS']
        self.threshold = 1
        self.prop = 1 / 2
        self.limits = {'CROISSANTS': 250, 'DJEMBES': 60}
        self.regression = {
                'DJEMBES': {
                'Intercept': 6913.009686452366,
                'CROISSANTS': 1.5177902648379593,
                'std': 23.282224239738}}
        self.multiple = 0
        self.track_pnl_baskets = {"basket1": 0, "basket2": 0}
        for product in self.asset2:
            # Determine maximum multiplier allowed by the limit and regression coefficients.
            mul = int(self.limits[product] // abs(self.regression[self.asset1][product]))
            self.multiple = max(self.multiple, mul)
        # Initialize current holdings for basket assets.
        self.current_holdings = {}

    ### -- Original Methods for Trading RAINFOREST_RESIN, KELP, and SQUID_INK -- ###

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if position_after_take > 0:
            clear_quantity = sum(volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
            filtered_bid = [price for price in order_depth.buy_orders if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None
            if mm_ask is None or mm_bid is None:
                mmmid_price = (best_ask + best_bid) / 2 if traderObject.get("kelp_last_price") is None else traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            if traderObject.get("kelp_last_price") is not None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.params[Product.KELP]["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def fair_value_squid_ink(self, order_depth: OrderDepth):
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        elif order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        elif order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        return None

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, prevent_adverse, adverse_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        orders: List[Order] = []
        asks_above_fair = [price for price in order_depth.sell_orders if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders if price < fair_value - disregard_edge]
        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None
        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            ask = best_ask_above_fair if abs(best_ask_above_fair - fair_value) <= join_edge else best_ask_above_fair - 1
        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            bid = best_bid_below_fair if abs(fair_value - best_bid_below_fair) <= join_edge else best_bid_below_fair + 1
        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1
        buy_order_volume, sell_order_volume = self.market_make(
            product, orders, bid, ask, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def synthetic_real_arb(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        arb_threshold = 50
        arb_threshold2 = 30
        required_products = ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1"]
        for prod in required_products:
            if prod not in state.order_depths:
                return orders
        cd: OrderDepth = state.order_depths["CROISSANTS"]
        jd: OrderDepth = state.order_depths["JAMS"]
        dd: OrderDepth = state.order_depths["DJEMBES"]
        pb_depth: OrderDepth = state.order_depths["PICNIC_BASKET1"]
        pb2_depth: OrderDepth = state.order_depths["PICNIC_BASKET2"]
        if not (cd.buy_orders and cd.sell_orders and jd.buy_orders and jd.sell_orders and dd.buy_orders and dd.sell_orders and pb_depth.buy_orders and pb_depth.sell_orders):
            return orders
        mid_c = (max(cd.buy_orders.keys()) + min(cd.sell_orders.keys())) / 2
        mid_j = (max(jd.buy_orders.keys()) + min(jd.sell_orders.keys())) / 2
        mid_d = (max(dd.buy_orders.keys()) + min(dd.sell_orders.keys())) / 2
        coefficients_regression_basket1 = [8.49318614, 2.66122078, 0.37259003]
        coeffcients_regression_basket2 = [4.1366589, 2.2213497, -0.14996876]
        synthetic_price = (coefficients_regression_basket1[0] * mid_c +
                           coefficients_regression_basket1[1] * mid_j +
                           coefficients_regression_basket1[2] * mid_d)
        synthetic_price2 = (coeffcients_regression_basket2[0] * mid_c +
                            coeffcients_regression_basket2[1] * mid_j +
                            coeffcients_regression_basket2[2] * mid_d)
        mid_basket = (max(pb_depth.buy_orders.keys()) + min(pb_depth.sell_orders.keys())) / 2
        mid_basket2 = (max(pb2_depth.buy_orders.keys()) + min(pb2_depth.sell_orders.keys())) / 2
        spread = mid_basket - synthetic_price
        spread2 = mid_basket2 - synthetic_price2
        threshold_liquidate1 =15 
        threshold_liquidate2 = 10
        position_basket_1 = state.position.get("PICNIC_BASKET1",0)
        position_basket_2 = state.position.get("PICNIC_BASKET2",0)
        if position_basket_1 >0:
            if spread > threshold_liquidate1:
                orders.append(Order("PICNIC_BASKET1", max(pb_depth.buy_orders.keys()), -1))
        elif position_basket_1 <0:
            if spread < -threshold_liquidate1:
                orders.append(Order("PICNIC_BASKET1", min(pb_depth.sell_orders.keys()), 1))
        if position_basket_2 >0:
            if spread2 > threshold_liquidate2:
                orders.append(Order("PICNIC_BASKET2", max(pb2_depth.buy_orders.keys()), -1))
        elif position_basket_2 <0:
            if spread2 < -threshold_liquidate2:
                orders.append(Order("PICNIC_BASKET2", min(pb2_depth.sell_orders.keys()), 1))

        if spread > arb_threshold:
            real_sell_price = max(pb_depth.buy_orders.keys())
            orders.append(Order("PICNIC_BASKET1", real_sell_price, -1))
            
        elif spread < -arb_threshold:
            real_buy_price = min(pb_depth.sell_orders.keys())
            orders.append(Order("PICNIC_BASKET1", real_buy_price, 1))
        if spread2 > arb_threshold2:
            real_sell_price = max(pb2_depth.buy_orders.keys())
            orders.append(Order("PICNIC_BASKET2", real_sell_price, -1))
        elif spread2 < -arb_threshold2:
            real_buy_price = min(pb2_depth.sell_orders.keys())
            orders.append(Order("PICNIC_BASKET2", real_buy_price, 1))
        return orders

    ### -- New Basket Trading Methods (for trader_max strategy) -- ###

    def take_bids(self, best_bid, best_bid_amount, q, asset, liquidate=False):
        orders = []
        n = len(best_bid)
        q0 = min(q, best_bid_amount[0])
        q1, q2 = 0, 0
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
        if q < 0:
            return self.take_asks(best_ask, best_ask_amount, abs(q), asset, liquidate=True)
        elif q > 0:
            return self.take_bids(best_bid, best_bid_amount, abs(q), asset, liquidate=True)
        return []

    def trader_max(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        # Initialize result containers for basket assets.
        result[self.asset1] = []
        for product in self.asset2:
            result[product] = []
        self.current_holdings[self.asset1] = state.position.get(self.asset1, 0)
        for x in self.asset2:
            self.current_holdings[x] = state.position.get(x, 0)
        order_depth_1 = state.order_depths[self.asset1]
        order_depth_2 = {x: state.order_depths[x] for x in self.asset2}

        # Best bid/ask for asset1.
        if order_depth_1.sell_orders:
            best_ask1, best_ask_amount1 = zip(*list(order_depth_1.sell_orders.items()))
            best_ask1, best_ask_amount1 = list(best_ask1), list(best_ask_amount1)
        else:
            best_ask1, best_ask_amount1 = [], []
        if order_depth_1.buy_orders:
            best_bid1, best_bid_amount1 = zip(*list(order_depth_1.buy_orders.items()))
            best_bid1, best_bid_amount1 = list(best_bid1), list(best_bid_amount1)
        else:
            best_bid1, best_bid_amount1 = [], []
        best_ask2, best_ask_amount2 = {}, {}
        best_bid2, best_bid_amount2 = {}, {}
        for x in self.asset2:
            od = order_depth_2[x]
            if od.sell_orders:
                best_ask2[x], best_ask_amount2[x] = zip(*list(od.sell_orders.items()))
            else:
                best_ask2[x], best_ask_amount2[x] = [], []
            if od.buy_orders:
                best_bid2[x], best_bid_amount2[x] = zip(*list(od.buy_orders.items()))
            else:
                best_bid2[x], best_bid_amount2[x] = [], []
            best_ask2[x], best_ask_amount2[x] = list(best_ask2[x]), list(best_ask_amount2[x])
            best_bid2[x], best_bid_amount2[x] = list(best_bid2[x]), list(best_bid_amount2[x])
        if not (list(best_ask_amount1) + list(best_bid_amount1)):
            return {self.asset1: [], self.asset2[0]: []}
        self.mid[self.asset1] = (best_bid1[0] + best_ask1[0]) / 2
        for x in self.asset2:
            self.mid[x] = (best_bid2[x][0] + best_ask2[x][0]) / 2
        multiples = list(range(1, 1 + self.multiple))[::-1]
        curr_long = self.current_holdings[self.asset1] > 0
        curr_short = self.current_holdings[self.asset1] < 0
        spread = self.mid[self.asset1] - self.regression[self.asset1]['Intercept']
        for product in self.asset2:
            spread -= self.regression[self.asset1][product] * self.mid[product]
        spread /= self.regression[self.asset1]['std']
        q = 0
        neutral = False
        if not (curr_short or curr_long):
            self.to_liquidate[self.asset1] = False
            for x in self.asset2:
                self.to_liquidate[x] = False
        if (curr_long and spread > self.threshold * self.prop) or \
           (curr_short and spread < -self.threshold * self.prop) or \
           self.to_liquidate[self.asset1] or any(self.to_liquidate[x] for x in self.asset2):
            neutral = True
        elif spread > self.threshold and not curr_short:
            q = 1
        elif spread < -self.threshold and not curr_long:
            q = -1
        if q > 0:
            for mul in multiples:
                bid = self.take_bids(best_bid1, best_bid_amount1, mul, self.asset1)
                if not bid:
                    continue
                asks = {}
                for product in self.asset2:
                    desired_quant = round(self.regression[self.asset1][product] * mul)
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
        return result

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        result = {}

        # Process original products' strategies if available.
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["take_width"],
                resin_position,
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["clear_width"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
            )
            resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = resin_take_orders + resin_clear_orders + resin_make_orders

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_fair_value = self.kelp_fair_value(state.order_depths[Product.KELP], traderObject)
            kelp_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["take_width"],
                kelp_position,
                self.params[Product.KELP]["prevent_adverse"],
                self.params[Product.KELP]["adverse_volume"],
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["clear_width"],
                kelp_position,
                buy_order_volume,
                sell_order_volume,
            )
            kelp_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = kelp_take_orders + kelp_clear_orders + kelp_make_orders

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            order_depth = state.order_depths[Product.SQUID_INK]
            position = state.position.get(Product.SQUID_INK, 0)
            fair_value = self.fair_value_squid_ink(order_depth)
            if fair_value is not None:
                squid_take_orders, buy_vol, sell_vol = self.take_orders(
                    Product.SQUID_INK,
                    order_depth,
                    fair_value,
                    self.params[Product.SQUID_INK]["take_width"],
                    position,
                )
            squid_clear_orders, buy_vol, sell_vol = self.clear_orders(
                Product.SQUID_INK,
                order_depth,
                fair_value,
                self.params[Product.SQUID_INK]["clear_width"],
                position,
                buy_vol,
                sell_vol,
            )
            squid_make_orders, _, _ = self.make_orders(
                Product.SQUID_INK,
                order_depth,
                fair_value,
                position,
                buy_vol,
                sell_vol,
                self.params[Product.SQUID_INK]["disregard_edge"],
                self.params[Product.SQUID_INK]["join_edge"],
                self.params[Product.SQUID_INK]["default_edge"],
                True,
                self.params[Product.SQUID_INK]["soft_position_limit"],
            )
            result[Product.SQUID_INK] = squid_take_orders + squid_clear_orders + squid_make_orders

        arb_orders = self.synthetic_real_arb(state)
        for order in arb_orders:
            if order.symbol in result:
                result[order.symbol].append(order)
            else:
                result[order.symbol] = [order]

        # ----- Call the trader_max (basket) logic and merge its orders into result -----
        basket_orders = self.trader_max(state)
        for asset, orders_list in basket_orders.items():
            if asset in result:
                result[asset].extend(orders_list)
            else:
                result[asset] = orders_list

        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        # logger.flush(state, result,conversions, traderData)
        return result, conversions, traderData
