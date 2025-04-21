from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math
from typing import Dict 
#from Logger import Logger
from collections import deque
import math
#logger = Logger()

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 5, # Essentially we will never take except very weird stuff happens (it never deviates more than 5)
        "clear_width": 0,
        # for making
        "disregard_edge": 0,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 0,  # joins orders within this edge
        "default_edge": 1000000,
        "soft_position_limit": 50,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 25,
        "reversion_beta": -0.5974,
        "disregard_edge": 0,
        "join_edge": 0,
        "default_edge": 2,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 50,
    }
}


class Trader:
    def __init__(self, params=None):
        #if params is [1.0]:
        # self.params = PARAMS
        # #self.params = params

        # self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50,Product.SQUID_INK: 50}
        # self.products = ['CROISSANTS', 'DJEMBES']
        # self.mid = {}  # To store mid-prices of basket assets
        # self.to_liquidate = {prod: False for prod in self.products}
        # self.asset1 = 'DJEMBES'
        # self.asset2 = ['CROISSANTS']
        # self.threshold = 1
        # self.prop = 1 / 2
        # self.limits = {'CROISSANTS': 250, 'DJEMBES': 60}
        # self.regression = {
        #         'DJEMBES': {
        #         'Intercept': 6913.009686452366,
        #         'CROISSANTS': 1.5177902648379593,
        #         'std': 23.282224239738}}
        # self.multiple = 0
        # self.track_pnl_baskets = {"basket1": 0, "basket2": 0}
        # for product in self.asset2:
        #     # Determine maximum multiplier allowed by the limit and regression coefficients.
        #     mul = int(self.limits[product] // abs(self.regression[self.asset1][product]))
        #     self.multiple = max(self.multiple, mul)


        # VOLC options
        # Initialize current holdings for basket assets.
        self.current_holdings = {}
        self.max_position = 1000
        self.base_spread = 1
        self.skew_param = 0.2
        self.base_size = 1
        self.spread_window = 500
        self.entry_threshold = 0.5
        self.exit_threshold = 0.1
        self.position_size = 2
        self.spread_buffer = 1.0  # Execution cost buffer
        self.p_history: dict[deque] = {}
        self.tick = 3000000
        self.tick_vol = {'VOLCANIC_ROCK':0, 'VOLCANIC_ROCK_VOUCHER_9500':0, 'VOLCANIC_ROCK_VOUCHER_9750':0, 'VOLCANIC_ROCK_VOUCHER_10000':0}
        self.limits_volc = {'VOLCANIC_ROCK':400, 'VOLCANIC_ROCK_VOUCHER_9500':200, 'VOLCANIC_ROCK_VOUCHER_9750':200, 'VOLCANIC_ROCK_VOUCHER_10000':200, 'VOLCANIC_ROCK_VOUCHER_10250':200, 'VOLCANIC_ROCK_VOUCHER_10500':200}
    
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

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
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
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
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
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume
    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
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
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
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
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
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
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1# penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume
    def make_orders_kelp(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        adverse_volume: int = 25,
    ):
        """
        Specialized market making function for KELP that considers:
        1. The trending nature of the product
        2. Presence of informed market makers (identified by order volume)
        3. Multiple price levels to determine optimal order placement
        """
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.KELP]
        
        # 1. Analyze order book structure and identify market maker orders
        mm_asks = [price for price in order_depth.sell_orders.keys() 
                if abs(order_depth.sell_orders[price]) >= adverse_volume]
        mm_bids = [price for price in order_depth.buy_orders.keys() 
                if order_depth.buy_orders[price] >= adverse_volume]
        
        # Sort price levels
        asks = sorted(order_depth.sell_orders.keys())
        bids = sorted(order_depth.buy_orders.keys(), reverse=True)
        
        # 2. Extract multiple levels of the order book
        best_ask = asks[0] if asks else None
        second_ask = asks[1] if len(asks) > 1 else None
        third_ask = asks[2] if len(asks) > 2 else None
        
        best_bid = bids[0] if bids else None
        second_bid = bids[1] if len(bids) > 1 else None
        third_bid = bids[2] if len(bids) > 2 else None
        
        market_spread = (best_ask - best_bid) if (best_ask and best_bid) else default_edge * 2

        bid = fair_value - default_edge
        ask = fair_value + default_edge
        
        # Adjust based on order book and market maker presence
        if mm_asks and mm_bids:
            # There are market makers on both sides - use their prices for guidance
            mm_spread = min(mm_asks) - max(mm_bids)
            
            # If market makers have a tight spread, be cautious
            if mm_spread <= 3:
                # Place orders just inside market maker prices when reasonable
                potential_ask = min(mm_asks) - 1
                potential_bid = max(mm_bids) + 1
                
                # Only use these prices if they don't cross and give us a reasonable spread
                if potential_ask > potential_bid and potential_ask - potential_bid >= 2:
                    ask = potential_ask
                    bid = potential_bid
                else:
                    # Fall back to placing around fair value with a minimum spread
                    ask = max(fair_value + 1, min(mm_asks) - 1)
                    bid = min(fair_value - 1, max(mm_bids) + 1)
        elif best_ask and best_bid:
            # No clear market maker signal, use visible order book
            
            # Consider regular market making approach when spread is wide enough
            if market_spread > 3:
                # Check for unusual volume as potential signal
                ask_volumes = [abs(order_depth.sell_orders[price]) for price in asks[:3] if price in asks]
                bid_volumes = [order_depth.buy_orders[price] for price in bids[:3] if price in bids]
                
                # If second or third level has significantly more volume, consider it
                if len(ask_volumes) > 1 and ask_volumes[1] > ask_volumes[0] * 2:
                    # Second ask level has much more volume - suggests resistance
                    ask = min(asks[1] - 1, best_ask + 1)
                else:
                    # Standard approach - penny the best ask
                    ask = best_ask - 1
                    
                if len(bid_volumes) > 1 and bid_volumes[1] > bid_volumes[0] * 2:
                    # Second bid level has much more volume - suggests support
                    bid = max(bids[1] + 1, best_bid - 1)
                else:
                    # Standard approach - penny the best bid
                    bid = best_bid + 1
            else:
                # Tight spread suggests uncertainty - place at fair value with minimum spread
                ask = max(fair_value + 1, best_ask)
                bid = min(fair_value - 1, best_bid)
                
        # 5. Adjust prices based on our current position
        # If we're long, be more aggressive selling; if short, be more aggressive buying
        position_adjustment = min(5, abs(position) // 10)  # Cap the adjustment
        
        if position > 10:
            ask = max(fair_value + 1, ask - position_adjustment)
        elif position < -10:
            bid = min(fair_value - 1, bid + position_adjustment)
        
        # 6. Final safety check - ensure our bid/ask don't cross
        if bid >= ask:
            mid_price = (bid + ask) / 2
            bid = mid_price - 1
            ask = mid_price + 1
        
        # 7. Generate the actual orders
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.KELP, round(bid), buy_quantity))
            
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.KELP, round(ask), -sell_quantity))
            
        return orders, buy_order_volume, sell_order_volume
    def kelp_take_strategy(
        self,
        order_depth: OrderDepth,
        position: int,
        traderObject: dict,
        buy_order_volume: int = 0,  # Track previously placed buy orders
        sell_order_volume: int = 0  # Track previously placed sell orders
    ) -> (List[Order], int, int):
        """
        A dedicated take strategy for KELP based on its strong mean reversion tendency.
        Tracks buy and sell order volumes to ensure position limits are respected.
        """
        orders = []
        position_limit = self.LIMIT[Product.KELP]
        reversion_beta = self.params[Product.KELP]["reversion_beta"]  # Around -0.6
        
        # Track remaining capacity based on existing orders
        remaining_buy_capacity = position_limit - position - buy_order_volume
        remaining_sell_capacity = position_limit + position - sell_order_volume
        
        # If no capacity left or insufficient order book data, return early
        if (remaining_buy_capacity <= 0 and remaining_sell_capacity <= 0) or not (order_depth.buy_orders and order_depth.sell_orders):
            return orders, buy_order_volume, sell_order_volume
            
        # 1. Calculate current mid price
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        current_mid = (best_bid + best_ask) / 2
        
        # 2. Track price history
        if "kelp_price_history" not in traderObject:
            traderObject["kelp_price_history"] = []
        
        traderObject["kelp_price_history"].append(current_mid)
        
        # Keep only the most recent 100 prices
        if len(traderObject["kelp_price_history"]) > 100:
            traderObject["kelp_price_history"] = traderObject["kelp_price_history"][-100:]
        
        # 3. Calculate recent price movement
        if len(traderObject["kelp_price_history"]) >= 5:
            # Calculate short-term movement (last 3 periods)
            recent_prices = traderObject["kelp_price_history"][-5:]
            short_term_change = recent_prices[-1] - recent_prices[-3]
            
            # Calculate medium-term average as reference point
            medium_term_average = sum(traderObject["kelp_price_history"][-20:]) / min(20, len(traderObject["kelp_price_history"]))
            
            # 4. Predict expected reversion
            expected_reversion = short_term_change * reversion_beta
            
            # 5. Set dynamic thresholds based on recent volatility
            if len(traderObject["kelp_price_history"]) >= 20:
                recent_volatility = np.std(traderObject["kelp_price_history"][-20:])
                # Set thresholds proportional to volatility
                significant_move_threshold = max(1.5, recent_volatility * 1.5)
            else:
                significant_move_threshold = 3  # Default if not enough history
            
            # 6. Take action based on predicted reversion
            # If we predict a downward reversion (after upward movement)
            if expected_reversion < -significant_move_threshold and remaining_sell_capacity > 0:
                # Price is expected to fall, so we want to sell (take existing bids)
                filtered_bids = [
                    (price, volume) for price, volume in order_depth.buy_orders.items()
                    if price > medium_term_average  # Only take bids that are "overpriced"
                ]
                
                if filtered_bids:
                    # Sort by price (highest first) to take the best bids
                    filtered_bids.sort(key=lambda x: x[0], reverse=True)
                    
                    for price, volume in filtered_bids:
                        # Calculate the maximum we can sell based on remaining capacity
                        max_sell = min(volume, remaining_sell_capacity)
                        
                        if max_sell > 0:
                            orders.append(Order(Product.KELP, price, -max_sell))
                            sell_order_volume += max_sell  # Track how much we're selling
                            remaining_sell_capacity -= max_sell  # Update remaining capacity
                            
                            # Stop if we've reached capacity
                            if remaining_sell_capacity <= 0:
                                break
            
            # If we predict an upward reversion (after downward movement)
            elif expected_reversion > significant_move_threshold and remaining_buy_capacity > 0:
                # Price is expected to rise, so we want to buy (take existing asks)
                filtered_asks = [
                    (price, -volume) for price, volume in order_depth.sell_orders.items()
                    if price < medium_term_average  # Only take asks that are "underpriced"
                ]
                
                if filtered_asks:
                    # Sort by price (lowest first) to take the best asks
                    filtered_asks.sort(key=lambda x: x[0])
                    
                    for price, volume in filtered_asks:
                        # Calculate the maximum we can buy based on remaining capacity
                        max_buy = min(volume, remaining_buy_capacity)
                        
                        if max_buy > 0:
                            orders.append(Order(Product.KELP, price, max_buy))
                            buy_order_volume += max_buy  # Track how much we're buying
                            remaining_buy_capacity -= max_buy  # Update remaining capacity
                            
                            # Stop if we've reached capacity
                            if remaining_buy_capacity <= 0:
                                break
        
        return orders, buy_order_volume, sell_order_volume
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
    def options_trader(self, state: TradingState,p2,p1 ="VOLCANIC_ROCK_VOUCHER_10500" ) -> List[Order]:
        #self.tick += 100
        result = {p1: [], p2: []}

        if p1 not in state.order_depths or p2 not in state.order_depths:
            return result, 0, ""

        p1_depth = state.order_depths[p1]
        p2_depth = state.order_depths[p2]

        mid_p1= self.get_mid_price(p1_depth)
        mid_p2= self.get_mid_price(p2_depth)

        if mid_p1 is None or mid_p2 is None:
            return result

        self.p_history.setdefault(p1, deque(maxlen =self.spread_window))
        self.p_history.setdefault(p2, deque(maxlen = self.spread_window))

        # Append the current mid prices to their respective deques
        self.p_history[p1].append(mid_p1)
        self.p_history[p2].append(mid_p2)
        spread = mid_p1 - mid_p2
        mean = np.mean(np.array(self.p_history[p1]) - np.array(self.p_history[p2]))
        std = np.std(np.array(self.p_history[p1]) - np.array(self.p_history[p2]))

        p1_pos = state.position.get(p1, 0)
        p2_pos = state.position.get(p2, 0)

        size_factor = min(max(1, abs((spread - mean) / std)), 5) if std > 0 else 1
        trade_size = int(self.base_size * size_factor)

        if std > 0:
            if spread > mean + self.entry_threshold * std + self.spread_buffer:
                if p1_pos > -self.max_position and p2_pos < self.max_position:
                    result[p1].append(Order(p1, int(round(mid_p1- 1)), -trade_size))
                    result[p2].append(Order(p2, int(round(mid_p2+ 1)), trade_size))

            elif spread < mean - self.entry_threshold * std - self.spread_buffer:
                if p1_pos < self.max_position and p2_pos > -self.max_position:
                    result[p1].append(Order(p1, int(round(mid_p1+ 1)), trade_size))
                    result[p2].append(Order(p2, int(round(mid_p2- 1)), -trade_size))

            elif abs(spread - mean) < self.exit_threshold * std:
                if p1_pos > 0:
                    result[p1].append(Order(p1, int(round(mid_p1- 1)), -p1_pos))
                if p1_pos < 0:
                    result[p1].append(Order(p1, int(round(mid_p1+ 1)), -p1_pos))
                if p2_pos > 0:
                    result[p2].append(Order(p2, int(round(mid_p2- 1)), -p2_pos))
                if p2_pos < 0:
                    result[p2].append(Order(p2, int(round(mid_p2+ 1)), -p2_pos))

        inv_skew = self.skew_param * p1_pos
        adjusted_mid = mid_p1- inv_skew
        half_spread = self.base_spread / 2
        bid_price = int(round(adjusted_mid - half_spread))
        ask_price = int(round(adjusted_mid + half_spread))

        buy_size = self.base_size if p1_pos < self.max_position else 1
        sell_size = self.base_size if p1_pos > -self.max_position else 1

        if bid_price > 0:
            result[p1].append(Order(p1, bid_price, buy_size))
        if ask_price > 0:
            result[p1].append(Order(p1, ask_price, -sell_size))
        result.pop(p1)
        return result
    
    def get_mid_price(self, order_depth: OrderDepth):
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return None
    
    def norm_cdf(self, x, mu=0, sigma=1):
        """Cumulative distribution function for a normal distribution."""
        z = (x - mu) / (sigma * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))
    def norm_pdf(self, x, mu=0, sigma=1):
        """Probability density function of a normal distribution."""
        coef = 1 / (sigma * math.sqrt(2 * math.pi))
        exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
        return coef * math.exp(exponent)

    def black_scholes_call(self, S, K, T, r, sigma):
        #Calculate the price of a European call option using Black-Scholes formula.
        if sigma <= 0:
            # Handle invalid volatility
            return max(0, S-K * np.exp(-r*T))
        
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        price = S * self.norm_cdf(d1) - K * np.exp(-r * T) * self.norm_cdf(d2)
        delta = self.norm_cdf(d1)
        return price, delta

    def vol_trade(self, state: TradingState, p, t, under = "VOLCANIC_ROCK") -> List[Order]:
        result = {p: [], under: []}
        if p not in state.order_depths or under not in state.order_depths:
            return result, 0, ""
        
        def implied_volatility_newton(S, V, K, T, r=0.0, initial_guess=0.3, 
                             precision=1e-8, max_iterations=100):
            #Calculate implied volatility using Newton-Raphson method.
            # Basic error checking
            # if V <= 0:
            #     return 0, 0  # Option has no value
            
            # if V >= S:
            #     return float('inf'), float('inf') # Option price exceeds underlying value
            
            intrinsic = max(0, S - K * np.exp(-r * T))
            if V < intrinsic:
                return 0, 1  # Option price below intrinsic value
            
            # if T <= 0:
            #     return float('nan'), float('nan')  # Expired option
            
            # Initial guess
            sigma = initial_guess
            
            # Calculate vega (derivative of price with respect to volatility)
            def vega(sigma):
                d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                return S * np.sqrt(T) * self.norm_pdf(d1)
            
            for i in range(max_iterations):
                price, delta = self.black_scholes_call(S, K, T, r, sigma)
                price_diff = price - V
                
                if abs(price_diff) < precision:
                    return sigma, delta
                
                v = vega(sigma)
                if abs(v) < 1e-10:
                    # Avoid division by zero
                    return float('nan')
                
                # Newton-Raphson update
                sigma = sigma - price_diff / v
                
                # Handle sigma going negative or too high
                if sigma <= 0:
                    sigma = 0.001
                elif sigma > 50:
                    return float('nan')
            
            # If didn't converge within max iterations
            return float('nan')

        p_depth = state.order_depths[p]
        s_depth = state.order_depths[under]
        time_converted = int(t)
        T = (7000000 - time_converted) / 7000000
        S = self.get_mid_price(s_depth)
        K = int(p.split("_")[-1])
        K_perc = K / S
        V = self.get_mid_price(p_depth)
        if V is None:
            return result
        if S is None:
            return result
        iv, delta = implied_volatility_newton(S=S, V=V, K=K, T=T)
        m_t = np.log(K / S) / np.sqrt(T)
        added = 0
        added_under = 0
        a = 0
        # 1.66379726*m_t**2 + 0.00311211*m_t + 0.02251656
        if iv < 2.15 * m_t**2 + 0.019 * m_t + 0.0225 - 0.001:   
            for price in state.order_depths[p].sell_orders:
                if K is None or S is None or T is None:
                    return result
                if price is None or price == 0:
                    return result
                K_perc = K / price
                iv_c, delta = implied_volatility_newton(S=S, V=price, K=K, T=T)
                if iv_c is None:
                    return result
                m_t = np.log(K / price) / np.sqrt(T)
                if iv_c < 2.15 * m_t**2 + 0.019 * m_t + 0.0225 - 0.001:  
                    #best_ask = min(p_depth.sell_orders.keys())
                    best_ask = price
                    best_ask_size = state.order_depths[p].sell_orders[price]
                    # SIGNAL

                    if state.position.get(p, 0) + added < self.limits_volc[p]:
                        max_p_buy = self.limits_volc[p] - (state.position.get(p, 0) + added)  # how much we can buy before hitting limit
                        buy_size = min(abs(best_ask_size), max_p_buy)
                        added += buy_size
                        result[p].append(Order(p, best_ask, buy_size))

                        # Delta hedge
                        s_price_sell = max(state.order_depths[under].buy_orders.keys())
                        max_under_sell = self.limits_volc[under] + (state.position.get(under, 0) + added_under) # how much we can sell before hitting -limit
                        hedge_size = min(int(round(delta * buy_size)), max_under_sell)
                        if hedge_size > 0:
                            added_under -= hedge_size
                            result[under].append(Order(under, s_price_sell, -hedge_size))

                    # self.pos[p] -= sell_size
                    # self.pos['VOLCANIC_ROCK'] -= round(0.5 * sell_size)

        #self.tick_vol[p] += 1
        added = 0
        added_under = 0
        b = 0
        m_t = np.log(K / S) / np.sqrt(T)
        if iv >= 2.15 * m_t**2 + 0.019 * m_t + 0.0225:  
            for price in state.order_depths[p].buy_orders:
                if K is None or S is None or T is None:
                    return result
                if price is None or price == 0:
                    return result
                K_perc = K / price
                iv_c, delta = implied_volatility_newton(S=S, V=price, K=K_perc, T=T)
                if iv_c is None:
                    return result
                m_t = np.log(K / price) / np.sqrt(T)
                if iv_c >= 2.15 * m_t**2 + 0.019 * m_t + 0.0225: 
                    #best_ask = min(p_depth.sell_orders.keys())
                    best_bid = price
                    best_bid_size = state.order_depths[p].buy_orders[price]
                    # SIGNAL
                    if (state.position.get(p, 0) + added) > -self.limits_volc[p]:
                        max_p_sell = (state.position.get(p, 0) + added) + self.limits_volc[p]  # how much we can sell before hitting -limit
                        sell_size = min(best_bid_size, max_p_sell)
                        added -= sell_size
                        result[p].append(Order(p, best_bid, int(-sell_size)))
                        #result[under].append(Order(under, s_price_buy, int(round(delta * sell_size))))

                        # self.pos[p] -= sell_size
                        # self.pos['VOLCANIC_ROCK'] -= round(0.5 * sell_size)
                                                # Delta hedge
                        s_price_buy = max(state.order_depths[under].sell_orders.keys())
                        max_under_buy = self.limits_volc[under] - (state.position.get(under, 0) + added_under) # how much we can sell before hitting -limit
                        hedge_size = min(int(round(delta * sell_size)), max_under_buy)
                        if hedge_size > 0:
                            added_under += hedge_size
                            result[under].append(Order(under, s_price_buy, hedge_size))

        # === Continuous Delta Hedging ===
        # Estimate net delta of options position
        opt_pos = state.position.get(p, 0)
        opt_price = self.get_mid_price(p_depth)
        _, total_delta = implied_volatility_newton(S=S, V=opt_price, K=K, T=T)

        net_opt_delta = opt_pos * total_delta
        underlying_pos = state.position.get(under, 0)
        desired_underlying_pos = -int(round(net_opt_delta))

        hedge_diff = desired_underlying_pos - underlying_pos

        if hedge_diff != 0:
            if hedge_diff > 0:
                # Buy underlying
                best_ask = min(s_depth.sell_orders.keys())
                max_buy = self.limits_volc[under] - underlying_pos
                hedge_size = min(hedge_diff, max_buy)
                if hedge_size > 0:
                    result[under].append(Order(under, best_ask, hedge_size))
            else:
                # Sell underlying
                best_bid = max(s_depth.buy_orders.keys())
                max_sell = self.limits_volc[under] + underlying_pos
                hedge_size = min(-hedge_diff, max_sell)
                if hedge_size > 0:
                    result[under].append(Order(under, best_bid, -hedge_size))

        #self.tick_vol[p] += 1
        return result
    
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        # if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
        #     rainforest_position = (
        #         state.position[Product.RAINFOREST_RESIN]
        #         if Product.RAINFOREST_RESIN in state.position
        #         else 0
        #     )
        #     rainforest_take_orders, buy_order_volume, sell_order_volume = (
        #         self.take_orders(
        #             Product.RAINFOREST_RESIN,
        #             state.order_depths[Product.RAINFOREST_RESIN],
        #             self.params[Product.RAINFOREST_RESIN]["fair_value"],
        #             self.params[Product.RAINFOREST_RESIN]["take_width"],
        #             rainforest_position,
        #         )
        #     )
        #     buy_order_volume = 0
        #     sell_order_volume = 0
        #     rainforest_clear_orders, buy_order_volume, sell_order_volume = (
        #         self.clear_orders(
        #             Product.RAINFOREST_RESIN,
        #             state.order_depths[Product.RAINFOREST_RESIN],
        #             self.params[Product.RAINFOREST_RESIN]["fair_value"],
        #             self.params[Product.RAINFOREST_RESIN]["clear_width"],
        #             rainforest_position,
        #             buy_order_volume,
        #             sell_order_volume,
        #         )
        #     )
        #     rainforest_make_orders, _, _ = self.make_orders(
        #         Product.RAINFOREST_RESIN,
        #         state.order_depths[Product.RAINFOREST_RESIN],
        #         self.params[Product.RAINFOREST_RESIN]["fair_value"],
        #         rainforest_position,
        #         buy_order_volume,
        #         sell_order_volume,
        #         self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
        #         self.params[Product.RAINFOREST_RESIN]["join_edge"],
        #         self.params[Product.RAINFOREST_RESIN]["default_edge"],
        #         True,
        #         self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
        #     )
        #     result[Product.RAINFOREST_RESIN] = (
        #          rainforest_clear_orders + rainforest_make_orders + rainforest_take_orders
        #     )
        # if Product.KELP in self.params and Product.KELP in state.order_depths:
        #     kelp_position = (
        #         state.position[Product.KELP]
        #         if Product.KELP in state.position
        #         else 0
        #     )
        #     kelp_fair_value = self.kelp_fair_value(
        #         state.order_depths[Product.KELP], traderObject
        #     )
        #    # Use our specialized take strategy for KELP
        #     kelp_take_orders, buy_order_volume, sell_order_volume = self.kelp_take_strategy(
        #         state.order_depths[Product.KELP],
        #         kelp_position,
        #         traderObject,
        #         0,
        #         0
        #     )
        #     kelp_clear_orders, buy_order_volume, sell_order_volume = (
        #         self.clear_orders(
        #             Product.KELP,
        #             state.order_depths[Product.KELP],
        #             kelp_fair_value,
        #             self.params[Product.KELP]["clear_width"],
        #             kelp_position,
        #             buy_order_volume,
        #             sell_order_volume,
        #         )
        #     )
        #     kelp_make_orders, _, _ = self.make_orders_kelp(
        #     state.order_depths[Product.KELP],
        #     kelp_fair_value,
        #     kelp_position,
        #     buy_order_volume,
        #     sell_order_volume,
        #     self.params[Product.KELP]["disregard_edge"],
        #     self.params[Product.KELP]["join_edge"],
        #     self.params[Product.KELP]["default_edge"],
        #     self.params[Product.KELP]["adverse_volume"],
        #     )
        #     result[Product.KELP] = (
        #          kelp_clear_orders + kelp_make_orders + kelp_take_orders
        #     )
        #     if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
        #         order_depth = state.order_depths[Product.SQUID_INK]
        #         position = state.position.get(Product.SQUID_INK, 0)
        #         fair_value = self.fair_value_squid_ink(order_depth)

        #         if fair_value is not None:
        #             take_orders, buy_vol, sell_vol = self.take_orders(
        #             Product.SQUID_INK,
        #             order_depth,
        #             fair_value,
        #             self.params[Product.SQUID_INK]["take_width"],
        #             position,
        #             )

        #         clear_orders, buy_vol, sell_vol = self.clear_orders(
        #             Product.SQUID_INK,
        #             order_depth,
        #             fair_value,
        #             self.params[Product.SQUID_INK]["clear_width"],
        #             position,
        #             buy_vol,
        #             sell_vol,
        #         )

        #         make_orders, _, _ = self.make_orders(
        #             Product.SQUID_INK,
        #             order_depth,
        #             fair_value,
        #             position,
        #             buy_vol,
        #             sell_vol,
        #             self.params[Product.SQUID_INK]["disregard_edge"],
        #             self.params[Product.SQUID_INK]["join_edge"],
        #             self.params[Product.SQUID_INK]["default_edge"],
        #             True,
        #             self.params[Product.SQUID_INK]["soft_position_limit"],
        #         )

        #         result[Product.SQUID_INK] = take_orders +clear_orders + make_orders

        # arb_orders = self.synthetic_real_arb(state)
        # for order in arb_orders:
        #     if order.symbol in result:
        #         result[order.symbol].append(order)
        #     else:
        #         result[order.symbol] = [order]
        # basket_orders = self.trader_max(state)
        # for asset, orders_list in basket_orders.items():
        #     if asset in result:
        #         result[asset].extend(orders_list)
        #     else:
        #         result[asset] = orders_list
        # "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750", "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500"
        for p2 in ["VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750", "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250"]:
            temp_result = self.options_trader(state,p2)
            self.p_history.pop("VOLCANIC_ROCK_VOUCHER_10500",None)
            result[p2] = temp_result[p2]   
        for p in ["VOLCANIC_ROCK_VOUCHER_10500"]:
            vol_trade_result = self.vol_trade(state, p = p, t = self.tick)
            if p in result:
                result[p].extend(vol_trade_result[p])
            else:
                result[p] = vol_trade_result[p]
            if 'VOLCANIC_ROCK' in result:
                result['VOLCANIC_ROCK'].extend(vol_trade_result['VOLCANIC_ROCK'])
            else:
                result['VOLCANIC_ROCK'] = vol_trade_result['VOLCANIC_ROCK']
        self.tick += 100
        



        #result.pop("CROISSANTS")
        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        #logger.flush(state,result,conversions,traderData)

        return result, conversions, traderData
