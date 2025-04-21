from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math
from typing import Dict 
from collections import deque
# from Logger import Logger
# logger = Logger()

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    MACARON = "MAGNIFICENT_MACARONS"
    BASKET1 = "PICNIC_BASKET1"
    BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"


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
    },

    Product.MACARON : {
            "coefficients" : np.array([-0.0612462,   0.14233246, -1.45858172, -1.47483229,  1.33068985, -0.36351738]),
            "intercept" : 5.787939456936101,
            
            # Trading parameters
            "default_edge" : 5,  # Default edge (similar to rainforest resin)
            "disregard_edge" : 0,  # Disregard orders near fair value
            "join_edge" : 2 ,# oin orders within this edge
            "soft_position_limit" : 75
}
}


class Trader:
    def __init__(self):
        self.params = PARAMS
        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50,
                      Product.SQUID_INK: 50,Product.MACARON:75,
                      Product.CROISSANTS:250, Product.JAMS: 350,
                      Product.DJEMBES: 60, Product.DJEMBES:60,
                      Product.BASKET1: 60, Product.BASKET2: 100,
                      Product.VOLCANIC_ROCK: 400,
                      Product.VOLCANIC_ROCK_VOUCHER_9500:200,
                      Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
                      Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
                      Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
                      Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
                      Product.MACARON:75
                      }
    # First when there are no trades just post 1 bid and ask at the best ask and bid to take both sides and see who fills them:
    # Then when we have info about who is filling that trades if it is a bad trader (e.g Paris) post a bigger trade against that bot
    # If we have traded against one of the bots that are good try to break even with that single trade (post again same bid/ask as ask/bid)
    # Do that on every iteration?
    """
    def sendTestTrade(self,state:TradingState):
        #Send one bid one ask at the best ask best bid respectively:
        orders = {}
        for product in state.order_depths.keys():
            buy_orders = state.order_depths[product].buy_orders.keys()
            sell_orders = state.order_depths[product].sell_orders.keys()
            if buy_orders and sell_orders:
                best_bid = max(buy_orders)
                best_ask = min(sell_orders)
                product_orders = [Order(product,best_bid,-1),Order(product,best_ask,1)]
                orders[product] = product_orders
        return orders"""
    
    """def analyzeMarketTrades(self,state:TradingState):
        marketTrades = state.market_trades
        # For now I'll asume that all the bots are good except paris
        for product in marketTrades.keys():
            trades = marketTrades[product]
            for trade in trades:
                logger.print(f"Porcodio buyer{trade.buyer} seller {trade.seller}")
        return"""
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

    
    def tradevs(self,state:TradingState,name,productsToTrade:List):
        marketTrades = state.market_trades
        result = {}
        for product in productsToTrade:
            try:
                position = state.position.get(product,0)
                availableBuy = self.LIMIT[product] - position
                availableSell = self.LIMIT[product] + position
                trades = marketTrades[product]
                productTrades= []
                for trade in trades:
                    if trade.buyer == name:
                        sellQty = trade.quantity
                        sellPrice= trade.price
                        sellQty = min(sellQty,availableSell)
                        sellQty = availableSell
                        sellTrade= Order(product,int(sellPrice),int(-sellQty))
                        productTrades.append(sellTrade)
                    if trade.seller ==name:
                        buyQty= trade.quantity
                        buyPrice= trade.price
                        buyQty = min(buyQty,availableBuy)
                        buyQty = availableBuy
                        buyTrade= Order(product,int(buyPrice),int(buyQty))
                        productTrades.append(buyTrade)
                result[product] = productTrades
            except:
                # print("Can't Trade this product")
                continue
        return result
    def tradeas(self,state:TradingState,name,productsToTrade:List):
        marketTrades = state.market_trades
        result = {}
        for product in productsToTrade:
            try:
                position = state.position.get(product,0)
                availableBuy = self.LIMIT[product] - position
                availableSell = self.LIMIT[product] + position
                trades = marketTrades[product]
                productTrades= []
                for trade in trades:
                    if trade.buyer == name:
                        sellQty = trade.quantity
                        sellPrice= trade.price
                        sellQty = min(sellQty,availableSell)
                        sellTrade= Order(product,int(sellPrice),int(sellQty))
                        productTrades.append(sellTrade)
                    if trade.seller ==name:
                        buyQty= trade.quantity
                        buyPrice= trade.price
                        buyQty = min(buyQty,availableBuy)
                        buyTrade= Order(product,int(buyPrice),int(-buyQty))
                        productTrades.append(buyTrade)
                result[product] = productTrades
            except:
                # print("Can't Trade this product")
                continue
        return result
    def run(self,state:TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        productsToTrade = [Product.DJEMBES,Product.VOLCANIC_ROCK_VOUCHER_10000,
                           Product.VOLCANIC_ROCK_VOUCHER_10250,Product.VOLCANIC_ROCK_VOUCHER_9500,Product.VOLCANIC_ROCK_VOUCHER_9750]
        allProducts = state.listings.keys()
        # productsToTrade = ["SQUID_INK"]
        result = self.tradevs(state,"Penelope",productsToTrade)
        # result ={}
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            rainforest_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            rainforest_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    rainforest_position,
                )
            )
            buy_order_volume = 0
            sell_order_volume = 0
            rainforest_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    rainforest_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            rainforest_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                rainforest_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                 rainforest_clear_orders + rainforest_make_orders + rainforest_take_orders
            )
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
           # Use our specialized take strategy for KELP
            kelp_take_orders, buy_order_volume, sell_order_volume = self.kelp_take_strategy(
                state.order_depths[Product.KELP],
                kelp_position,
                traderObject,
                0,
                0
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            kelp_make_orders, _, _ = self.make_orders_kelp(
            state.order_depths[Product.KELP],
            kelp_fair_value,
            kelp_position,
            buy_order_volume,
            sell_order_volume,
            self.params[Product.KELP]["disregard_edge"],
            self.params[Product.KELP]["join_edge"],
            self.params[Product.KELP]["default_edge"],
            self.params[Product.KELP]["adverse_volume"],
            )
            result[Product.KELP] = (
                 kelp_clear_orders + kelp_make_orders + kelp_take_orders
            )
            if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
                order_depth = state.order_depths[Product.SQUID_INK]
                position = state.position.get(Product.SQUID_INK, 0)
                fair_value = self.fair_value_squid_ink(order_depth)

                if fair_value is not None:
                    take_orders, buy_vol, sell_vol = self.take_orders(
                    Product.SQUID_INK,
                    order_depth,
                    fair_value,
                    self.params[Product.SQUID_INK]["take_width"],
                    position,
                    )

                clear_orders, buy_vol, sell_vol = self.clear_orders(
                    Product.SQUID_INK,
                    order_depth,
                    fair_value,
                    self.params[Product.SQUID_INK]["clear_width"],
                    position,
                    buy_vol,
                    sell_vol,
                )

                make_orders, _, _ = self.make_orders(
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

                result[Product.SQUID_INK] = take_orders +clear_orders + make_orders
            arb_orders = self.synthetic_real_arb(state)
            for order in arb_orders:
                if order.symbol in result:
                    result[order.symbol].append(order)
                else:
                    result[order.symbol] = [order]

        traderData = jsonpickle.encode(traderObject)
        # logger.flush(state,result,1,traderData)
        return result,1,traderData


