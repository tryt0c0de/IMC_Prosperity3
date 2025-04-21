from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math
from typing import Dict 
from collections import deque
from Logger import Logger
logger = Logger()

class Product: 
    MACARON = "MAGNIFICENT_MACARONS"


class Trader:
    def __init__(self):
        # Initialize a window with maximum length of 10
        self.window_size = 10
        self.sunlight_index_window = deque(maxlen=self.window_size)
        self.LIMIT = {
            Product.MACARON: 75
        }
        self.prev_non_zero_derivative = 0

    def get_derivative(self, state: TradingState):
        current_sunlight_index = state.observations.conversionObservations[Product.MACARON].sunlightIndex
        
        # Add the current value to the window
        self.sunlight_index_window.append(current_sunlight_index)
        
        # Check if we have enough values to calculate the long-term derivative
        if len(self.sunlight_index_window) < self.window_size:
            # Window not yet filled, return None as requested
            return None
        
        # Window is full - calculate derivative between current and oldest value
        # The oldest value is at index 0 in the deque
        derivative = current_sunlight_index - self.sunlight_index_window[0]
        
        # Store non-zero derivatives for reference
        if derivative != 0:
            self.prev_non_zero_derivative = derivative
        
        logger.print(f"Sunlight derivative: {derivative}")
        return derivative
    def directional_trade(self, state: TradingState) -> List[Order]:
        orders = []
        derivative = self.get_derivative(state)
        available_to_buy =(self.LIMIT[Product.MACARON]) - state.position.get(Product.MACARON, 0)
        avalilable_to_sell = state.position.get(Product.MACARON, 0) +(self.LIMIT[Product.MACARON])
        sells= state.order_depths[Product.MACARON].sell_orders
        buys = state.order_depths[Product.MACARON].buy_orders
        conversions = 0 
        #available_to_buy= min(10, available_to_buy)
        #avalilable_to_sell = min(10, avalilable_to_sell)
        if not derivative:
            return [], 0, available_to_buy, avalilable_to_sell
        if derivative > 0:
            # If the sunlight index is increasing, sell MACARON
            orders.append(Order(Product.MACARON, max(buys),-avalilable_to_sell))
            conversions = 10
        elif derivative < 0:
            # If the sunlight index is decreasing, sell MACARON 
            orders.append(Order(Product.MACARON,min(sells), available_to_buy))
            conversions = -10
        elif derivative == 0:
            return [],conversions, available_to_buy, avalilable_to_sell

        return orders,conversions, available_to_buy, avalilable_to_sell
    def market_make_macaron(self, state: TradingState,buy_orders_volume,sell_orders_volume) -> List[Order]:
        od = state.order_depths[Product.MACARON]
        #When mixing both strats I'll need to get the available to buy and sell also from the directional trade
        available_to_buy = self.LIMIT[Product.MACARON] - state.position.get(Product.MACARON, 0)
        available_to_sell = state.position.get(Product.MACARON, 0)+ self.LIMIT[Product.MACARON]
        #Limit buy and sell positions to 10 or min available:
        if abs(state.position.get(Product.MACARON, 0)) > 20:
            return []
        available_to_buy = min(10, available_to_buy)
        available_to_sell = min(10, available_to_sell)
        if not od.buy_orders or not od.sell_orders:
            return []
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid = (best_bid + best_ask) / 2


        # I'm going to place better asks that the current asks very close to the best one if the price is going up
        derivative = self.get_derivative(state)
        spread = best_ask - best_bid    
        skew = 4
        if not derivative:
            return  []
        if derivative>0:
            bid_price = best_bid +spread/2 -skew
            ask_price = best_ask -1
        elif derivative <0:
            bid_price = best_bid +1
            ask_price = best_ask - spread/2 -skew
        elif derivative == 0: 
            #We can tighten up the spread since the directionality of the price is uncertain
            bid_price = best_bid + spread/2 - 2
            ask_price = best_ask - spread/2 + 2
        


        mm = []
        if available_to_buy> 0:
            mm.append(Order(Product.MACARON, int(bid_price),  available_to_buy))
        if available_to_sell> 0:
            mm.append(Order(Product.MACARON, int(ask_price), -available_to_sell))
        return mm
    def run(self,state: TradingState):
        result = {}
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        orders,conversions,buy_order_volumes,sell_order_volumes= self.directional_trade(state)
        # mm_orders = self.market_make_macaron(state,buy_order_volumes,sell_order_volumes)
        result[Product.MACARON] =orders 
        
        conversions = 0
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        return result,conversions, traderData