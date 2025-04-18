from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math
from typing import Dict 
from Logger import Logger
logger = Logger()

class Product: 
    MACARON = "MAGNIFICENT_MACARONS"


class Trader:
    def __init__(self):
        self.prev_sunlight_index = 0.0
        self.LIMIT = {
            Product.MACARON: 75
        }
        self.prev_non_zero_derivative =0
    

    def get_derivative(self, state: TradingState) -> float:
        current_sunlight_index = state.observations.conversionObservations[Product.MACARON].sunlightIndex
        if self.prev_sunlight_index == 0.0:
            self.prev_sunlight_index = current_sunlight_index
            return 0.0
        derivative = current_sunlight_index - self.prev_sunlight_index
        self.prev_sunlight_index = current_sunlight_index
        self.prev_non_zero_derivative = derivative if derivative != 0 else self.prev_non_zero_derivative
        return derivative
    def manage_position_macaron(self, state: TradingState) -> List[Order]:
        orders = []
        derivative = self.get_derivative(state)
        available_to_buy =self.LIMIT[Product.MACARON] - state.position.get(Product.MACARON, 0)
        avalilable_to_sell = state.position.get(Product.MACARON, 0) +self.LIMIT[Product.MACARON]
        sells= state.order_depths[Product.MACARON].sell_orders
        buys = state.order_depths[Product.MACARON].buy_orders
        conversions = 0 
        #available_to_buy= min(10, available_to_buy)
        #avalilable_to_sell = min(10, avalilable_to_sell)
        if derivative > 0:
            # If the sunlight index is increasing, sell MACARON
            orders.append(Order(Product.MACARON, max(buys),-avalilable_to_sell))
            conversions = 10
        elif derivative < 0:
            # If the sunlight index is decreasing, sell MACARON 
            orders.append(Order(Product.MACARON,min(sells), available_to_buy))
            conversions = -10

        return orders,conversions
    def run(self,state: TradingState):
        result = {}
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        orders,conversions = self.manage_position_macaron(state)
        result[Product.MACARON] = orders
        
        # conversions = 0
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        return result,conversions, traderData



