from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import numpy as np
from collections import deque
import jsonpickle
from Logger import Logger
logger = Logger()

class Product: 
    MACARON = "MAGNIFICENT_MACARONS"
PARAMS={
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
        # Position limits
        self.LIMIT = {
            Product.MACARON: 75
        }
        
        
        # Store price history for EMA calculation
        self.mid_price_history = deque(maxlen=200)
        
        # Regression model coefficients from training
        
        # Trading parameters
        
    def calculate_ema(self, values, span):
        """Simple EMA calculation"""
        if len(values) < 3:
            return sum(values) / len(values) if values else None
            
        alpha = 2 / (span + 1)
        # alpha =1
        ema = values[0]
        for price in values[1:]:
            ema = (price * alpha) + (ema * (1 - alpha))

        return ema
    
    
    def calculate_fair_price_macarons(self, state: TradingState):
        """Calculate fair price using regression model"""
        # Get market data for mid price calculation
        od = state.order_depths[Product.MACARON]
        if not od.buy_orders or not od.sell_orders:
            return None
            
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        
        # Store mid price for EMA calculation
        self.mid_price_history.append(mid_price)
        
        # Get feature values from market data
        sunlight_index = state.observations.conversionObservations[Product.MACARON].sunlightIndex
        sugar_price = state.observations.conversionObservations[Product.MACARON].sugarPrice  
        export_tariff = state.observations.conversionObservations[Product.MACARON].exportTariff
        import_tariff = state.observations.conversionObservations[Product.MACARON].importTariff
        
        # Calculate EMAs for model features
        price_list = list(self.mid_price_history)
        if len(price_list) >= 50:
            ema_short = self.calculate_ema(price_list, 50)
            ema_medium = self.calculate_ema(price_list, 200) if len(price_list) >= 200 else ema_short
            ema_diff = ema_short - ema_medium
            
            # Create feature vector
            features = np.array([sunlight_index, sugar_price, export_tariff, import_tariff, 
                               ema_short, ema_medium])
            
            # Calculate fair price using regression model
            fair_price = PARAMS[Product.MACARON]["intercept"] + np.dot(PARAMS[Product.MACARON]["coefficients"], features)
            
            return fair_price
        else:
            # Not enough data for EMAs, use mid price as fallback
            return mid_price
    
    def make_orders_macarons(self, product, order_depth, fair_value, position, disregard_edge, join_edge, default_edge, soft_position_limit):
        """Market making function based on rainforest resin logic"""
        orders = []
        
        # Find asks above fair value (potential sells)
        asks_above_fair = [
            price for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        
        # Find bids below fair value (potential buys)
        bids_below_fair = [
            price for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]
        
        # Determine best prices to penny or join
        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None
        
        # Set ask price
        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny
        
        # Set bid price
        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair  # join
            else:
                bid = best_bid_below_fair + 1  # penny
        
        # Adjust prices based on position
        if position > soft_position_limit:
            ask -= 3  # Lower ask to encourage selling
        elif position < -soft_position_limit:
            bid += 3  # Raise bid to encourage buying
        
        # Calculate quantities
        buy_quantity = self.LIMIT[product] - position
        sell_quantity = self.LIMIT[product] + position
        
        # Place orders if quantities are positive
        if buy_quantity > 0:
            orders.append(Order(product, bid, buy_quantity))
            
        if sell_quantity > 0:
            orders.append(Order(product, ask, -sell_quantity))
            
        return orders
            
    def run(self, state: TradingState):
        result = {}
        
        # Get trader data from previous iteration
        trader_data = {}
        if state.traderData != None and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        
        # Check if macarons are in the order depths
        if Product.MACARON not in state.order_depths:
            return {}, 0, jsonpickle.encode(trader_data)
        
        # Get position
        position_macarons = state.position.get(Product.MACARON, 0)
        
        # Calculate fair value using regression model
        fair_value = self.calculate_fair_price_macarons(state)
        
        if fair_value is not None:
            # Generate market making orders
            macaron_orders = self.make_orders_macarons(
                Product.MACARON,
                state.order_depths[Product.MACARON],
                fair_value,
                position_macarons,
                PARAMS[Product.MACARON]["disregard_edge"],
                PARAMS[Product.MACARON]["join_edge"],
                PARAMS[Product.MACARON]["default_edge"],
                PARAMS[Product.MACARON]["soft_position_limit"]
            )
            
            result[Product.MACARON] = macaron_orders
        logger.flush(state,result,0,jsonpickle.encode(trader_data))
        return result, 0, jsonpickle.encode(trader_data)