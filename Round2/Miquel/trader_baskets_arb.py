from datamodel import Order, OrderDepth, TradingState
from typing import List
import jsonpickle
from Logger import Logger
logger = Logger()

class Trader:
    def __init__(self):
        self.arb_threshold = 50  # Arbitrage threshold
        self.arb_threshold2 = 50  # Arbitrage threshold for second basket
    def synthetic_real_arb(self, state: TradingState) -> List[Order]:
        """
        Arbitrage between the synthetic basket and the real basket.
        
        Synthetic Basket is constructed as:
            6 CROISSANTS + 3 JAMS + 1 DJEMBES
            
        Real Basket is represented by:
            PICNIC_BASKET1
        
        Compute the mid-price for the synthetic basket from the underlying instruments and
        compare it to the mid-price for PICNIC_BASKET1.
        
        If:
          spread = (real basket mid-price) - (synthetic basket price)
          
        is greater than a positive threshold, then the real basket is overpriced – so short the real basket
        and long the synthetic replication (i.e. buy underlying instruments). If the spread is below negative the threshold,
        take the opposite side.
        """
        orders: List[Order] = []
        
        # Required instruments: Underlying components and the real basket.
        required_products = ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1"]
        for prod in required_products:
            if prod not in state.order_depths:
                return orders  # Abort if any required order book is missing
        
        # Retrieve order depths for underlying instruments and the real basket.
        cd: OrderDepth = state.order_depths["CROISSANTS"]
        jd: OrderDepth = state.order_depths["JAMS"]
        dd: OrderDepth = state.order_depths["DJEMBES"]
        pb_depth: OrderDepth = state.order_depths["PICNIC_BASKET1"]
        pb2_depth: OrderDepth = state.order_depths["PICNIC_BASKET2"]
        
        # Ensure both buy and sell orders exist for all instruments.
        if not (cd.buy_orders and cd.sell_orders and
                jd.buy_orders and jd.sell_orders and
                dd.buy_orders and dd.sell_orders and
                pb_depth.buy_orders and pb_depth.sell_orders):
            return orders
        
        # Compute mid-prices for the underlying instruments.
        mid_c = (max(cd.buy_orders.keys()) + min(cd.sell_orders.keys())) / 2
        mid_j = (max(jd.buy_orders.keys()) + min(jd.sell_orders.keys())) / 2
        mid_d = (max(dd.buy_orders.keys()) + min(dd.sell_orders.keys())) / 2
        
        # Calculate the synthetic basket price.
        synthetic_price = 6 * mid_c + 3 * mid_j + 1 * mid_d
        synthetic_price2 = 4 * mid_c + 2 * mid_j 
        
        # Compute the mid-price for the real basket (PICNIC_BASKET1).
        mid_basket = (max(pb_depth.buy_orders.keys()) + min(pb_depth.sell_orders.keys())) / 2
        mid_basket2 = (max(pb2_depth.buy_orders.keys()) + min(pb2_depth.sell_orders.keys())) / 2
        
        # Compute the arbitrage spread.
        spread = mid_basket - synthetic_price
        spread2 = mid_basket2 - synthetic_price2
        
        # If spread is significantly positive, the real basket is trading at a premium.
        if spread > self.arb_threshold:
            # Short real basket (sell at best bid) and buy synthetic basket (buy underlying at best ask)
            real_sell_price = max(pb_depth.buy_orders.keys())
            orders.append(Order("PICNIC_BASKET1", real_sell_price, -1))
            
            buy_c_price = min(cd.sell_orders.keys())
            buy_j_price = min(jd.sell_orders.keys())
            buy_d_price = min(dd.sell_orders.keys())
            # orders.append(Order("CROISSANTS", buy_c_price, 6))
            # orders.append(Order("JAMS", buy_j_price, 3))
            # orders.append(Order("DJEMBES", buy_d_price, 1))
        
        # If spread is significantly negative, the real basket is trading at a discount.
        elif spread < -self.arb_threshold:
            # Buy real basket (at best ask) and short synthetic basket (short underlying at best bid)
            real_buy_price = min(pb_depth.sell_orders.keys())
            orders.append(Order("" \
            "PICNIC_BASKET1", real_buy_price, 1))
            
            sell_c_price = max(cd.buy_orders.keys())
            sell_j_price = max(jd.buy_orders.keys())
            sell_d_price = max(dd.buy_orders.keys())
            # orders.append(Order("CROISSANTS", sell_c_price, -6))
            # orders.append(Order("JAMS", sell_j_price, -3))
            # orders.append(Order("DJEMBES", sell_d_price, -1))
        if spread2>self.arb_threshold2:
            real_sell_price = max(pb2_depth.buy_orders.keys())
            orders.append(Order("PICNIC_BASKET2", real_sell_price, -1))
            
            buy_c_price = min(cd.sell_orders.keys())
            buy_j_price = min(jd.sell_orders.keys())
            # orders.append(Order("CROISSANTS", buy_c_price, 4))
            # orders.append(Order("JAMS", buy_j_price, 2))
        elif spread2<-self.arb_threshold2:
            real_buy_price = min(pb2_depth.sell_orders.keys())
            orders.append(Order("PICNIC_BASKET2", real_buy_price, 1))
            
            sell_c_price = max(cd.buy_orders.keys())
            sell_j_price = max(jd.buy_orders.keys())
            # orders.append(Order("CROISSANTS", sell_c_price, -4))
            # orders.append(Order("JAMS", sell_j_price, -2))
        
        
        return orders

    def run(self, state: TradingState):
        """
        The main run method for the Trader.
        This method applies the synthetic vs. real basket arbitrage logic and returns the orders.
        """
        arb_orders = self.synthetic_real_arb(state)
        result = {}
        for order in arb_orders:
            if order.symbol in result:
                result[order.symbol].append(order)
            else:
                result[order.symbol] = [order]
        traderData = jsonpickle.encode({})
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
