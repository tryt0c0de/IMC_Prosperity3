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

class Trader:
    def __init__(self):
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
        self.coefs     = np.array([-0.05050389,  0.12043986, -1.61310752, -0.78988358,  1.33330671,
       -0.36972679])  # length 6
        self.intercept =15.57885144091017              # scalar

        # for realtime EMA update
        self.alpha_short  = 2/(50+1)
        self.alpha_medium = 2/(200+1)
        self.ema_short    = None
        self.ema_medium   = None

    def _update_ema(self, mid):
        if self.ema_short is None:
            self.ema_short = mid
            self.ema_medium = mid
        else:
            self.ema_short  = self.alpha_short  * mid + (1-self.alpha_short)  * self.ema_short
            self.ema_medium = self.alpha_medium * mid + (1-self.alpha_medium) * self.ema_medium
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
    
    def marketMakeMacarons(self,state:TradingState):
        observations = state.observations.conversionObservations[Product.MACARON]
        order_depths = state.order_depths
        if observations.sunlightIndex <56 or Product.MACARON not in order_depths.keys():
            #If less than 50 we cant market make or macaron not being traded
            return{}
        #else:
        bids = order_depths[Product.MACARON].buy_orders
        asks = order_depths[Product.MACARON].sell_orders
        best_bid= min(bids.keys())
        best_ask = max(asks.keys())
        mid = (best_bid+best_ask)/2
        spread = 4
        skew = 0
        buyPrice = int(mid-spread+skew)
        sellPrice = int(mid+spread-2)
        position = state.position.get(Product.MACARON,0)
        available_to_buy = self.LIMIT[Product.MACARON]-position
        available_to_sell = self.LIMIT[Product.MACARON]+position
        #Limit of buy or sell to 10 or min available to manage position:
        max_size = 50
        buyQty = min(max_size,available_to_buy)
        sellQty = min(max_size,available_to_sell)
        buyTrade = Order(Product.MACARON,buyPrice,buyQty)
        sellTrade = Order(Product.MACARON,sellPrice,-sellQty)
        result ={Product.MACARON:[buyTrade,sellTrade]}
        return result
    def run(self,state:TradingState):
        # productsToTrade = [Product.DJEMBES,Product.VOLCANIC_ROCK_VOUCHER_10000,
        #                    Product.VOLCANIC_ROCK_VOUCHER_10250,Product.VOLCANIC_ROCK_VOUCHER_9500,Product.VOLCANIC_ROCK_VOUCHER_9750]
        # allProducts = state.listings.keys()
        # # productsToTrade = ["SQUID_INK"]
        # result = self.tradevs(state,"Penelope",productsToTrade)
        # # result = {}
        # productsWith = allProducts-productsToTrade
        # productsWith = [Product.SQUID_INK]
        # # result2 = self.tradeas(state,"Camilla",productsWith)
        # result.update(result2)
        result = self.marketMakeMacarons(state)
        logger.flush(state,result,1,"")
        return result,1,"trader"
    