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
                        sellTrade= Order(product,sellPrice,-availableSell)
                        productTrades.append(sellTrade)
                    if trade.seller ==name:
                        buyQty= trade.quantity
                        buyPrice= trade.price
                        buyTrade= Order(product,buyPrice,availableBuy)
                        productTrades.append(buyTrade)
                result[product] = productTrades
            except:
                print("Can't Trade this product")
        return result
    def run(self,state:TradingState):
        productsToTrade = [Product.DJEMBES,Product.RAINFOREST_RESIN,Product.SQUID_INK,Product.VOLCANIC_ROCK_VOUCHER_10000,
                           Product.VOLCANIC_ROCK_VOUCHER_10250,Product.VOLCANIC_ROCK_VOUCHER_9500,Product.VOLCANIC_ROCK_VOUCHER_9750]
        result = self.tradevs(state,"Penelope",productsToTrade)
        logger.flush(state,result,1,"")
        return result,1,""


