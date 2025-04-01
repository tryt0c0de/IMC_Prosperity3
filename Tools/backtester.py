#Backtester for the IMC Trading Strategy Replicating the competition by reading the csv file
#Based on the backtester from the previous competition made by the linear utility team
import json
from collections import defaultdict
from datamodel import TradingState, Order, Listing, OrderDepth, Trade, Observation, Order, UserId
import pandas as pd

class Backtester:
    def __init__(self, trader,listings: dict[str,Listing], position_limit: dict[str,int], fair_marks,
                 market_data: pd.DataFrame, trade_history:pd.DataFrame,file_name: str = None):
        self.trader = trader
        self.listings = listings
        self.position_limit = position_limit
        self.fair_marks = fair_marks
        self.market_data = market_data
        self.trade_history = trade_history.sort_values(by = ["timestamp","symbol"])
        self.file_name = file_name
        
        self.observations = [Observation({},{}) for _ in range(len(self.market_data))]

        self.current_position = {product: 0 for product in self.listings.keys()}
        self.pnl_history = []
        self.pnl = {product: 0 for product in self.listings.keys()}
        self.cash = {product: 0 for product in self.listings.keys()}
        self.trades = []
        self.sandbox_logs = []



        
        


