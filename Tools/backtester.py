from typing import Dict, List, Any
import pandas as pd
import json
from collections import defaultdict
from datamodel import TradingState, Listing, OrderDepth, Trade, Observation, Order, UserId

class Backtester:
    def __init__(self, trader, listings: Dict[str, Listing], position_limit: Dict[str, int], fair_marks, 
                 market_data: pd.DataFrame, trade_history: pd.DataFrame, file_name: str = None):
        self.trader = trader
        self.listings = listings
        self.market_data = market_data
        self.position_limit = position_limit
        self.fair_marks = fair_marks
        self.trade_history = trade_history.sort_values(by=['timestamp', 'symbol'])
        self.file_name = file_name

        self.observations = [Observation({}, {}) for _ in range(len(market_data))]

        self.current_position = {product: 0 for product in self.listings.keys()}
        self.pnl_history = []
        self.pnl = {product: 0 for product in self.listings.keys()}
        self.cash = {product: 0 for product in self.listings.keys()}
        self.trades = []
        self.sandbox_logs = []
    def run(self)_
        traderData = ""
        timestamp_group_md = self.market_data.groupby('timestamp')
        timestamp_group_th = self.trade_history.groupby('timestamp')

        own_trades = defaultdict(list)
        market_data_dict = defaultdict(list)
        pnl_product = defaultdict(list)

        trade_history_dict = {}
        for timestamp, group in timestamp_group_th:
            trades = []
            for _, row in group.iterrows():
                symbol = row['symbol']
                price = row['price']
                quantity = row['quantity']
                buyer = row['buyer'] if pd.notnull(row['buyer']) else "" 
                seller = row['seller'] if pd.notnull(row['seller']) else "" 
                trade = Trade(symbol, price, quantity, buyer, seller)
                trades.append(trade)
            trade_history_dict[timestamp] = trades

        for timestamp, group in timestamp_group_md:
            order_depth = 
