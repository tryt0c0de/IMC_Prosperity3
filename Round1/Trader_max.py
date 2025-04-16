import pandas as pd
from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Listing, Trade, Observation, ProsperityEncoder
from typing import List


class Trader:
    def __init__(self):
        params = [10, 100, 1]
        self.products = ['KELP', 'SQUID_INK']
        self.products = ['SQUID_INK']
        self.df = pd.DataFrame({col: [] for col in ['timestamp'] + [f'{c}_{prod}' for c in ['std', 'ewm_fast', 'ewm_slow', 'w_price', 'ub', 'spread'] for prod in self.products]})
        self.span_fast = int(params[0])
        self.span_slow = int(params[1])
        self.coef1 = 1+ (0.005 * params[2])
        self.coef2 = 1 + (0.01 * params[2])
        self.coef3 = 1 + (0.025 * params[2])
        self.max_holdings = {prod: 50 for prod in self.products}
        self.current_holdings = {prod: 0 for prod in self.products}
        #self.max_order_size = 10
        self.w_price = {}
        self.ewm_fast = {}
        self.ewm_slow = {}
        self.ub = {}
        self.spread = {}
        self.holding_ratio = {}
        self.std = {}

        self.touch_price = 0

        self.signal_df = pd.DataFrame({col:[] for col in ['timestamp', 'sig']})


    def run(self, state: TradingState):

        traderData = "SAMPLE"
        conversions = 1

        timestamp = state.timestamp
        self.signal_df.loc[len(self.signal_df)] = [timestamp, 0]
        result = {}

        for product in self.products:
            self.current_holdings[product] = state.position.get(product, 0)
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Get best bid and ask
            best_ask, best_ask_amount = zip(*list(order_depth.sell_orders.items())) if order_depth.sell_orders else ([], [])
            best_bid, best_bid_amount = zip(*list(order_depth.buy_orders.items())) if order_depth.buy_orders else ([], [])

            best_ask = list(best_ask)
            best_ask_amount = list(best_ask_amount)
            best_bid = list(best_bid)
            best_bid_amount = list(best_bid_amount)

            if not (best_ask_amount + best_bid_amount):
                if not self.df.empty:
                    for prod in self.products:
                        self.std[prod] = self.df[f'std_{prod}'].iloc[-1]
                        self.ewm_fast[prod] = self.df[f'ewm_fast_{prod}'].iloc[-1]
                        self.ewm_slow[prod] = self.df[f'ewm_slow_{prod}'].iloc[-1]
                        self.w_price[prod] = self.df[f'w_price_{prod}'].iloc[-1]
                        self.ub[prod] = self.df[f'ub_{prod}'].iloc[-1]
                        self.spread[prod] = self.df[f'spread_{prod}'].iloc[-1]
                    timestamp = self.df['timestamp'].iloc[-1] + 100
                else:

                    return result, conversions, traderData
                break

            self.w_price[product] = (best_bid[0] + best_ask[0]) / 2


            self.std[product], self.ewm_fast[product], self.ewm_slow[product], self.ub[product], self.spread[product] = 0,0,0,0,0


            look = 50
            if timestamp >= (look + self.span_slow) * 100:
                # ADJUST ?????
                def moving(span):
                    return pd.Series(self.df[f'w_price_{product}'].tolist() + [self.w_price[product]]).ewm(span=span, adjust=False)

                self.ewm_fast[product] = moving(self.span_fast).mean().iloc[-1]
                self.ewm_slow[product] = moving(self.span_slow).mean().iloc[-1]

                curr_long = (self.current_holdings[product] > 0)
                curr_short = (self.current_holdings[product] < 0)

                q = 0
                neutral = True
                self.spread[product] = self.ewm_slow[product]/self.ewm_fast[product]

                bound = 0#.005

                if self.spread[product] > 1+bound/2:
                    if curr_long:
                        q = -1
                    elif curr_short and self.touch_price != 0:
                        if self.touch_price/best_ask[0] > 1.05:
                            q = 1

                    elif self.df[f'spread_{product}'].iloc[-look:].min() < 1-bound/2 and self.spread[product] > 1+bound:
                        q = -1
                        neutral = False



                if self.spread[product] < 1-bound/2:
                    if curr_short:
                        q = 1
                    elif curr_long and self.touch_price != 0:
                        if best_bid[0]/self.touch_price > 1.05:
                            q = -1
                    elif self.df[f'spread_{product}'].iloc[-look:].max() > 1+bound/2 and self.spread[product] < 1-bound:
                        q = 1
                        neutral = False


                q *= self.max_holdings[product]

                if neutral and q != 0:
                    q = - self.current_holdings[product]


                if q != 0:
                    self.signal_df.loc[len(self.signal_df)-1] = [timestamp, 1 if neutral else q]


                if q < 0:
                    self.touch_price = best_bid[0]
                    orders.append(Order(product, self.touch_price, max(q, -best_bid_amount[0])))

                elif q > 0:
                    self.touch_price = best_ask[0]
                    orders.append(Order(product, self.touch_price, min(q, -best_ask_amount[0])))

                if neutral and q != 0:
                    self.touch_price = 0

            result[product] = orders

        # Update dataframe with new weighted prices
        self.df.loc[len(self.df)] = [timestamp] + [col[prod] for col in [self.std, self.ewm_fast, self.ewm_slow, self.w_price, self.ub, self.spread] for prod in self.products]




        return result, conversions, traderData
