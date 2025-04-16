

def __init__(self):

        self.products = ['CROISSANTS', 'DJEMBES']

        self.mid = {}
        self.to_liquidate = {prod: False for prod in self.products}

        self.asset1 = 'DJEMBES'
        self.asset2 = ['CROISSANTS']
        self.threshold = 1
        self.prop = 1/2


        self.limits = {'CROISSANTS': 250, 'DJEMBES': 60}


        self.regression = {
                'DJEMBES': {
                'Intercept': 6913.009686452366,
                'CROISSANTS': 1.5177902648379593,
                'std': 23.282224239738
            }}


        self.multiple = 0
        for product in self.asset2:
            mul = int(self.limits[product]//abs(self.regression[self.asset1][product]))
            self.multiple = max(self.multiple, mul)


def take_bids(self, best_bid, best_bid_amount, q, asset, liquidate=False):
    orders = []
    n = len(best_bid)
    q0 = min(q, best_bid_amount[0])
    q1, q2 = 0, 0

    if q0 < q and n > 1:
        q1 = min(q - q0, best_bid_amount[1])

        if q0 + q1 < q and n > 2:
            q2 = min(q - q0 - q1, best_bid_amount[2])

    if liquidate or q0 + q1 + q2 == q:
        orders.append(Order(asset, best_bid[0], -q0))
        if q1:
            orders.append(Order(asset, best_bid[1], -q1))
        if q2:
            orders.append(Order(asset, best_bid[2], -q2))

    return orders


def take_asks(self, best_ask, best_ask_amount, q, asset, liquidate=False):
    orders = []
    n = len(best_ask)
    q0 = max(-q, best_ask_amount[0])
    q1, q2 = 0, 0

    if q0 > -q and n > 1:
        q1 = max(-q - q0, best_ask_amount[1])

        if q0 + q1 > -q and n > 2:
            q2 = max(-q - q0 - q1, best_ask_amount[2])

    if liquidate or q0 + q1 + q2 == -q:
        orders.append(Order(asset, best_ask[0], -q0))
        if q1:
            orders.append(Order(asset, best_ask[1], -q1))
        if q2:
            orders.append(Order(asset, best_ask[2], -q2))

    return orders


def liquidate(self, best_bid, best_bid_amount, best_ask, best_ask_amount, asset):
    q = self.current_holdings[asset]
    if q < 0:
        return self.take_asks(best_ask, best_ask_amount, abs(q), asset, liquidate=True)
    elif q > 0:
        return self.take_bids(best_bid, best_bid_amount, abs(q), asset, liquidate=True)
    return []




def trader_max(self, state: TradingState):

    result = {}
    result[self.asset1] = []
    for product in self.asset2:
        result[product] = []

    self.current_holdings[self.asset1] = state.position.get(self.asset1, 0)
    for x in self.asset2:
        self.current_holdings[x] = state.position.get(x, 0)

    order_depth_1 = state.order_depths[self.asset1]
    order_depth_2 = {x: state.order_depths[x] for x in self.asset2}


    # Get best bid and ask
    best_ask1, best_ask_amount1 = zip(*list(order_depth_1.sell_orders.items())) if order_depth_1.sell_orders else (
    [], [])
    best_bid1, best_bid_amount1 = zip(*list(order_depth_1.buy_orders.items())) if order_depth_1.buy_orders else (
    [], [])


    best_ask2, best_ask_amount2 = {},{}
    best_bid2, best_bid_amount2 = {},{}
    for x in self.asset2:
        best_ask2[x], best_ask_amount2[x] = zip(
            *list(order_depth_2[x].sell_orders.items())) if order_depth_2[x].sell_orders else (
            [], [])
        best_bid2[x], best_bid_amount2[x] = zip(
            *list(order_depth_2[x].buy_orders.items())) if order_depth_2[x].buy_orders else (
            [], [])

        best_ask2[x], best_ask_amount2[x] = list(best_ask2[x]), list(best_ask_amount2[x])
        best_bid2[x], best_bid_amount2[x] = list(best_bid2[x]), list(best_bid_amount2[x])


    if not (best_ask_amount1 + best_bid_amount1):
        return {self.asset1: 0, self.asset2[0]: 0}


    self.mid[self.asset1] = (best_bid1[0] + best_ask1[0]) / 2
    for x in self.asset2:
        self.mid[x] = (best_bid2[x][0] + best_ask2[x][0]) / 2


    multiples = list(range(1, 1+self.multiple))[::-1]

    curr_long = (self.current_holdings[self.asset1] > 0)
    curr_short = (self.current_holdings[self.asset1] < 0)

    spread = self.mid[self.asset1] - self.regression[self.asset1]['Intercept']
    for product in self.asset2:
        spread -= self.regression[self.asset1][product] * self.mid[product]
    spread /= self.regression[self.asset1]['std']

    q = 0
    neutral = False

    if not (curr_short or curr_long):
        self.to_liquidate[self.asset1] = False
        for x in self.asset2:
            self.to_liquidate[x] = False

    if (curr_long and spread > self.threshold*self.prop) or (curr_short and spread < -self.threshold*self.prop) or self.to_liquidate[self.asset1] or True in [self.to_liquidate[x] for x in self.asset2]:
        neutral = True

    elif spread > self.threshold and not curr_short:
        q = 1
    elif spread < -self.threshold and not curr_long:
        q = -1

    if q > 0 :
        for mul in multiples:
            bid = self.take_bids(best_bid1, best_bid_amount1, mul, self.asset1)
            if not bid:
                continue
            asks = {}
            for product in self.asset2:
                desired_quant = round(self.regression[self.asset1][product]*mul)
                ask = self.take_asks(best_ask2[product], best_ask_amount2[product], desired_quant, product)
                if not (ask and desired_quant):
                    asks = {}
                    break
                asks[product] = ask
            if not asks:
                continue

            result[self.asset1] = bid
            for product in self.asset2:
                result[product] = asks[product]
            break

    if q < 0:
        for mul in multiples:
            ask = self.take_asks(best_ask1, best_ask_amount1, mul, self.asset1)
            if not ask:
                continue
            bids = {}
            for product in self.asset2:
                desired_quant = round(self.regression[self.asset1][product] * mul)
                bid = self.take_asks(best_bid2[product], best_bid_amount2[product], desired_quant, product)
                if not (bid and desired_quant):
                    bids = {}
                    break
                bids[product] = bid
            if not bids:
                continue

            result[self.asset1] = ask
            for product in self.asset2:
                result[product] = bids[product]
            break

    if neutral:
        self.to_liquidate[self.asset1] = True
        for x in self.asset2:
            self.to_liquidate[x] = True

        result[self.asset1] = self.liquidate(best_bid1, best_bid_amount1, best_ask1, best_ask_amount1, self.asset1)
        for product in self.asset2:
            result[product] = self.liquidate(best_bid2[product], best_bid_amount2[product], best_ask2[product], best_ask_amount2[product], product)

    return result