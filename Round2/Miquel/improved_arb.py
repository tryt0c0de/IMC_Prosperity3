from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import jsonpickle

from Logger import Logger

logger = Logger()

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"

###############################################################################
# 1) PARAMS: define for each product:
#    - For baskets: store the underlying coefficients to compute fair value
#    - For others: we can store a static "fair_value" or a function that updates it
#    - Also store market-making parameters (take_width, clear_width, etc.)
###############################################################################

PARAMS: Dict[str, Dict] = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,         # Example static fair value
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 30,
        "prevent_adverse": False,
        "adverse_volume": 0,
    },

    Product.KELP: {
        "fair_value": None,         # We'll compute in code using reversion logic, for example
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "soft_position_limit": 50,
        "prevent_adverse": True,
        "adverse_volume": 25,
        "reversion_beta": -0.6,
    },

    Product.SQUID_INK: {
        "fair_value": None,         # Will compute from best bid/ask mid
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 50,
        "prevent_adverse": False,
        "adverse_volume": 0,
    },

    # Picnic baskets: define coefficients of underlying for fair value
    Product.PICNIC_BASKET1: {
        "coeffs": {
            Product.CROISSANTS: 6,
            Product.JAMS: 3,
            Product.DJEMBES: 1,
        },
        "take_width": 5,         # Example widths for a basket
        "clear_width": 1,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 60,
        "prevent_adverse": False,
        "adverse_volume": 0,
    },
    Product.PICNIC_BASKET2: {
        "coeffs": {
            Product.CROISSANTS: 4,
            Product.JAMS: 2,
        },
        "take_width": 5,
        "clear_width": 1,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 100,
        "prevent_adverse": False,
        "adverse_volume": 0,
    },
}

###############################################################################
# 2) THE TRADER
###############################################################################

class Trader:
    def __init__(self, params=None):
        # Merge user-provided params with default PARAMS if desired:
        self.params = params if params else PARAMS
        
        # Hard position limits from competition (or your own logic):
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60
        }

    ###############################################################################
    # 2a)  Utility: Compute fair values for Baskets
    ###############################################################################
    def compute_basket_fair_value(
        self, 
        basket_product: str, 
        state: TradingState
    ) -> float:
        """
        Given PICNIC_BASKET1 or PICNIC_BASKET2, compute its "fair" price
        from the mid-prices of its underlying items (per the coefficients).
        Return None if we cannot compute (missing order book data).
        """
        coeffs = self.params[basket_product]["coeffs"]  # e.g. {CROISSANTS:6, JAMS:3, DJEMBES:1}
        total_value = 0.0

        for underlying_product, qty in coeffs.items():
            # If order book missing or no quotes, we skip
            if (underlying_product not in state.order_depths or 
                not state.order_depths[underlying_product].buy_orders or 
                not state.order_depths[underlying_product].sell_orders):
                return None
            
            od = state.order_depths[underlying_product]
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            mid_price = 0.5 * (best_bid + best_ask)
            total_value += qty * mid_price
        
        return total_value

    ###############################################################################
    # 2b)  Utility: Fair value for KELP (example: reversion logic)
    ###############################################################################
    def kelp_fair_value(self, order_depth: OrderDepth, traderObject: dict) -> float:
        """
        Example from your code:
        - We look for the best bid/ask that have volumes >= adverse_volume
        - Then compute a mid. Apply reversion logic from the last price, etc.
        """
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        # Filter out large volumes if we want to avoid them
        filtered_asks = [
            price for price in order_depth.sell_orders.keys()
            if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
        ]
        filtered_bids = [
            price for price in order_depth.buy_orders.keys()
            if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
        ]
        if filtered_asks and filtered_bids:
            mm_ask = min(filtered_asks)
            mm_bid = max(filtered_bids)
            mmmid_price = 0.5 * (mm_ask + mm_bid)
        else:
            # If we have no "big-volume" quotes, just default to normal mid
            mmmid_price = 0.5 * (best_ask + best_bid)

        last_price = traderObject.get("kelp_last_price", None)
        if last_price is not None:
            last_returns = (mmmid_price - last_price) / last_price
            beta = self.params[Product.KELP]["reversion_beta"]
            predicted_returns = last_returns * beta
            fair = mmmid_price + (mmmid_price * predicted_returns)
        else:
            fair = mmmid_price

        traderObject["kelp_last_price"] = mmmid_price
        return fair

    ###############################################################################
    # 2c)  Utility: Fair value from best bid/ask for SQUID_INK
    ###############################################################################
    def fair_value_squid_ink(self, order_depth: OrderDepth):
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return 0.5 * (best_ask + best_bid)
        elif order_depth.sell_orders:
            return float(min(order_depth.sell_orders.keys()))
        elif order_depth.buy_orders:
            return float(max(order_depth.buy_orders.keys()))
        return None

    ###############################################################################
    # 2d) TAKER Logic: If best_ask < (fair_value - take_width), buy; 
    #                  If best_bid > (fair_value + take_width), sell
    ###############################################################################
    def take_best_orders(
        self,
        product: str,
        fair_value: float,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        """
        Check the best ask & best bid vs. fair_value ± take_width.
        If there's an edge, we 'take' that liquidity.
        Return how many units we ended up buying or selling.
        """
        position_limit = self.LIMIT[product]

        # TAKE the best ask if it’s well below our fair_value
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = -order_depth.sell_orders[best_ask]  # since sells are negative
            if not prevent_adverse or abs(best_ask_volume) <= adverse_volume:
                # Condition to buy from the ask
                if best_ask < (fair_value - take_width):
                    # Max we can buy before hitting position limit
                    can_buy = position_limit - (position + buy_order_volume)
                    quantity = min(best_ask_volume, can_buy)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        # Adjust local orderbook
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        # TAKE the best bid if it’s well above our fair_value
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]  # positive
            if not prevent_adverse or abs(best_bid_volume) <= adverse_volume:
                # Condition to sell into the bid
                if best_bid > (fair_value + take_width):
                    can_sell = position_limit + (position - sell_order_volume)
                    quantity = min(best_bid_volume, can_sell)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        # Adjust local orderbook
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    ###############################################################################
    # 2e) CLEAR Logic: Offload position if we see extremely favorable quotes
    ###############################################################################
    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        """
        If we’re net long, we might look for any buy orders above (fair_value + width)
        to sell into. If we’re net short, we might look for any sells below (fair_value - width)
        to buy from. (You can tweak logic as needed.)
        """
        pos_after_take = position + buy_order_volume - sell_order_volume
        position_limit = self.LIMIT[product]

        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        # If we’re net long, see if we can sell above (fair_value + width)
        if pos_after_take > 0 and order_depth.buy_orders:
            # sum volumes from all buy orders >= fair_for_ask
            clear_qty = sum(
                vol for px, vol in order_depth.buy_orders.items() if px >= fair_for_ask
            )
            # We'll unload at most pos_after_take
            unload_amount = min(clear_qty, pos_after_take)
            # Also be sure not to exceed how many we can still sell
            can_sell = position_limit + (position - sell_order_volume)
            to_sell = min(unload_amount, can_sell)
            if to_sell > 0:
                orders.append(Order(product, fair_for_ask, -to_sell))
                sell_order_volume += to_sell

        # If we’re net short, see if we can buy below (fair_value - width)
        if pos_after_take < 0 and order_depth.sell_orders:
            clear_qty = sum(
                -vol for px, vol in order_depth.sell_orders.items() if px <= fair_for_bid
            )
            # We'll buy at most abs(pos_after_take)
            buy_amount = min(clear_qty, abs(pos_after_take))
            can_buy = position_limit - (position + buy_order_volume)
            to_buy = min(buy_amount, can_buy)
            if to_buy > 0:
                orders.append(Order(product, fair_for_bid, to_buy))
                buy_order_volume += to_buy

        return buy_order_volume, sell_order_volume

    ###############################################################################
    # 2f) MAKE Logic: Place passive bids & asks around “fair_value”
    #    - add small offsets or try to join/penny existing orders
    ###############################################################################
    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []

        # Identify potential places to "join" or "penny" existing orders
        # For example, if the best ask is > fair_value+disregard_edge, we might place an ask < that best ask to capture a fill.
        asks_above_fair = [
            px for px in order_depth.sell_orders.keys()
            if px > fair_value + disregard_edge
        ]
        bids_below_fair = [
            px for px in order_depth.buy_orders.keys()
            if px < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        # Start with a default offset from fair
        ask_price = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            gap_ask = best_ask_above_fair - fair_value
            if gap_ask <= join_edge:
                # Join their price
                ask_price = best_ask_above_fair
            else:
                # Undercut by 1 if not too close
                ask_price = best_ask_above_fair - 1

        bid_price = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            gap_bid = fair_value - best_bid_below_fair
            if gap_bid <= join_edge:
                # Join their price
                bid_price = best_bid_below_fair
            else:
                # Outbid by 1
                bid_price = best_bid_below_fair + 1

        # Adjust if we have a large position
        if manage_position:
            if position > soft_position_limit:
                # If we’re heavily long, we might want to place a more aggressive ask to reduce inventory
                ask_price = max(ask_price - 1, 1)
            elif position < -soft_position_limit:
                # If we’re heavily short
                bid_price += 1

        # Now place the market-making orders subject to total position limit
        # "market_make" is basically "bid at bid_price, ask at ask_price"
        max_can_buy = self.LIMIT[product] - (position + buy_order_volume)
        max_can_sell = self.LIMIT[product] + (position - sell_order_volume)

        # Place a buy (bid) if we can
        if max_can_buy > 0 and bid_price > 0:
            orders.append(Order(product, bid_price, max_can_buy))

        # Place a sell (ask) if we can
        if max_can_sell > 0 and ask_price > 0:
            orders.append(Order(product, ask_price, -max_can_sell))

        return orders, buy_order_volume, sell_order_volume

    ###############################################################################
    # 3) MAIN "run" ENTRY POINT
    ###############################################################################
    def run(self, state: TradingState):
        # We may keep some persistent data in traderData (like last KELP price)
        traderObject = {}
        if state.traderData:
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except:
                traderObject = {}

        result = {}

        ###########################################################################
        # Handle each product
        ###########################################################################
        for product in self.params:
            if product not in state.order_depths:
                continue

            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            buy_order_volume = 0
            sell_order_volume = 0

            # 1) Determine fair value
            if product == Product.KELP:
                fair_value = self.kelp_fair_value(order_depth, traderObject)
                continue
            elif product == Product.SQUID_INK:
                fair_value = self.fair_value_squid_ink(order_depth)
                continue
            elif product == Product.PICNIC_BASKET1 or product == Product.PICNIC_BASKET2:
                fair_value = self.compute_basket_fair_value(product, state)
            else:
                # e.g. RAINFOREST_RESIN or static fair_value
                fair_value = self.params[product].get("fair_value", None)
                continue

            if fair_value is None:
                # If we can't compute a fair value, skip
                continue

            # 2) Taker logic (take out-of-line quotes)
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product=product,
                fair_value=fair_value,
                take_width=self.params[product]["take_width"],
                orders=[],
                order_depth=order_depth,
                position=position,
                buy_order_volume=buy_order_volume,
                sell_order_volume=sell_order_volume,
                prevent_adverse=self.params[product].get("prevent_adverse", False),
                adverse_volume=self.params[product].get("adverse_volume", 0),
            )
            taker_orders = []
            # Because we manipulated 'order_depth' in place, we might want to store orders
            # in the function. Let's re-run the function but have it produce the final Orders.
            # Or we can do the same logic inline. For clarity, let's just do it inline:

            taker_orders_list: List[Order] = []
            # Re-run in a small snippet to "materialize" the orders.
            # In practice, you might store them from inside the function. 
            # We'll do it inline for demonstration:
            # We'll just replicate the checks quickly:

            # Re-check best_ask
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = -order_depth.sell_orders[best_ask]
                if not self.params[product].get("prevent_adverse", False) or abs(best_ask_volume) <= self.params[product].get("adverse_volume", 0):
                    if best_ask < (fair_value - self.params[product]["take_width"]):
                        can_buy = self.LIMIT[product] - (position + buy_order_volume)
                        quantity = min(best_ask_volume, can_buy)
                        if quantity > 0:
                            taker_orders_list.append(Order(product, best_ask, quantity))

            # Re-check best_bid
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                if not self.params[product].get("prevent_adverse", False) or abs(best_bid_volume) <= self.params[product].get("adverse_volume", 0):
                    if best_bid > (fair_value + self.params[product]["take_width"]):
                        can_sell = self.LIMIT[product] + (position - sell_order_volume)
                        quantity = min(best_bid_volume, can_sell)
                        if quantity > 0:
                            taker_orders_list.append(Order(product, best_bid, -quantity))

            # 3) Clear logic
            clear_orders_list: List[Order] = []
            buy_order_volume_cleared, sell_order_volume_cleared = self.clear_position_order(
                product=product,
                fair_value=fair_value,
                width=self.params[product]["clear_width"],
                orders=clear_orders_list,
                order_depth=order_depth,
                position=position,
                buy_order_volume=buy_order_volume,
                sell_order_volume=sell_order_volume
            )

            # 4) Make logic
            make_orders_list, _, _ = self.make_orders(
                product=product,
                order_depth=order_depth,
                fair_value=fair_value,
                position=position,
                buy_order_volume=buy_order_volume_cleared,
                sell_order_volume=sell_order_volume_cleared,
                disregard_edge=self.params[product]["disregard_edge"],
                join_edge=self.params[product]["join_edge"],
                default_edge=self.params[product]["default_edge"],
                manage_position=True,
                soft_position_limit=self.params[product]["soft_position_limit"]
            )

            # Combine all orders for this product
            final_orders = taker_orders_list + clear_orders_list + make_orders_list
            if final_orders:
                result[product] = final_orders

        # Example usage of conversions, or just set to 0
        conversions = 0
        # Save any updates to traderObject
        traderData = jsonpickle.encode(traderObject)

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
