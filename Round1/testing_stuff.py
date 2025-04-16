import math
import jsonpickle
import numpy as np
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
from Logger import Logger
logger = Logger()
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        # Par de parámetros antiguos, algunos no se usarán directamente:
        "fair_value": 10000,           # Se deja de referencia (no se usará fijo)
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 20,
        # Parámetros Bollinger (sugeridos):
        "boll_window": 5000,          # Tamaño de ventana de precios
        "z_high": 2.0,                # Umbral de z-score para sobrecompra
        "z_low": -2.0                 # Umbral de z-score para sobreventa
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 25,
        "reversion_beta": -0.229/2.5, # Reversión AR para KELP
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 10,
    }
}

class Trader:
    def __init__(self, params=None):
        # Si no se pasan parámetros, usar los predeterminados
        self.params = PARAMS if params is None else params

        # Límites de posición
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50
        }

    ###########################################################################
    #                              MÉTODOS UTILITARIOS                         #
    ###########################################################################
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
        Toma liquidez en caso de que el mejor bid/ask esté muy alejado
        de nuestro fair_value ± take_width. 
        """
        position_limit = self.LIMIT[product]

        # Si hay órdenes de venta (best ask)
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]  # volumen es negativo

            # Chequeo de si queremos o no evitar órdenes grandes (adverse)
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                # Si best_ask está por debajo de fair_value - take_width,
                # conviene comprar (está muy barato)
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        # Ajustamos volúmenes y el OrderDepth
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        # Si hay órdenes de compra (best bid)
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                # Si best_bid está por encima de fair_value + take_width,
                # conviene vender (está demasiado caro)
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        """
        Coloca órdenes en el bid y ask de acuerdo a la posición y límites,
        para hacer market making pasivo.
        """
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bid, buy_quantity))

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, ask, -sell_quantity))

        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        """
        Intenta cerrar o reducir posición si está del lado equivocado,
        colocando órdenes cerca del fair_value (± width).
        """
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        # Si estamos largos (position_after_take > 0), vendemos parte para reducir
        if position_after_take > 0:
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        # Si estamos cortos (position_after_take < 0), compramos para reducir
        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    ###########################################################################
    #                   FUNCIÓN DE CÁLCULO BOLLINGER (NUEVA)                  #
    ###########################################################################
    def compute_bollinger_fair_value(
        self,
        product: str,
        order_depth: OrderDepth,
        traderObject: Dict,
        window_size: int,
        z_high: float,
        z_low: float
    ) -> float:
        """
        Calcula el fair_value usando Bollinger Bands:
         - Almacena el mid_price en una ventana (en traderObject).
         - Calcula media y std de la ventana.
         - Devuelve un fair_value adaptado según la señal de z-score.
        
        Si z-score > z_high, señal de venta (precio muy alto); 
        si z-score < z_low, señal de compra.
        Caso contrario, fair_value = mid_price.
        """
        # 1) obtener best_bid y best_ask
        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            # sin datos suficientes, retornar valor anterior o un default
            return 10000  # fallback

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # 2) Guardar en traderObject la ventana de midprices
        if "bollinger_windows" not in traderObject:
            traderObject["bollinger_windows"] = {}
        if product not in traderObject["bollinger_windows"]:
            traderObject["bollinger_windows"][product] = []

        window = traderObject["bollinger_windows"][product]
        window.append(mid_price)
        if len(window) > window_size:
            window.pop(0)

        # 3) Cálculo de media y std
        mean_price = np.mean(window)
        std_price = np.std(window) if len(window) > 1 else 1e-9

        # 4) z-score actual
        z_score = (mid_price - mean_price) / std_price

        # 5) Ajuste del fair_value
        #    - si z_score muy alto => esperamos bajada => fair_value algo menor
        #    - si z_score muy bajo => esperamos subida => fair_value algo mayor
        if z_score > z_high:
            # Precios sobrecomprados -> vendemos
            # Retornar fair_value ligeramente por debajo del mid actual
            # para impulsar a 'take_orders' a vender
            fair_value = mean_price  # o (mid_price - algo)
        elif z_score < z_low:
            # Precios sobrevendidos -> compramos
            # Retornar fair_value por encima del mid actual
            fair_value = mean_price
        else:
            # en rango normal => fair_value = mid_price (neutral)
            fair_value = mid_price

        # Opcional: También se puede escalar la diferencia si se quiere
        # más agresividad según la magnitud de z_score.

        return fair_value

    ###########################################################################
    #                  FUNCIONES DE KELP / SQUID_INK EXISTENTES               #
    ###########################################################################
    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        """
        Lógica de mean reversion simple, ya presente en tu código (para KELP).
        """
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None

            if mm_ask is None or mm_bid is None:
                if traderObject.get("kelp_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) is not None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.params[Product.KELP]["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price

            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def fair_value_squid_ink(self, order_depth: OrderDepth):
        """
        Lógica de fair_value básica para SQUID_INK.
        """
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        elif order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        elif order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        return None

    ###########################################################################
    #        FUNCIONES GENERALES PARA CREAR / TOMAR / LIMPIAR ÓRDENES         #
    ###########################################################################
    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

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
        """
        Coloca órdenes pasivas alrededor de un fair_value estimado,
        pennying niveles si están a más de disregard_edge de dicho fair_value,
        o uniéndose si están cerca (<= join_edge).
        Si manage_position=True, ajustamos un poco los niveles según la posición.
        """
        orders: List[Order] = []
        asks_above_fair = [
            price for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            # Ajustar un tick según posición
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product, orders, bid, ask, position,
            buy_order_volume, sell_order_volume
        )

        return orders, buy_order_volume, sell_order_volume

    ###########################################################################
    #                           LOOP PRINCIPAL (run)                          #
    ###########################################################################
    def run(self, state: TradingState):
        """
        Punto de entrada: en cada iteración se llama a run() con el estado actual.
        Retorna diccionario {product: [orders]} y (conversions, traderData).
        """
        # Recuperamos info previa si existe
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        #######################################################################
        # 1) RAINFOREST_RESIN con Bollinger Bands
        #######################################################################
        if (Product.RAINFOREST_RESIN in self.params and
            Product.RAINFOREST_RESIN in state.order_depths):

            order_depth_rr = state.order_depths[Product.RAINFOREST_RESIN]
            position_rr = state.position.get(Product.RAINFOREST_RESIN, 0)

            # Calculamos el fair_value basado en Bollinger
            boll_win = self.params[Product.RAINFOREST_RESIN]["boll_window"]
            z_hi = self.params[Product.RAINFOREST_RESIN]["z_high"]
            z_lo = self.params[Product.RAINFOREST_RESIN]["z_low"]

            resin_fair_value = self.compute_bollinger_fair_value(
                Product.RAINFOREST_RESIN,
                order_depth_rr,
                traderObject,
                boll_win,
                z_hi,
                z_lo
            )

            # 1.1) Tomamos órdenes si están fuera de rango
            rr_take_orders, buy_vol, sell_vol = self.take_orders(
                Product.RAINFOREST_RESIN,
                order_depth_rr,
                resin_fair_value,
                self.params[Product.RAINFOREST_RESIN]["take_width"],
                position_rr
            )

            # 1.2) Intentamos limpiar posición si estamos demasiado largos/cortos
            rr_clear_orders, buy_vol, sell_vol = self.clear_orders(
                Product.RAINFOREST_RESIN,
                order_depth_rr,
                resin_fair_value,
                self.params[Product.RAINFOREST_RESIN]["clear_width"],
                position_rr,
                buy_vol,
                sell_vol
            )

            # 1.3) Market making pasivo alrededor del fair_value de Bollinger
            rr_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                order_depth_rr,
                resin_fair_value,
                position_rr,
                buy_vol,
                sell_vol,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,  # manage_position
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"]
            )

            # Unimos todas las órdenes
            result[Product.RAINFOREST_RESIN] = rr_take_orders + rr_clear_orders + rr_make_orders

        #######################################################################
        # 2) KELP (lógica anterior de mean reversion con reversion_beta)
        #######################################################################
        if (Product.KELP in self.params and
            Product.KELP in state.order_depths):

            order_depth_kelp = state.order_depths[Product.KELP]
            position_kelp = state.position.get(Product.KELP, 0)

            kelp_fair_value = self.kelp_fair_value(order_depth_kelp, traderObject)
            if kelp_fair_value is not None:
                kelp_take_orders, buy_vol, sell_vol = self.take_orders(
                    Product.KELP,
                    order_depth_kelp,
                    kelp_fair_value,
                    self.params[Product.KELP]["take_width"],
                    position_kelp,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
                kelp_clear_orders, buy_vol, sell_vol = self.clear_orders(
                    Product.KELP,
                    order_depth_kelp,
                    kelp_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    position_kelp,
                    buy_vol,
                    sell_vol,
                )
                kelp_make_orders, _, _ = self.make_orders(
                    Product.KELP,
                    order_depth_kelp,
                    kelp_fair_value,
                    position_kelp,
                    buy_vol,
                    sell_vol,
                    self.params[Product.KELP]["disregard_edge"],
                    self.params[Product.KELP]["join_edge"],
                    self.params[Product.KELP]["default_edge"],
                )
                result[Product.KELP] = kelp_take_orders + kelp_clear_orders + kelp_make_orders

        #######################################################################
        # 3) SQUID_INK (lógica simple dada en tu código)
        #######################################################################
        if (Product.SQUID_INK in self.params and
            Product.SQUID_INK in state.order_depths):

            order_depth_si = state.order_depths[Product.SQUID_INK]
            position_si = state.position.get(Product.SQUID_INK, 0)
            fair_value_si = self.fair_value_squid_ink(order_depth_si)

            if fair_value_si is not None:
                take_orders_si, buy_vol, sell_vol = self.take_orders(
                    Product.SQUID_INK,
                    order_depth_si,
                    fair_value_si,
                    self.params[Product.SQUID_INK]["take_width"],
                    position_si
                )
                clear_orders_si, buy_vol, sell_vol = self.clear_orders(
                    Product.SQUID_INK,
                    order_depth_si,
                    fair_value_si,
                    self.params[Product.SQUID_INK]["clear_width"],
                    position_si,
                    buy_vol,
                    sell_vol,
                )
                make_orders_si, _, _ = self.make_orders(
                    Product.SQUID_INK,
                    order_depth_si,
                    fair_value_si,
                    position_si,
                    buy_vol,
                    sell_vol,
                    self.params[Product.SQUID_INK]["disregard_edge"],
                    self.params[Product.SQUID_INK]["join_edge"],
                    self.params[Product.SQUID_INK]["default_edge"],
                    True,  # manage_position
                    self.params[Product.SQUID_INK]["soft_position_limit"],
                )
                result[Product.SQUID_INK] = take_orders_si + clear_orders_si + make_orders_si

        # conversions no se usa aquí, por defecto 1
        conversions = 1
        # Guardamos la info en traderObject
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state,result,conversions,traderData)

        return result, conversions, traderData
