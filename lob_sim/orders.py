from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# sides
class OrderSide(Enum):
    BID = "bid"
    ASK = "ask"


# states
class OrderStatus(Enum):
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"


# types
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


# order
@dataclass
class Order:
    order_id: int
    side: OrderSide
    order_type: OrderType
    quantity: float
    timestamp: float
    price: Optional[float] = None
    status: OrderStatus = field(default=OrderStatus.OPEN)
    filled_quantity: float = field(default=0.0)

    @property
    def remaining_quantity(self) -> float:
        # unfilled
        return self.quantity - self.filled_quantity


# execution
@dataclass
class Trade:
    trade_id: int
    buy_order_id: int
    sell_order_id: int
    price: float
    quantity: float
    timestamp: float
