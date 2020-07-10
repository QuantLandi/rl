from typing import Dict, Union
from enum import IntEnum

import numpy as np
import pandas as pd


class Action(IntEnum):
    IDLE = 0
    BUY = 1
    SELL = 2


class Position(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = 2

    def __repr__(self):
        if self == FLAT:
            return "flat"
        elif self == LONG:
            return "long"
        else:
            return "short"


FLAT = Position.FLAT
LONG = Position.LONG
SHORT = Position.SHORT

IDLE = Action.IDLE
BUY = Action.BUY
SELL = Action.SELL


def pos2dir(position: Position) -> int:
    if position == LONG:
        return 1
    elif position == SHORT:
        return -1
    else:
        raise ValueError("direction cannot be computed on FLAT position")


def compute_pnl(
    position: Position,
    open_price: int,
    exit_price: int,
    instrument_multiplier: int,
    order_commission: float,
) -> float:
    pnl = (
        instrument_multiplier * pos2dir(position) * (exit_price - open_price)
        - 2 * order_commission
    )
    return pnl


def get_instrument_info(symbol: str) -> Dict[str, Union[int, float]]:
    if symbol != "BD#":
        raise ValueError("symbol not BD#")
    return {"multiplier": 1000, "order_commission": 1.11}


def compute_exit_price(position: Position, bid: int, ask: int) -> int:
    return bid if position == Position.LONG else ask
