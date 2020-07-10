import datetime
from typing import List, Tuple

import pytz
import numpy as np
import pandas as pd

from . import market
# import market
from .market import Action, Position, IDLE, BUY, SELL, FLAT, LONG, SHORT
# from market import Action, Position, IDLE, BUY, SELL, FLAT, LONG, SHORT


class Environment:
    pass


class BundEnvironment(Environment):
    def __init__(
        self,
        symbol: str,
        datestr: str,
        df: pd.DataFrame,
        market_features: List[str],
    ) -> None:
        self.symbol = symbol
        self.datestr = datestr
        self.df = df
        self.features = self.df.values.astype(np.float32)
        self.mkt_feature_idxs = np.argwhere(
            np.isin(np.array(self.df.columns), np.array(market_features))
        ).flatten()
        self.bid_idx = list(df.columns).index("bid")
        self.ask_idx = list(df.columns).index("ask")
        self.ts = df.index.values
        instrument_info = market.get_instrument_info(symbol)
        self.instrument_multiplier = instrument_info["multiplier"]
        self.instrument_order_commission = instrument_info["order_commission"]
        self.no_trading_before = 9
        self.reset()
        for market_feature in market_features:
            if market_feature not in self.df.columns:
                raise ValueError(f'unknown market feature "{market_feature}"')
        required_columns = ["bid", "ask"]
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(
                "missing important columns:\n"
                "required:\t{}\n"
                "found:\t{}".format(required_columns, list(self.df.columns))
            )
        self.market_features = market_features
        self.state_size = self.get_state_vector().shape[0]

    def reset(self):
        self.idx = 0
        self.day_cum_pnl = 0.0
        self.n_trades = 0
        self.position = FLAT
        self.open_price = None
        self.open_ts = None
        self.prev_unrealised_pnl = None

    def get_state_vector(self):
        state_vector = np.array([self.idx, self.position])
        return state_vector

    @property
    def position_one_hot_encoding(self) -> np.ndarray:
        encoding = np.zeros(3, dtype=np.float32)
        encoding[int(self.position)] = 1.0
        return encoding

    def do_step(self, action: Action) -> float:
        self.check_valid_action(action)
        if self.idx > self.df.shape[0]:
            raise StopIteration("reached the end of the data")
        reward = 0.0
        if self.position == FLAT:
            if action != IDLE:
                self.position = LONG if action == BUY else SHORT
                if action == BUY:
                    self.open_price = self.features[self.idx, self.ask_idx]
                else:
                    self.open_price = self.features[self.idx, self.bid_idx]
                self.open_ts = self.ts[self.idx]
                bid_price = self.features[self.idx, self.bid_idx]
                ask_price = self.features[self.idx, self.ask_idx]
                spread = (ask_price - bid_price) * self.instrument_multiplier
                reward = -self.instrument_order_commission - spread
                self.prev_unrealised_pnl = self.unrealised_pnl
        else:
            if action != IDLE:
                pnl = self.unrealised_pnl - self.prev_unrealised_pnl
                self.position = FLAT
                self.open_price = None
                self.open_ts = None
                self.day_cum_pnl += pnl
                self.n_trades += 1
                reward = pnl - self.instrument_order_commission
            else:
                unrealised_pnl = self.unrealised_pnl
                reward = unrealised_pnl - self.prev_unrealised_pnl
                self.prev_unrealised_pnl = unrealised_pnl
        self.idx += 1
        with open("reward.log", "a") as r_log:
            r_log.write(
                "bid: "+str(self.features[self.idx-1, self.bid_idx])+
                ", ask: "+str(self.features[self.idx-1, self.ask_idx])+
                ", action: "+str(action)+
                ", position: "+str(self.position)+
                ", reward: "+str(reward)+"\n"
            )
        return reward

    @property
    def unrealised_pnl(self) -> float:
        if self.position == FLAT:
            return None
        potential_exit_price = market.compute_exit_price(
            self.position,
            self.features[self.idx, self.bid_idx],
            self.features[self.idx, self.ask_idx],
        )
        unrealised_pnl = market.compute_pnl(
            self.position,
            self.open_price,
            potential_exit_price,
            self.instrument_multiplier,
            0,
        )
        return unrealised_pnl

    def get_valid_action_mask(self) -> Action:
        if self.position == FLAT:
            return np.array([True, True, True], dtype=np.bool)
        else:
#             if self.open_ts is not None and self.trade_duration < 60:
#                 return np.array([True, False, False], dtype=np.bool)
            if self.position == LONG:
#                if self.idx == self.df.shape[0] - 1:
#                    return np.array([False, False, True], dtype=np.bool)
                return np.array([True, False, True], dtype=np.bool)
            else:
#                if self.idx == self.df.shape[0] - 1:
#                    return np.array([False, True, False], dtype=np.bool)
                return np.array([True, True, False], dtype=np.bool)

    def check_valid_action(self, action: Action) -> None:
        current_hour = self.df.index[self.idx].hour
        if self.position == FLAT:
            return
        elif self.position == LONG and action == BUY:
            raise ValueError("invalid action: cannot BUY if already LONG")
        elif self.position == SHORT and action == SELL:
            raise ValueError("invalid action: cannot SELL if already SHORT")

    @property
    def trade_duration(self):
        if self.open_ts is None:
            return None
        return (self.ts[self.idx] - self.open_ts).astype(np.int64) / 1e9

    def terminal_state(self) -> bool:
        return self.idx == (self.df.shape[0] - 1)
