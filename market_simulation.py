#!/usr/bin/env python

__author__ = 'Fred Flores'
__version__ = '0.0.1'
__date__ = '2020-05-09'
__email__ = 'fredflorescfa@gmail.com'

"""Simulate stock price volatility using a simple rules-based heterogeneous agent-based model."""

from cellular_automata import rules_2d
from cellular_automata import seeds_2d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def append_borders(x):
    # For agents on the edge of the matrix, connected with neighbors on the opposite end of the matrix.
    # This effectively creates an infinite lattice where every agent has a complete neighborhood.

    left_col = x[:, :1]
    right_col = x[:, -1:]
    x = np.append(right_col, x, axis=1)
    x = np.append(x, left_col, axis=1)

    top_row = x[:1, :]
    bot_row = x[-1:, :]
    x = np.append(bot_row, x, axis=0)
    x = np.append(x, top_row, axis=0)

    return x


def remove_borders(x):
    return x[1:-1, 1:-1]


def neighborhood(x, i, j, radius=1, method='moore'):
    """Return a subset of array x centered at (i, j) and radius = integer > 0"""
    assert isinstance(x, np.ndarray), 'Invalid numpy array x.'
    assert x.shape[0] == x.shape[1], 'x is not a square matrix.'
    for idx in [i, j]:
        assert isinstance(i, (int, np.integer)), 'Invalid integer in (i, j).'
    assert isinstance(radius, (int, np.integer)), 'Invalid integer for radius=.'
    assert radius <= np.min(x.shape), 'Radius is too large for given matrix x.'
    assert method in ['moore', 'von_neumann'], 'Invalid method. Enter \'moore\' or \'von_neumann\'.'

    hood = x[i - radius: i + radius + 1: 1, j - radius: j + radius + 1: 1]

    if method == 'moore':
        return hood

    if method == 'von_neumann':
        cross = np.empty(shape=hood.shape)
        cross[:] = np.nan
        cross[:, radius] = hood[:, radius]
        cross[radius, :] = hood[radius, :]

        return cross


class Market(object):

    def __init__(self, size, steps, pct_chartists, price_initial,
                 price_target, price_deviation, price_sensitivity):

        self.shape = [size, size]
        self.steps = steps
        self.pct_chartists = pct_chartists
        self.price_initial = price_initial
        self.price_target = np.random.normal(loc=price_target, scale=price_deviation,
                                             size=self.shape)
        self.price_sensitivity = price_sensitivity
        self.market = self.initialize_traders()

    def initialize_traders(self):
        trader_type = np.empty(shape=self.shape)

        index = np.arange(0, self.shape[0])
        for i in index:
            for j in index:
                trader_threshold = np.random.uniform(0, 1, 1)
                if trader_threshold < self.pct_chartists:
                    trader_type[i, j] = 0  # chartists
                else:
                    trader_type[i, j] = 1  # fundamentalist

        return trader_type

    def trade_fundamentalists(self, price_current):
        """Compute difference between perceived fundamental value and current price."""

        return self.market * (self.price_target - price_current)

    def trade_chartists(self, fundamental_trades):

        trades = np.zeros(self.shape)
        fundamental_trades = append_borders(fundamental_trades)
        trader_type = append_borders(self.market)

        for i in np.arange(1, trader_type.shape[0]-1):
            for j in np.arange(1, trader_type.shape[0]-1):

                # Compute chartists' reaction to fundamentalists' trades from the previous step
                is_chartist = trader_type[i, j] == 0
                if is_chartist:
                    hood = neighborhood(fundamental_trades, i, j, method='moore')
                    trades[i-1, j-1] = np.sum(hood) / 8  # mimic avg trade in Moore neighborhood

        return trades

    def price_change(self, trades):
        return self.price_sensitivity * (np.sum(trades) / np.power(self.shape[0], 2))

    def simulate(self):

        # Initialize environment
        market = self.initialize_traders()  # fundamentalist or chartist
        trades_fu = self.trade_fundamentalists(self.price_initial)  # fundamentalist trades
        price_current = self.price_initial + self.price_change(trades_fu)

        # Containers for analytics
        P = [self.price_initial, price_current]  # time series of prices
        Q = [(0, 0, 0), (np.sum(trades_fu), np.sum(trades_fu), 0)]  # time series of trades

        for s in np.arange(1, self.steps+1):
            # Execute trades
            trades_ct = self.trade_chartists(trades_fu)
            trades_fu = self.trade_fundamentalists(price_current)

            # Estimate price change based on simple supply and demand
            price_current += self.price_change(np.sum(trades_fu + trades_ct))

            # Track changes
            P.append(price_current)
            Q.append([np.sum(trades_ct + trades_fu), np.sum(trades_ct), np.sum(trades_fu)])

        return P, Q / np.power(self.shape[0], 2)





