#!/usr/bin/env python

__author__ = 'Fred Flores'
__version__ = '0.0.1'
__date__ = '2020-05-09'
__email__ = 'fredflorescfa@gmail.com'

"""Simulate stock price volatility using a simple rules-based heterogeneous agent-based model (HAM)."""

from cellular_automata import rules_2d
from cellular_automata import seeds_2d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def neighborhood(x, i, j, radius=1, method='moore'):
    """Return a square subset of matrix x centered at (i, j) and with radius = integer > 0
       if radius = 1, the "moore" method returns an 8 neighbor matrix (includes corners)
                      the "von_neumann" method excludes the corners by setting them to NaN."""

    assert isinstance(x, np.ndarray), 'Invalid numpy array x.'
    assert x.shape[0] == x.shape[1], 'x is not a square matrix.'
    for idx in [i, j]:
        assert isinstance(i, (int, np.integer)), 'Invalid integer in (i, j).'
        assert idx in np.arange(radius, x.shape[0] + 1 - radius), 'Index and radius are incompatible.'
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
                 price_target, price_sensitivity,
                 demand_sensitivity_fu=0.5, demand_sensitivity_ct=0.5,
                 demand_threshold_ct=0, demand_gamma_ct=0.5, boundary_type='periodic'):

        self.shape = [size, size]
        self.steps = steps
        self.pct_chartists = pct_chartists
        self.price_initial = price_initial
        self.price_target = price_target
        self.price_sensitivity = price_sensitivity
        self.market = self.initialize_traders()

        self.demand_sensitivity_fu = demand_sensitivity_fu
        self.demand_sensitivity_ct = demand_sensitivity_ct

        self.demand_threshold_ct = demand_threshold_ct
        self.demand_gamma_ct = demand_gamma_ct

        self.boundary_type = boundary_type

    def append_borders(self, x):
        """Ensure complete neighborhood for each agent on the edges of the grid"""

        if self.boundary_type == 'fixed':
            zero_col = np.zeros(x[:, :1].shape).reshape(-1, 1)
            x = np.append(zero_col, x, axis=1)
            x = np.append(x, zero_col, axis=1)

            zero_row = np.zeros(x[:1, :].shape)
            x = np.append(zero_row, x, axis=0)
            x = np.append(x, zero_row, axis=0)

        if self.boundary_type == 'periodic':
            left_col = x[:, :1]
            right_col = x[:, -1:]
            x = np.append(right_col, x, axis=1)
            x = np.append(x, left_col, axis=1)

            top_row = x[:1, :]
            bot_row = x[-1:, :]
            x = np.append(bot_row, x, axis=0)
            x = np.append(x, top_row, axis=0)

        if self.boundary_type == 'reflexive':
            left_col = x[:, :1]
            right_col = x[:, -1:]

            x = np.append(x, right_col, axis=1)
            x = np.append(left_col, x, axis=1)

            top_row = x[:1, :]
            bot_row = x[-1:, :]

            x = np.append(x, bot_row, axis=0)
            x = np.append(top_row, x, axis=0)

        return x

    def remove_borders(self, x):
        return x[1:-1, 1:-1]

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

        return self.market * (self.demand_sensitivity_fu * (self.price_target - price_current))

    def trade_chartists(self, fundamental_trades):

        trades = np.zeros(self.shape)
        fundamental_trades = self.append_borders(fundamental_trades)
        trader_type = self.append_borders(self.market)

        for i in np.arange(1, trader_type.shape[0]-1):
            for j in np.arange(1, trader_type.shape[0]-1):

                # Compute chartists' reaction to fundamentalists' trades from the previous step
                is_chartist = trader_type[i, j] == 0
                if is_chartist:
                    nbrs = neighborhood(fundamental_trades, i, j, radius=1, method='moore')
                    nbrs_sum = np.sum(nbrs) - self.demand_threshold_ct
                    trades[i-1, j-1] = self.demand_sensitivity_ct * np.tanh(self.demand_gamma_ct * nbrs_sum)
                    # print(trades[i-1, j-1])

        return trades

    def price_change(self, trades):
        return self.price_sensitivity * np.sum(trades)

    def simulate(self):

        # Initialize environment
        market = self.initialize_traders()  # fundamentalist or chartist
        trades_fu = self.trade_fundamentalists(self.price_initial)  # fundamentalist trades
        price_current = self.price_initial + self.price_change(trades_fu)

        # Containers for analytics
        P = [self.price_initial, price_current]  # time series of prices
        Q = [(0, 0, 0), (np.sum(trades_fu), np.sum(trades_fu), 0)]  # time series of trades
        V = [(0, 0, 0), (np.sum(np.abs(trades_fu)), np.sum(np.abs(trades_fu)), 0)]

        for s in np.arange(1, self.steps+1):
            # Execute trades
            trades_ct = self.trade_chartists(trades_fu)
            trades_fu = self.trade_fundamentalists(price_current)

            # Estimate price change based on simple supply and demand
            price_current += self.price_change(np.sum(trades_fu + trades_ct))

            # Track indices
            P.append(price_current)
            Q.append([np.sum(trades_fu + trades_ct),
                      np.sum(trades_fu), np.sum(trades_ct)])
            V.append([np.sum(np.abs(trades_fu) + np.abs(trades_ct)),
                      np.sum(np.abs(trades_fu)), np.sum(np.abs(trades_ct))])

        return P, Q, V





