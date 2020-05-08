#!/usr/bin/env python

__author__ = 'Fred Flores'
__version__ = '0.0.1'
__date__ = '2020-05-02'
__email__ = 'fredflorescfa@gmail.com'

""" A replication of Qui, Kandhani, and Sloot's 2007 paper published in Physical Review E 75:
    Understanding the complex dynamics of stock markets through cellular automata."""

import numpy as np
import pandas as pd


default_parameters = {'size': 10,
                      'steps': 50,
                      'pct_imitators': 0.20,
                      'initial_price': 105,
                      'fundamental_value': 100,
                      'sensitivity_to_excess_demand': 0.5,
                      'sensitivity_to_news_imitators': 0.7,
                      'sensitivity_to_news_fundamentalists': 0.2,
                      'num_lookback_periods': 100,
                      'sensitivity_to_price_volatility': 20,
                      'volatility_threshold': 0.01,
                      'lower_bound_on_trading': 0.05}


def append_neighbors(x):
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


class Configuration(object):
    __slots__ = default_parameters.keys()

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MarketDynamicSimulator(Configuration):
    __slots__ = default_parameters.keys()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_agents(self):
        agent = np.empty((self.size, self.size))

        index = np.arange(0, self.size)
        for i in index:
            for j in index:
                agent_threshold = np.random.uniform(0, 1, 1)
                if agent_threshold < self.pct_imitators:
                    agent[i, j] = 0  # classify as imitator
                else:
                    agent[i, j] = 1  # classify as fundamentalist

        return agent

    def trade(self):

        # debug = pd.DataFrame(agent)
        # debug.to_csv(r'C:\Users\fredf\python_projects\rebuild_research\cellular_automata\debug.csv')

        N = self.size ** 2

        # Initialize price and track activity
        price_update = self.initial_price
        P = [price_update]  # price history
        Q = [0]  # net trades

        for s in np.arange(0, self.steps):
            agent = self.initialize_agents()
            trade_im = np.zeros((self.size, self.size))  # no trades, waits for fundamentalists to trade first

            

            x = append_neighbors(trade_fu)

            for i in np.arange(1, agent.shape[0] + 1):
                for j in np.arange(1, agent.shape[0] + 1):

                    # Compute imitators' reaction to fundamentalists' trades from the previous step
                    is_imitator = agent[i - 1, j - 1] == 0
                    if is_imitator:
                        neighborhood = x[i - 1:i + 2:1, j - 1:j + 2:1]
                        trade_im[i - 1, j - 1] = (np.sum(neighborhood) - x[i, j]) / 8  # the average trade in Moore neighborhood

            excess_demand = np.sum(trade_fu) + np.sum(trade_im)
            price_update = price_update + ((self.sensitivity_to_excess_demand * excess_demand) / N)

            Q.append(excess_demand)
            P.append(price_update)

            # Compute fundamentalists' trades
            agent = self.initialize_agents()
            agent = self.initialize_agents()
            trade_fu[:, :] = (self.fundamental_value - price_update) * agent

        return P, Q












