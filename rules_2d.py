#!/usr/bin/env python

__author__ = 'Fred Flores'
__version__ = '0.0.1'
__date__ = '2020-04-27'
__email__ = 'fredflorescfa@gmail.com'

import numpy as np


def rule_942(x):
    current_state = x[1, 1]
    N = x[0, 1] + x[1, 2] + x[2, 1] + x[1, 0]

    if N in [1, 4]:
        next_state = 1
    else:
        next_state = current_state

    return next_state


def rule_110(x):
    current_state = int(x[1, 1])
    N = int(np.sum(x) - current_state)

    next_state = 0  # assume everyone dies

    if (N in [2, 3]) & (current_state == 1):  # lives on
        next_state = 1

    if (N == 3) & (current_state == 0):  # reproduction
        next_state = 1

    return next_state


def rule_746(x):
    current_state = int(x[1, 1])
    N = int(np.sum(x) - current_state)

    next_state = current_state

    if N == 3:  # reproduction
        next_state = 1

    if N > 4:  # reproduction
        next_state = 0

    return next_state


def rule_976(x):
    N = int(np.sum(x))

    if N == 4:  # reproduction
        next_state = 1

    if N == 5:  # reproduction
        next_state = 0

    if N < 4:  # reproduction
        next_state = 0

    if N > 5:  # reproduction
        next_state = 1

    return next_state


def market(x):
    current_state = int(x[1, 1])
    N = int(np.sum(x) - current_state)

    # Value Momentum Trader
    if (N <= 5) & (current_state == 1):  # buy
        next_state = 1

    if (N >= 6) & (current_state == 1):  # sell
        next_state = 0

    # Growth Trader
    if (N <= 2) & (current_state == 0):  # sell
        next_state = 0
    if (N >= 3) & (current_state == 0):  # buy
        next_state = 1

    return next_state


rule = {'942': (rule_942, 'Rule 942'),
        '110': (rule_110, 'Rule 110 Conway\'s Game of Life'),
        '746': (rule_746, 'Rule 746 Circular Pattern of Growth'),
        '976': (rule_976, 'Rule 976 Smooth Regional Boundaries'),
        'market': (market, 'Market')}


