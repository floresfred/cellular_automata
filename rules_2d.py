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


def rule_conway(x):

    current_state = int(x[1, 1])
    N = int(np.sum(x) - current_state)

    next_state = 0  # assume everyone dies

    # if (N < 2) & (current_state == 1):  # dies by underpopulation
    #     next_state = 0

    if (N in [2, 3]) & (current_state == 1):  # lives on
        next_state = 1

    # if (N > 3) & (current_state == 1):  # dies by overpopulation
    #     next_state = 0

    if (N == 3) & (current_state == 0):  # reproduction
        next_state = 1

    return next_state


rule = {'942': rule_942,
        'conway': rule_conway}


