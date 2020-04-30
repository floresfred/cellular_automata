#!/usr/bin/env python

__author__ = 'Fred Flores'
__version__ = '0.0.1'
__date__ = '2020-04-27'
__email__ = 'fredflorescfa@gmail.com'

import numpy as np


def center(shape):
    mid_point = int(shape[0]/2)
    x = np.zeros(shape)
    x[mid_point, mid_point] = 1
    return x


def random(shape):
    return np.random.randint(0, 2, shape)


def triomino_1(shape):
    mid_point = int(shape[0] / 2)
    x = np.zeros(shape)
    x[mid_point - 1, mid_point + 1] = 1
    x[mid_point, mid_point] = 1
    x[mid_point + 1, mid_point] = 1
    return x  # nothing; all dies


def triomino_2(shape):
    mid_point = int(shape[0] / 2)
    x = np.zeros(shape)
    x[mid_point - 1, mid_point] = 1
    x[mid_point, mid_point] = 1
    x[mid_point + 1, mid_point] = 1
    return x  # col -> row -> col repeats


def triomino_3(shape):
    mid_point = int(shape[0] / 2)
    x = np.zeros(shape)
    x[mid_point, mid_point + 1] = 1
    x[mid_point, mid_point] = 1
    x[mid_point + 1, mid_point] = 1
    return x  # stable 4 square


def triomino_4(shape):
    mid_point = int(shape[0] / 2)
    x = np.zeros(shape)
    x[mid_point - 1, mid_point + 1] = 1
    x[mid_point, mid_point] = 1
    x[mid_point + 1, mid_point - 1] = 1
    return x  # stable 4 square


def tetromino_4(shape):
    mid_point = int(shape[0] / 2)
    x = np.zeros(shape)
    x[mid_point, mid_point - 1] = 1
    x[mid_point - 1, mid_point] = 1
    x[mid_point, mid_point] = 1
    x[mid_point, mid_point + 1] = 1
    return x  # stable 4 square


def glider(shape):
    mid_point = shape[0] - 4
    x = np.zeros(shape)
    x[mid_point, mid_point] = 1
    x[mid_point, mid_point + 1] = 1
    x[mid_point, mid_point - 1] = 1
    x[mid_point + 1, mid_point - 1] = 1
    x[mid_point + 2, mid_point] = 1
    return x  # stable 4 square


initialize = {'center': center,
              'random': random,
              'triomino_1': triomino_1,
              'triomino_2': triomino_2,
              'triomino_3': triomino_3,
              'triomino_4': triomino_4,
              'tetromino_4': tetromino_4,
              'glider': glider}
