#!/usr/bin/env python

__author__ = 'Fred Flores'
__version__ = '0.0.1'
__date__ = '2020-04-27'
__email__ = 'fredflorescfa@gmail.com'

from cellular_automata import rules_2d
from cellular_automata import seeds_2d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class BinaryState(object):

    def __init__(self, size=30, neighbors=4, code='942', seed_type='center', boundary_type='fixed'):

        assert isinstance(size, int), 'Invalid integer for \'size\''
        if np.mod(size, 2) == 0:
            size += 1  # give the matrix a true center cell
        self.size = size
        self.shape = (self.size, self.size)

        assert isinstance(neighbors, int), 'Invalid integer for \'neighborhood\''
        assert neighbors in [4, 8], 'Invalid neighborhood. Enter 4 = Von Neumann or 8 = Moore'
        self.neighbors = neighbors

        assert code in rules_2d.rule.keys(), 'Invalid code'
        self.code = code
        self.update_cell = rules_2d.rule[code][0]  # implementation of active cell update
        self.title = rules_2d.rule[code][1]

        assert boundary_type in ['periodic', 'reflexive', 'fixed'], \
            'Invalid border type, Enter \'periodic\' or \'reflexive\' or \'fixed\''
        self.boundary_type = boundary_type

        self.seed = seeds_2d.initialize[seed_type](self.shape)

    def append_boundary(self, x):
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

    def update_grid(self, x):
        """Update every cell in the grid. This is a single step in the evolution.
           Compute number in each state and number of state changes."""

        # Append boundary rows and columns to matrix
        x = self.append_boundary(x)  # the boundary is recomputed at each step
        y = np.copy(x)

        # For each cell within boundary, compute state according to rules.
        chg_0_1 = 0  # the number of cells that changed from state 0 to state 1
        chg_1_0 = 0  # the number of cells that changes from state 1 to state 0
        chg_none = 0  # the number of cells that did not change
        index = np.arange(1, x.shape[0] - 1)
        for i in index:
            for j in index:
                neighborhood = x[i - 1:i + 2:1, j - 1:j + 2:1]  # 3x3 sub matrix centered at i, j
                y[i, j] = self.update_cell(neighborhood)
                change = int(y[i, j] - x[i, j])
                if change == -1:
                    chg_1_0 += 1
                if change == 0:
                    chg_none += 1
                if change == 1:
                    chg_0_1 += 1

        # Compute statistics excluding boundary
        total = np.power(x[1:-1:1, 1:-1:1].shape[0] - 1, 2)
        start_1 = np.sum(x[1:-1:1, 1:-1:1])
        end_1 = np.sum(y[1:-1:1, 1:-1:1])
        stats = [total, start_1, end_1, chg_1_0, chg_none, chg_0_1]

        return y[1:-1:1, 1:-1:1], stats  # remove the boundary

    def grid_frame(self, steps, figure_size=(12, 12)):
        """ Compute the final grid at given number of steps"""

        x = self.seed
        counts = []
        for n in np.arange(0, steps):
            x, stats = self.update_grid(x)
            counts.append(stats)

        counts = np.array(counts)

        fig = plt.figure(figsize=figure_size)
        color_map = matplotlib.colors.ListedColormap(['white', 'black'])
        img = plt.imshow(x, interpolation='nearest', cmap=color_map)
        img.axes.grid(False)
        plt.show()

        return x, counts

    def grid_animation(self, steps, figure_size=(12, 12), speed=100):
        """Display a step by step animation of the cellular automata rule. """

        steps -= 1
        x = self.seed

        fig = plt.figure(figsize=figure_size)
        color_map = matplotlib.colors.ListedColormap(['white', 'black'])
        im = plt.imshow(x[1:-1:1,1:-1:1], interpolation='nearest', cmap=color_map, animated=True)
        counter = 0

        def update_figure(*args):
            nonlocal x, counter, fig

            counter += 1
            x, stats = self.update_grid(x)
            plt.title(self.title + ' | Step: ' + str(counter), fontsize=14)

            im.set_array(x[1:-1:1,1:-1:1])

            return im,  # why is this comma necessary?

        ani = animation.FuncAnimation(fig, update_figure, frames=steps,
                                      interval=speed, blit=False, repeat=False)

        plt.show()














