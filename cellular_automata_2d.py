#!/usr/bin/env python

__author__ = 'Fred Flores'
__version__ = '0.0.1'
__date__ = '2020-04-27'
__email__ = 'fredflorescfa@gmail.com'

from cellular_automata import rules_2d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class BinaryState(object):

    def __init__(self, size=30, neighbors=4, code='942', seed_type='center', boundary_type='fixed'):

        assert isinstance(size, int), 'Invalid integer for \'size\''
        if np.mod(size, 2) == 0:
            size += 1
        self.size = size
        self.shape = (self.size, self.size)

        assert isinstance(neighbors, int), 'Invalid integer for \'neighborhood\''
        assert neighbors in [4, 8], 'Invalid neighborhood. Enter 4 = Von Neumann or 8 = Moore'
        self.neighbors = neighbors

        assert code in rules_2d.rule.keys(), 'Invalid code'
        self.code = code
        self.update_cell = rules_2d.rule[code][0]  #  implementation of active cell update
        self.title = rules_2d.rule[code][1]

        assert seed_type in ['center', 'random'], 'Invalid seed type. Enter \'center\' or \'random\''
        self.seed_type = seed_type

        assert boundary_type in ['periodic', 'reflexive', 'fixed'], \
            'Invalid border type, Enter \'periodic\' or \'reflexive\' or \'fixed\''
        self.boundary_type = boundary_type

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

    def seed(self):
        """Return an initial matrix initialized with a random or fixed states."""

        if self.seed_type == 'center':
            center = int(self.size/2)
            x = np.zeros(self.shape)
            x[center, center] = 1

        if self.seed_type == 'random':
            x = np.random.randint(0, 2, self.shape)

        return x

    def update_grid(self, x):
        """Update every cell in the grid. This is a single step in the evolution."""

        x = self.append_boundary(x)  # the boundary is recomputed at each step
        y = np.copy(x)

        index = np.arange(1, x.shape[0] - 1)  # ignore the cells on the boundary
        for i in index:
            for j in index:
                neighborhood = x[i - 1:i + 2:1, j - 1:j + 2:1]  # 3x3 sub matrix centered at i, j
                y[i, j] = self.update_cell(neighborhood)

        return y[1:-1:1, 1:-1:1]  # remove the boundary

    def grid_frame(self, steps):
        """ Compute the final grid at given number of steps"""

        x = self.seed()

        for n in np.arange(0, steps):
            x = self.update_grid(x)

        color_map = matplotlib.colors.ListedColormap(['white', 'black'])
        img = plt.imshow(x, interpolation='nearest', cmap=color_map)
        img.axes.grid(False)
        plt.show()

        return x

    def grid_animation(self, steps, figure_size=(12, 12), speed=100):
        """Display a step by step animation of the cellular automata rule. """

        steps -= 1

        x = self.seed()

        fig = plt.figure(figsize=figure_size)
        color_map = matplotlib.colors.ListedColormap(['white', 'black'])
        im = plt.imshow(x, interpolation='nearest', cmap=color_map, animated=True)
        counter = 0

        def update_figure(*args):
            nonlocal x, counter, fig

            counter += 1
            x = self.update_grid(x)
            plt.title(self.title + ' | Step: ' + str(counter), fontsize=14)

            im.set_array(np.copy(x))

            return im,  # why is this comma necessary?

        ani = animation.FuncAnimation(fig, update_figure, frames=steps,
                                      interval=speed, blit=False, repeat=False)

        plt.show()













