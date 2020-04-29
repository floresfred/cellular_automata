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

    def __init__(self, size=30, neighbors=4, code='942', seed_type='center'):

        assert isinstance(size, int), 'Invalid integer for \'size\''
        if np.mod(size, 2) == 0:
            size += 1
        self.size = size
        self.shape = (self.size, self.size)

        assert isinstance(neighbors, int), 'Invalid integer for \'neighborhood\''
        assert neighbors in [4, 8], 'Invalid neighborhood. 4 = Von Neumann, 8 = Moore'
        self.neighbors = neighbors

        assert code in rules_2d.rule.keys(), 'Invalid code'
        self.code = code
        self.update_cell = rules_2d.rule[code]

        assert seed_type in ['center', 'random'], 'Invalid seed type. Enter \'center\' or \'random\''
        self.seed_type = seed_type

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
        """Apply rule to grid"""

        y = np.copy(x)

        index = np.arange(1, x.shape[0] - 1)  # ignore the cells on the matrix edge
        for i in index:
            for j in index:
                neighborhood = x[i - 1:i + 2:1, j - 1:j + 2:1]  # 3x3 sub matrix centered at i, j
                y[i, j] = self.update_cell(neighborhood)

        return y

    def grid_frame(self, steps):
        """ Compute the grid at step"""

        x = self.seed()

        for n in np.arange(0, steps):
            x = self.update_grid(x)

        color_map = matplotlib.colors.ListedColormap(['white', 'black'])
        img = plt.imshow(x, interpolation='nearest', cmap=color_map)
        img.axes.grid(False)
        plt.show()

        return x

    def grid_animation(self, steps, figure_size=(12, 12), speed=100):

        assert steps < self.size, 'Number of steps exceeds size of matrix.'
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
            plt.title('Rule: ' + self.code + ' | Step: ' + str(counter), fontsize=14)

            im.set_array(np.copy(x))

            return im,  # why is this comma necessary?

        ani = animation.FuncAnimation(fig, update_figure, frames=steps,
                                      interval=speed, blit=False, repeat=False)

        plt.show()













