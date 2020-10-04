import numpy as np


class Image:
    def __init__(self, data, label: int, width: int, height: int):
        self.label = label
        self.data = data
        self.displayData = np.empty(0)
        self.width = width
        self.height = height

    def Data2D(self, forceRecalculate=False):
        if not forceRecalculate and self.displayData.size != 0:
            return self.displayData

        self.displayData = np.empty((self.width, self.height))

        for row in range(self.height):
            for col in range(self.width):
                self.displayData[row, col] = self.data[row * self.width + col]

        return self.displayData

    def expectedVector(self):
        v = np.zeros(10)
        v[self.label] = 1
        return v
