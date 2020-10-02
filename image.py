import numpy as np


class Image:
    def __init__(self, data, label: int, width: int, height: int):
        self.label = label
        self.data = data
        self.matrixData = np.empty(0)
        self.width = width
        self.height = height

    def Data2D(self, forceRecalculate=False):
        if not forceRecalculate and self.matrixData.size != 0:
            return self.matrixData

        self.matrixData = np.empty((self.width, self.height))

        for row in range(self.height):
            for col in range(self.width):
                self.matrixData[row, col] = self.data[row * self.width + col]

        return self.matrixData
