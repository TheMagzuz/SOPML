import numpy as np
import mlmath
from image import Image


class Layer:
    def __init__(
        self,
        nodes: int,
        previous: Layer = None,
        weights: np.ndarray = None,
        biases: np.ndarray = None,
    ):
        self.nodeCount = nodes
        self.previous = previous
        self.outputValues = np.empty(0)
        self.inputValues = np.empty(0)

        if previous != None:
            if weights != None:
                self.weights = weights
            else:
                self.weights = np.empty((nodes, previous.nodeCount))

            if biases != None:
                self.biases = biases
            else:
                self.biases = np.empty(nodes)

    def calculateValues(self, image: Image, forceRecalculate=False) -> np.ndarray:
        """
        Calculate the values of the layer, and set them on the layer object

        Returns
        -------
        The calculated values

        """
        if forceRecalculate and self.outputValues.size != 0:
            return self.outputValues
        if self.previous == None:
            return image.data

        prevValues = self.previous.calculateValues(image)
        self.inputValues = np.dot(self.weights, prevValues) + self.biases
        self.outputValues = np.vectorize(mlmath.sigmoid)(self.inputValues)

        return self.outputValues

    def cost(self, target: np.ndarray) -> float:
        if self.nodeCount != target.shape[0]:
            raise Exception("Target not same size as output layer")
        if self.outputValues == None:
            raise Exception("Values not calculated")

        s = 0

        for i in range(self.nodeCount):
            s += ((target[i] - self.outputValues) ** 2) / 2
        return s
