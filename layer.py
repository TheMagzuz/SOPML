import numpy as np
import mlmath


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
        if previous != None:
            if weights != None:
                self.weights = weights
            else:
                self.weights = np.empty((nodes, previous.nodeCount))

            if biases != None:
                self.biases = biases
            else:
                self.biases = np.empty((nodes, previous.nodeCount))

    def calculateValues(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the values of the layer, and set them on the layer object

        Returns
        -------
        The calculated values

        """
        if self.previous == None:
            return data
        prevValues = self.previous.calculateValues(data)
        self.outputValues = np.empty((self.nodeCount))

        for nodeIndex in range(self.nodeCount):
            sum = 0
            for prevIndex in range(self.previous.nodeCount):
                sum += (
                    prevValues[prevIndex] * self.weights[prevIndex]
                    + self.biases[prevIndex]
                )
            self.inputValues[nodeIndex] = sum
            self.outputValues[nodeIndex] = mlmath.sigmoid(sum)
        return self.outputValues

    def cost(self, target: np.ndarray) -> float:
        if self.nodeCount != target.shape[0]:
            raise Exception("Target not same size as output layer")
        if self.outputValues == None:
            raise Exception("Values not calculated")

        s = 0

        for i in range(self.nodeCount):
            s += (target[i] - self.outputValues) ** 2
        s /= self.nodeCount
