from __future__ import annotations
import numpy as np
import mlmath


class Layer:
    def __init__(
        self,
        nodes: int,
        previous: Layer = None,
        next: Layer = None,
        weights: np.ndarray = None,
        biases: np.ndarray = None,
    ):
        """Initializes a layer for neural networks
        Parameters:
            nodes (int): The amount nodes in the layer
            previous (Layer): The layer preceding this layer. None if this is the first layer
            next (Layer):
        """
        self.nodeCount = nodes
        self.previous = previous
        self.next = next
        self.outputValues = np.empty(0)
        self.inputValues = np.empty(0)

        self.deltas = None

        if previous != None:
            if weights != None:
                self.weights = weights
            else:
                self.weights = np.empty((nodes, previous.nodeCount))

            if biases != None:
                self.biases = biases
            else:
                self.biases = np.empty(nodes)

    def calculateValues(self, data: np.ndarray, forceRecalculate=True) -> np.ndarray:
        """
        Calculate the values of the layer, and set them on the layer object

        Returns
        -------
        The calculated values

        """
        if not forceRecalculate and self.outputValues.size != 0:
            return self.outputValues
        if self.previous == None:
            self.inputValues = data
            self.outputValues = mlmath.sigmoid(self.inputValues)
            return self.outputValues

        prevValues = self.previous.calculateValues(data)
        self.inputValues = np.dot(self.weights, prevValues) + self.biases
        self.outputValues = mlmath.sigmoid(self.inputValues)

        return self.outputValues

    def cost(self, target: np.ndarray) -> float:
        if self.nodeCount != target.shape[0]:
            raise Exception("Target not same size as output layer")
        s = 0

        for i in range(self.nodeCount):
            s += (target[i] - self.outputValues[i]) ** 2
        return s / 2

    def calculateDeltas(self, target: np.ndarray, forceRecalculate=True) -> np.ndarray:
        if not forceRecalculate and self.deltas != None:
            return self.deltas
        self.deltas = np.empty(self.nodeCount)
        if self.next == None:  # If there is no next layer, ie. this is the output layer
            self.deltas = (
                self.outputValues
                * (1 - self.outputValues)
                * (target - self.outputValues)
            )
        else:
            frontDeltas = self.next.calculateDeltas(target, True)
            self.deltas = (
                self.outputValues
                * (1 - self.outputValues)
                * (np.dot(self.next.weights.transpose(), frontDeltas))
            )
        return self.deltas

    def __repr__(self):
        return str(self.__dict__)
