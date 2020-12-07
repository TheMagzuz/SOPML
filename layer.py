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

    def calculateValues(self, data: np.ndarray, forceRecalculate=False) -> np.ndarray:
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
            self.outputValues = np.vectorize(mlmath.sigmoid)(self.inputValues)
            return self.outputValues

        prevValues = self.previous.calculateValues(data)
        self.inputValues = np.dot(self.weights, prevValues) + self.biases
        self.outputValues = np.vectorize(mlmath.sigmoid)(self.inputValues)

        return self.outputValues

    def cost(self, target: np.ndarray) -> float:
        if self.nodeCount != target.shape[0]:
            raise Exception("Target not same size as output layer")
        s = 0

        for i in range(self.nodeCount):
            s += ((target[i] - self.outputValues[i]) ** 2) / 2
        return s

    def calculateDeltas(self, target: np.ndarray, forceRecalculate=False) -> np.ndarray:
        if not forceRecalculate and self.deltas != None:
            return self.deltas
        self.deltas = [0] * self.nodeCount
        if self.next == None:  # If there is no next layer, ie. this is the output layer
            for r in range(self.nodeCount):
                self.deltas[r] = (
                    self.outputValues[r]
                    * (1 - self.outputValues[r])
                    * (target[r] - self.outputValues[r])
                )
        else:
            frontDeltas = self.next.calculateDeltas(target, False)
            for r in range(self.nodeCount):
                sDownstream = 0
                for s in range(self.next.weights.shape[0]):
                    sDownstream += self.next.weights[s, r] * frontDeltas[s]
                self.deltas[r] = (
                    self.outputValues[r] * (1 - self.outputValues[r]) * sDownstream
                )

        return self.deltas

    def __repr__(self):
        return str(self.__dict__)
