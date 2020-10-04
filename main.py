from layer import Layer
import numpy as np
import typing
import random


def createLayers(layers: list):
    layerList = []
    l = Layer(layers[0])
    layerList.append(l)
    for i in range(1, len(layers)):
        l = Layer(layers[i], l)
        layerList.append(l)
    return layerList


def randomizeLayers(layers: typing.List[Layer], variance: float):
    for l in layers:
        rV = np.vectorize(lambda _: random.uniform(-variance, variance))
        l.weights = rV(l.weights)
        l.biases = rV(l.biases)
