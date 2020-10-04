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
        for i in range(l.nodeCount):
            l.weights[i] = random.uniform(-variance, variance)
            l.biases[i] = random.uniform(-variance, variance)
