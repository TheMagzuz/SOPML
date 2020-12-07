from layer import Layer
import numpy as np
import typing
import random
import mlmath
from dataparser import Dataparser
from pprint import pprint
from time import perf_counter


learningRate = 0.3


def run():
    # Get training images and labels
    tStart = perf_counter()
    print("Loading training images...")
    dpTrain = Dataparser()
    dpTrain.loadLabels()
    dpTrain.loadImages()

    tLoadTrain = perf_counter()
    print(f"Done! Loading training images took {tLoadTrain-tStart}s")

    # Get test images and labels
    tLoadTest = perf_counter()
    dpTest = Dataparser()
    dpTest.loadLabels()
    dpTest.loadImages()

    tLoadTest = perf_counter()
    print(f"Done! Loading test images took {tLoadTest-tLoadTrain}s")

    print("Creating layers...")
    layersTemplate = [len(dpTrain.images[0].normalizedData), 16, 16, 10]
    layers = createLayers(layersTemplate)
    randomizeLayers(layers, 0.05)

    tCreate = perf_counter()

    print(f"Done! Creating layers took {tCreate-tLoadTrain}s")

    for layer in layers:
        if not hasattr(layer, "weights"):
            pprint("None")
        else:
            pprint(layer.weights)

    layers[-1].calculateValues(
        np.array(dpTrain.images[0].normalizedData), forceRecalculate=True
    )
    firstCost = layers[-1].cost(dpTrain.images[0].expectedVector())
    print(f"Initial cost: {firstCost}")

    print("Running on all training examples...")
    for _ in range(1):
        trainingPass(layers)
    tTrain = perf_counter()
    print(f"Done! Running all training examples took {tTrain-tCreate}s")
    for layer in layers:
        if not hasattr(layer, "weights"):
            pprint("None")
        else:
            pprint(layer.weights)

    layers[-1].calculateValues(
        np.array(dpTrain.images[0].normalizedData), forceRecalculate=True
    )
    secondCost = layers[-1].cost(dpTrain.images[0].expectedVector())
    print(f"New cost: {secondCost}")
    print("Outputs:")
    for output in layers[-1].outputValues:
        print(output)
    print(f"Label: {dpTrain.images[0].label}")


def trainingPass(layers):
    for t in dpTrain.images:
        layers[-1].calculateValues(np.array(t.normalizedData), forceRecalculate=True)
        layers[0].calculateDeltas(t.expectedVector(), forceRecalculate=True)
        for layer in layers:
            if layer.previous == None:
                continue
            for current, prev in np.ndindex(layer.weights.shape):
                change = (
                    learningRate * layer.deltas[current] * layer.inputValues[current]
                )
                layer.weights[current, prev] += change


def createLayers(layers: list):
    layerList = []
    l = Layer(layers[0])
    layerList.append(l)
    for i in range(1, len(layers)):
        l = Layer(layers[i], l)
        layerList.append(l)
        layerList[i - 1].next = l
    return layerList


def randomizeLayers(layers: typing.List[Layer], variance: float):
    rV = np.vectorize(lambda _: random.uniform(-variance, variance))
    for l in layers:
        if not hasattr(l, "weights"):
            continue
        l.weights = rV(l.weights)
        l.biases = rV(l.biases)  # Maybe leave out biases?


def findDerivatives(layers: typing.List[Layer], target: np.ndarray):
    cw = []
    """ The partial derivative of cost over weight: ∂C/∂w_kj"""

    for i in range(len(layers)):  # For every layer
        if (
            i == 0
        ):  # If this is the output layer, the derivative can be calculated by the delta rule
            cw[i] = []
            for outnode, innode in np.ndindex(layers[i].weights.shape):
                cw[i][outnode, innode] = mlmath.deltaRule(
                    target[outnode],
                    layers[i].outputValues[outnode],
                    layers[i - 1].outputValues[innode],
                )
        else:
            cw[i] = []
            for outnode in range(layers[i].weights.shape[0]):
                sDownstream = 0
                """ The sum of the error of the downstream nodes """

                # for innode in range(layers[i].weights.shape[1]):
                # sDownstream += layers[i].weights[outnode, innode] *


if __name__ == "__main__":
    run()
