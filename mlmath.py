import math
import numpy as np


def sigmoid(x):
    """
    Return σ(x)

    σ(x) = 1/(1 + e^(-x))
    """
    return 1 / (1 + np.exp(-x))
    # return x if x > 0 else 0


def sigmoid_prime(x):
    """
    Return σ'(x)

    σ'(x) = σ(x)*(1-σ(x))
    """
    return sigmoid(x) * (1 - sigmoid(x))


def normalize(x, min, max):
    """
    Return a value between 0 and 1, representing where it lies between min and max
    """
    return (x - min) / (max - min)


def deltaRule(target, output, prevOutput):
    return -(target - output) * output * (1 - output) * prevOutput
