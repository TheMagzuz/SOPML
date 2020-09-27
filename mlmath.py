import math


def sigmoid(x):
    """
    Return σ(x)

    σ(x) = 1/(1 + e^(-x))
    """
    return 1 / (1 + math.exp(-x))


def normalize(x, min, max):
    """
    Return a value between 0 and 1, representing where it lies between min and max
    """
    return (x - min) / (max - min)
