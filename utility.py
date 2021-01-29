from math import exp


def sigmoid(x):
    return 1 / (1 + exp(-x))


def d_sigmoid(y):
    return y * (1-y)