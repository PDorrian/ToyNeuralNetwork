import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def d_sigmoid(y):
    return y * (1-y)