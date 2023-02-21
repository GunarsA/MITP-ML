import math

import matplotlib.pyplot as plt
import numpy as np

X = np.array([4.0, 1.0, 0.0, 0.0, 7.0])  # + 2000
Y = np.array([2.5, 1.5, 2.2, 2.15, 5.0])  # * 1000

W_1 = 0
b_1 = 0
W_2 = 0
b_2 = 0


def linear(W, b, x):
    return W * x + b


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def model(x, W_1, b_1, W_2, b_2):
    return linear(W_2, b_2, sigmoid(linear(W_1, b_1, x)))


def loss_mae(y_prim, y):
    return np.sum(np.abs(y_prim - y))


Y_prim = model(X, W_1, b_1, W_2, b_2)
loss = loss_mae(Y_prim, Y)

print(f'Y_prim: {Y_prim}')
print(f'Y: {Y}')
print(f'loss: {loss}')
