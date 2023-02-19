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
    return 1 / (1 + math.e ** -x)


def model(x, W_1, b_1, W_2, b_2):
    return linear(W_2, b_2, sigmoid(linear(W_1, b_1, x)))


def loss_mae(y_prim, y):
    return np.sum(np.abs(y_prim - y))


def dW_linear(W, b, x):
    return x


def db_linear(W, b, x):
    return 1


def dx_linear(W, b, x):
    return W


def dx_sigmoid(x):
    return np.exp(-x) / (1.0 + np.exp(-x)) ** 2


def dy_prim_loss_mae(y_prim, y):
    return (y_prim - y) / (np.abs(y_prim - y) + 1e-8)


def dW_1_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    d_layer_1 = dW_linear(W_1, b_1, x )
    d_layer_2 = dx_sigmoid(linear(W_1, b_1, x))
    d_layer_3 = dx_linear(W_2, b_2, sigmoid(linear(W_1, b_1, x)))
    d_loss = dy_prim_loss_mae(y_prim, y)
    return d_loss * d_layer_3 * d_layer_2 * d_layer_1


def db_1_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    d_layer_1 = db_linear(W_1, b_1, x)
    d_layer_2 = dx_sigmoid(linear(W_1, b_1, x))
    d_layer_3 = dx_linear(W_2, b_2, sigmoid(linear(W_1, b_1, x)))
    d_loss = dy_prim_loss_mae(y_prim, y)
    return d_loss * d_layer_3 * d_layer_2 * d_layer_1


def dW_2_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    d_layer_3 = dW_linear(W_2, b_2, sigmoid(linear(W_1, b_1, x)))
    d_loss = dy_prim_loss_mae(y_prim, y)
    return d_loss * d_layer_3


def db_2_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    d_layer_3 = db_linear(W_2, b_2, sigmoid(linear(W_1, b_1, x)))
    d_loss = dy_prim_loss_mae(y_prim, y)
    return d_loss * d_layer_3


learning_rate = 1e-1
losses = []
for epoch in range(1000):

    Y_prim = model(X, W_1, b_1, W_2, b_2)
    loss = loss_mae(Y_prim, Y)

    dW_1 = np.sum(dW_1_loss(X, W_1, b_1, W_2, b_2, Y_prim, Y))
    dW_2 = np.sum(dW_2_loss(X, W_1, b_1, W_2, b_2, Y_prim, Y))
    db_1 = np.sum(db_1_loss(X, W_1, b_1, W_2, b_2, Y_prim, Y))
    db_2 = np.sum(db_2_loss(X, W_1, b_1, W_2, b_2, Y_prim, Y))

    W_1 -= dW_1 * learning_rate
    W_2 -= dW_2 * learning_rate
    b_1 -= db_1 * learning_rate
    b_1 -= db_1 * learning_rate

    print(f'Y_prim: {Y_prim}')
    print(f'Y: {Y}')
    print(f'loss: {loss}')
    losses.append(loss)

plt.plot(losses)
plt.show()