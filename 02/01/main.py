import numpy as np


def dot(X, Y):

    is_transposed = False

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    if X.shape[1] != Y.shape[0]:
        is_transposed = True
        Y = np.transpose(Y)

    X_rows = X.shape[0]
    Y_rows = Y.shape[0]
    Y_columns = Y.shape[1]

    product = np.zeros((X_rows, Y_columns))
    # TODO implement algorithm
    for i in range(X_rows):
        for j in range(Y_columns):
            product[i, j] = np.sum(X[i, :] * Y[:, j])

    if is_transposed:
        product = np.transpose(product)

    if product.shape[0] == 1:
        product = product.flatten()

    return product

A = np.array([
    [1, 3, 6],
    [5, 2, 8]
])

B = np.array([
    [1, 3],
    [5, 2],
    [6, 9]
])
C = np.array([1, 2, 3])
D = np.array([1, 2])

print(dot(C, B), np.dot(C, B))
print(dot(B, D), np.dot(B, D))



