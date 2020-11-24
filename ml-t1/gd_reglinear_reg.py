import numpy as np
from linearRegCostFunction import linearRegCostFunction, gradiente_linear_reg


def gd_reglinear_reg(X, y, alpha, lamb, epochs,theta):
    cost = np.zeros(epochs)

    for i in range(epochs):
        gradient = gradiente_linear_reg(X, y, lamb, theta)
        theta = theta - (alpha * gradient)
        cost[i] = linearRegCostFunction(X, y, theta, lamb)

    return cost[-1], theta
