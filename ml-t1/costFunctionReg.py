import numpy as np
from sigmoide import sigmoide

def costFunctionReg(theta, X, y, lamb):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    m = len(X)
    h = sigmoide(X * theta.T)

    grad0 = np.multiply(-y, np.log(h))
    grad1 = np.multiply((1 - y), np.log(1 - h))
    grad_sum =  np.sum(grad0 - grad1) / m
    reg_term = lamb * np.sum(np.power(theta,2))/(2 * m)
    J = grad_sum + reg_term

    #gd
    grad = np.zeros((1,theta.shape[1]))

    erro = h - y
    term = np.multiply(erro, X)
    gd_term = np.sum(term, axis = 0) / m
    
    grad[:,1:] = gd_term[:,1:] + (theta[:,1:]/m) * lamb
    grad[:,0] = gd_term[:,0]

    return  J, grad
