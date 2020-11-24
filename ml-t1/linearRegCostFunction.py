import numpy as np

def linearRegCostFunction(X, y, theta, lamb):
    
    m = len(X)
    h = np.dot(X,theta)
    erro = h - y
    grad_term = np.sum(np.power(erro,2))/(2*m)
    reg_term =  lamb * np.sum(np.power(theta[1:],2))/(2*m)

    custo = grad_term + reg_term
    
    #gd
    grad =  erro.dot(X)/m
    grad[1:] = grad[1:] + (theta[1:]/m) * lamb
    
    return custo, grad
