import numpy as np
import matplotlib.pyplot as plt
from linearRegCostFunction import linearRegCostFunction
import scipy.optimize as opt
from plot_ex5data1 import plot

def optimize_and_plot_linearReg(X, y, theta, lamb, epochs):

    costFunction = lambda opt_theta: linearRegCostFunction(X, y, opt_theta, lamb)    
    res = opt.minimize(costFunction, theta, jac=True, method='TNC', options={'maxiter': epochs})
    
    theta_min = res.x
    
    xx = np.arange(-50,40,1)
    xx = np.c_[np.ones(xx.shape[0]),xx]

    yy = np.dot(xx, theta_min)
    
    plot(X[:,1],y)
    plt.plot(xx[:,1],yy, c='b')

    return res.x
