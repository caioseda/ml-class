import matplotlib.pyplot as plt
from mapFeature import mapFeature
import numpy as np
import plot_ex2data2
from predizer_aprovacao import predizer
from mapFeature import mapFeature
from sigmoide import sigmoide

def plot(data, theta):
    
    t1 = np.arange(-1,1.5,0.01)
    t2 = np.arange(-1,1.5,0.01)

    T1, T2 = np.meshgrid(t1,t2)

    T1_flat = T1.ravel()
    T2_flat = T2.ravel()

    pontos = np.array([T1_flat, T2_flat]).T

    X_pontos = mapFeature(pontos[:,0],pontos[:,1], 6)

    y_pred = predict(X_pontos,theta)

    Z = np.array(y_pred, ndmin=2)
    Z = Z.reshape(T1.shape)

    plot_ex2data2.plot(data)
    plt.contour(T1, T2, Z, levels=[0], linewidths=2, colors='g',alpha=0.8)
    # plt.contourf(T1, T2, Z, levels=[np.min(Z), 0.5, np.max(Z)], cmap='Greens', alpha=0.4)


def predict(X, theta):
    y_pred = sigmoide(X * np.matrix(theta).T)
    return np.where(y_pred < 0.5, 0, 1)

