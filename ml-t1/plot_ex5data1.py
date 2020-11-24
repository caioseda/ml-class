import scipy.io as spio
import numpy as np
import os
import matplotlib.pyplot as plt 

def importarDados(insertOnes=True, filepath='/data/ex5data1.mat', names=['Teste 1', 'Teste 2', 'Aceito']):
    path = os.getcwd() + filepath
    data = spio.loadmat(path)
    data = spio.loadmat(path)

    if insertOnes:
        data["X"] = np.c_[np.ones(len(data["X"])), data["X"]]
        data["Xval"] = np.c_[np.ones(len(data["Xval"])), data["Xval"]]
        data["Xtest"] = np.c_[np.ones(len(data["Xtest"])), data["Xtest"]]

    return data["X"], data["y"][:,0], data["Xval"], data["yval"][:,0], data["Xtest"], data["ytest"][:,0]

def plot(X,y, filename = 'target/plot5.1.png'):
    plt.scatter(X, y, marker='x', c='r')
    plt.xlabel("Mudança no nível da água (x)", fontsize=12)
    plt.ylabel("Quantidade de água fluido da barragem (y)", fontsize=11)
    plt.xlim((-50,40))
    plt.ylim((0,40))

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    plt.savefig(filename)