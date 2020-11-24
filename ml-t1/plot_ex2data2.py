import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def importarDados(insertOnes=True, filepath='/data/ex2data2.txt', names=['Teste 1', 'Teste 2', 'Aceito']):
    path = os.getcwd() + filepath
    data = pd.read_csv(path, header=None, names=names)

    data.head()

    if insertOnes:
        data.insert(0, 'Ones', 1)

    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1:cols]

    X = np.array(X.values)
    y = np.array(y.values)

    return data, X, y

def plot(data, filename = 'target/plot3.1.png'):

    positivo = data[data['Aceito'].isin([1])]
    negativo = data[data['Aceito'].isin([0])]

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(positivo['Teste 1'], positivo['Teste 2'], s=50, c='k', marker='+', label='Aceito')
    ax.scatter(negativo['Teste 1'], negativo['Teste 2'], s=50, c='y', marker='o', label='Nao Aceito')
    ax.legend()
    ax.set_xlabel('Resultado do Teste 1', fontsize=14)
    ax.set_ylabel('Resultado do Teste 2', fontsize=14)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    plt.savefig(filename)
    # plt.show()
    plt.grid(False)