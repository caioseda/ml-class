import numpy as np


def mapFeature(X1, X2, potencia):
    
    features = np.ones((X1.shape[0],1))
    
    for i in range(1, potencia+1):
        for j in range(i+1):
            features = np.c_[features,(X1**(i-j))*(X2**j)]
            
    return features