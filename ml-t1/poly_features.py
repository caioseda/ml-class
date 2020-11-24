import numpy as np

def poly_features(X, p):
    X_poly = np.ones((X.shape[0], p))
    for i in range(p):
        X_poly[:,i] = X**(i+1)
    return X_poly