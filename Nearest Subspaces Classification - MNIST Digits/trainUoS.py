"""
Author: Phillip Kearns, John Lipor
Nearest Subspace Classifier demo - trainUoS
python version: 3.7.2
"""
import numpy as np
from scipy.sparse.linalg import svds

def trainUoS(X, y, r):
    """
    Learn union of r - dimensional subspaces that best fit a dataset from K classes. 
    
    Syntax: Ufull = trainUoS(X,y,r)

    Inputs:
    :param X: A D x N matrix of training data
    :param y: y is an N x 1 vectors of labels
    :param r: r is the rank/dimension of the subspaces

    Output:  
    Ufull is a D x r x K matrix where Ufull(:,:,k) is a basis for
    the subspace that best represents class k
    """
    D, _  = X.shape
    K     = len(np.unique(y))
    Ufull = np.zeros((D, r, K))
    
    for ii in range(K):
        kk = ii*100
        U, _, _ = svds(X[:,kk:kk+100],r)
        Ufull[:,:,ii] = U[:,:]
    return Ufull
