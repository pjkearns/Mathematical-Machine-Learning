"""
Authors: Phillip Kearns, John Lipor
KRR demo - KRR.py
python version: 3.7.2
"""

import numpy as np
from numpy import exp, eye
from numpy.linalg import norm, inv
from scipy.linalg import lstsq
from kernels import linear, polynomial, rbf

def KRR(X, y, lam, sigma, Xtest=None):
    """
    Train and predict Kernel Ridge Regression

    Syntax: ytrain, ytest = KRR(X, y, lam, Xtest)
    
    Inputs: 
        :param X:     A N x D matrix of training data
        :param lam:   The ridge regression tuning parameter
        :param sigma: The RBF kernel parameter
        :param Xtest: Optional matrix of test data
    
    Outputs:
        ytrain is the set of predicted labels for the training data X
        ytest is the set of predicted labels for the test data Xtest
    """
    N, D = X.shape
    K = np.zeros((N,N))
    
    # Train KRR
    for ii in range(N):
        for jj in range(N):
            K[ii,jj] = rbf(X[ii,:],X[jj,:],sigma)

    ytrain = y.T@lstsq(K + lam*np.eye(N),K)[0]

    ## Task 3
    if Xtest is not None:       
        Ntest, _ = Xtest.shape
        Ktest = np.zeros((N,Ntest))
        for ii in range(N):
            for jj in range(Ntest):
                Ktest[ii,jj] = rbf(X[ii,:],Xtest[jj,:],sigma)

        ytest = ytrain@lstsq(K + lam*np.eye(N),Ktest)[0]
    else:
        ytest=[]
    
    return ytrain, ytest

