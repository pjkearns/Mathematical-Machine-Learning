import numpy as np
from scipy.linalg import lstsq, pinv, norm
from numpy.random import rand, randn, permutation

def BP_irls(A,b,p,maxIter=100):
# Solve basis pursuit using IRLS.
# % Inputs:       A is a (N x D) matrix whose rows are the measurement locations         
# %               b is a vector of length N whose values are the measurement values
# %               p is the norm to minimize
# %               
# % Outputs:      xhat is the vector of length D containing the approximate solution
    
    # algorithm parameters
    eps = 1e-5
    tol = 1e-4
    xhat = pinv(A)@b

    N, D = A.shape
    xhat = randn(D,1)
    
    for kk in range(maxIter):
        xprev = xhat
        w = (xhat**2 + eps)**((p-2)/2)
        Winv = np.diag(1/w.squeeze())
        xhat = (Winv@A.T)@lstsq(A@Winv@A.T,b)[0]
        
        # stopping condition
        err = norm(xprev - xhat)/norm(xhat)
        if err < tol:
            break

    return xhat


def st(X,tau):
# Soft-thresholding/shrinkage operator
# % Syntax:       Xs = st(X,tau)
# % Inputs:       X is the input matrix
# %               tau is the input shrinkage parameter
# %
# % Outputs:      Xs is the result of applying soft thresholding to every element in X
    Xs = np.multiply(np.sign(X),np.maximum(0,np.abs(X)-tau))
    return Xs


def lassoADMM(A,b,lam,maxIter=100): 
# Solve LASSO using ADMM.
# % Inputs:       A is a (N x D) matrix whose rows are the measurement locations         
# %               b is a vector of length N whose values are the measurement values
# %               lambda is the regularization tuning parameter
# 
# % Outputs:      xhat is the vector of length D containing the approximate solution
    
    # algorithm parameters
    rho = 10*lam
    tol = 1e-4

    N, D = A.shape
    u = np.zeros((D,1))
    z = np.zeros((D,1))

    Atrans = A.T@A + rho*np.eye(D)
    Abtrans = A.T@b

    for kk in range(maxIter):
        xhat = lstsq(Atrans,Abtrans + rho*(z-u))[0]         
        z    = st(xhat + u,lam/rho)
        u   += xhat - z

        # stopping condition
        err = norm(xhat - z)/norm(xhat)
        if err < tol:
            break

    return xhat