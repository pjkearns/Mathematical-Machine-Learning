"""
Authors: Phillip Kearns, John Lipor
LRMC demo - sIRLSp
python version: 3.7.3
"""

import numpy as np
import time
from numpy.linalg import norm
from scipy.linalg import fractional_matrix_power

def sIRLSp(X, mask, p):
    """
    % Schatten p-norm minimization for matrix completion using IRLS
    % 
    % Syntax:       Xhat = sIRLSp(X,mask,p)
    %
    % Inputs:       X is a matrix of size (M x N)
    %               mask is a logical-valued matrix of sampling locations of size (M x N)
    %               p is the norm to minimize
    %
    % Outputs:      Xhat is a completed estimate of X of size (M x N)
    """
    M, N = np.shape(X)

    ## Algorithm parameters
    tol      = 1e-8
    maxIter  = 1000
    gammaMin = 10**(-20)
    eta      = 1.1
    sn       = norm(X)
    
    ## Initialize optmization variables
    Xmask = np.multiply(X,mask)
    Xhat = Xmask
    
    gamma = 0.01*(sn**2)
    s     = gamma**(1-p/2)

    for kk in range(maxIter):
        Xprev = Xhat
        W    = fractional_matrix_power(Xprev.T@Xprev + gamma*np.eye(N),p/2 - 1)
        Xhat = np.multiply(Xprev - s*Xprev@W, ~mask) + Xmask
            
        gamma = max(gamma/eta,gammaMin)
        s     = gamma**(1-p/2)
    
        # stopping conditions
        err = norm(Xprev - Xhat, ord='fro')/norm(Xhat, ord='fro')
        if err < tol:
            break
        Xprev = Xhat
    return Xhat


