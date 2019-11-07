"""
Author: John Lipor, Phillip Kearns
RPCA demo - pcp.py
python version: 3.7.3
"""

import numpy as np
from numpy.linalg import norm
from numpy import sqrt, zeros
from svt import svt
from st import st

def pcp(Y):
    """
    Principal Component Pursuit solved with ADMM

    Syntax:     L, S = pcp(Y)

    Inputs:     
        :param Y: A matrix of size [D, N]

    Ouputs:
        L: The low rank matrix with size [D, N]
        S: The sparse matrix with size [D, N]
    """
    ### Parameters that we'll use
    D, N = np.shape(Y)
    normY = norm(Y, ord='fro')

    ### Algorithm parameters
    lam = 1 / sqrt(max(D, N))
    rho = 10 * lam
    tol = 1e-4
    maxIter = 1000

    ### Initialize the variables of optimization
    L = zeros([D, N])
    S = zeros([D, N])
    Z = zeros([D, N])

    ### ADMM Iterations
    for ii in range(maxIter):
        L = svt((Y - S + (1/rho)*Z),1/rho)
        S = st((Y - L + (1/rho)*Z),lam/rho)
        Z = Z + rho*(Y - L - S)
    
        ## Calculate error and output at each iteration:
        err = norm(Y - L - S,ord='fro')
        if err < tol:
            break

    return L, S

