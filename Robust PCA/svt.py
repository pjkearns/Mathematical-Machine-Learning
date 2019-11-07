"""
Authors: Phillip Kearns, John Lipor
RPCA demo - svt.py
python version: 3.7.3
"""

import numpy as np
import numpy.linalg as la
from st import st

def svt(X, tau):
    """
    Singular value thresholding operator

    Syntax:     Xs = svt(X, tau)

    Inputs:
        :param X: The input matrix
        :param tau: The input shrinkage parameter

    Output: 
        Xs is the result of applying singular value thresholding to X
    """
    U, S, Vt = la.svd(X,full_matrices=False)
    Xs = np.dot(U * st(S,tau), Vt)
    return Xs
