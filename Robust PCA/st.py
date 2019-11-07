"""
Authors: Phillip Kearns, John Lipor
RPCA - demo - st.py
python version: 3.7.3
"""

import numpy as np
from numpy import sign, amax, absolute

def st(X, tau):
    """
    Soft - thresholding/shrinkage operator

    Syntax:     Xs = st(X, tau)

    Input: 
    :param X: Input matrix
    :param tau: Shrinkage parameter

    Output:
    Xs: The result of applying soft thresholding to every element in X
    """

    Xs = np.multiply(sign(X),np.maximum(0,absolute(X)-tau))
    return Xs
