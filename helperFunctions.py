"""
Author: John Lipor
Spectral Clustering Demo - Helper Functions
python version: Python 3.7.2
"""

import numpy as np
import scipy as sp
import math
from numpy import linalg as lin
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import linear_sum_assignment


def myBestMap(trueLabels, estLabels):

    Inf = math.inf

    trueLabelVals = np.unique(trueLabels)
    kTrue = len(trueLabelVals)
    estLabelVals = np.unique(estLabels)
    kEst = len(estLabelVals)

    cost_matrix = np.zeros([kEst, kTrue])
    for ii in range(kEst):
        inds = np.where(estLabels == estLabelVals[ii])
        for jj in range(kTrue):
            cost_matrix[ii,jj] = np.size(np.where(trueLabels[inds] == trueLabelVals[jj]))
    
    rInd, cInd = linear_sum_assignment(-cost_matrix)

    outLabels = Inf * np.ones(np.size(estLabels)).reshape(np.size(trueLabels), 1)

    for ii in range(rInd.size):
        outLabels[estLabels == estLabelVals[rInd[ii]]] = trueLabelVals[cInd[ii]]

    outLabelVals = np.unique(outLabels)
    if np.size(outLabelVals) < max(outLabels):
        lVal = 1
        for ii in range(np.size(outLabelVals)):
            outLabels[outLabels == outLabelVals[ii]] = lVal
            lVal += 1       
    return outLabels
    
def missRate(trueLabels, estLabels):
    estLabels = myBestMap(trueLabels, estLabels)
    err = np.sum(trueLabels != estLabels) / np.size(trueLabels)

    return err, estLabels

# K - nearest neighbor algorithm
def myKNN(X, nNeighbors, wtype, sigma=None):
    """
    Customized version of KNN

    Inputs:
    -------
        X: data matrix of size (dimension) x (number of samples).
        nNeighbors: number of neighbors to connect for each point.
        type: weight type; 'constant' or 'gaussian'.
        sigma: tuning parameter for Gaussian weights.

    Outputs:
    -------
        W: weighted adjacency matrix.
    """
    if sigma is None:
        sigma = 1
    
    _, N = X.shape
    dMat = squareform(pdist(X.T))

    W = np.zeros((N, N))
    for ii in range(N):
        di = dMat[ii, :]
        vals = np.sort(di)
        inds = np.argsort(di)
        if wtype == 'constant':
            W[ii, inds[0: nNeighbors]] = 1
            W[inds[0: nNeighbors], ii] = 1
        else:
            W[ii, inds[0: nNeighbors]] = np.exp(-vals[0:nNeighbors]**2 / (2 * sigma**2))
            W[inds[0: nNeighbors], ii] = np.exp(-vals[0:nNeighbors]**2 / (2 * sigma**2))
   
    W = W - np.diag(np.diag(W))
    return W
