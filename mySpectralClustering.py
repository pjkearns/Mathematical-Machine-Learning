"""
Author: Phillip Kearns
Spectral Clustering Demo - mySpectralClustering
python version: Python 3.7.2
"""
import numpy as np
import scipy as sp
import math
from numpy import linalg as lin
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import linear_sum_assignment

def mySpectralClustering(W, K, normalized):
    """
    Customized version of Spectral Clustering

    Inputs:
    -------
        W: weighted adjacency matrix of size N x N
        K: number of output clusters
        normalized: 1 for normalized Laplacian, 0 for unnormalized
    
    Outputs:
    -------
        estLabels: estimated cluster labels
        Y: transformed data matrix of size K x N
    """

    D = np.diag(np.sum(W,axis=0))
    L = D - W
    if normalized == 1:
        L = lin.inv(D) @ L 
    vals, vecs = lin.eig(L)
    idx  = vals.argsort()[::-1]   
    vals = vals[idx]
    vecs = vecs[:,idx]
    N, _ = W.shape
    Y = np.zeros((K,N))
    for kk in range(K):
        Y[kk,:] = vecs[:,kk]
    kmeans = KMeans(n_clusters=K).fit(Y.T)
    estlabels = kmeans.labels_
    return estlabels, Y