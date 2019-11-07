"""
Author: Phillip Kearns, John Lipor
Nearest Subspace Classifier demo - nsClassify
python version: 3.7.2
"""
import numpy as np

def nsClassify(Xtest, Ufull):
    """
    Nearest-subspace classifier

    Syntax: estLabels = nsClassify(Xtest,Ufull)

    Inputs:
    :param Xtest: A D x N matrix of test data
    :param Ufull: Ufull is a D x r x K matrix where Ufull(:,:,k) is a basis for
    the subspace that best represents class k

    Output:
    estLabels is an N x 1 vector of estimated class labels
    """
    D, N = Xtest.shape
    _, r, K = Ufull.shape
    labels = np.zeros((N,K))
    estLabels = np.zeros((N,1))
    
    for nn in range(N):
        for kk in range(K):
            labels[nn,kk] = np.linalg.norm(Xtest[:,nn] - Ufull[:,:,kk]@Ufull[:,:,kk].T@Xtest[:,nn])
    
    estLabels = np.argmin(labels, axis=1)

    return estLabels