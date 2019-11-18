"""
Authors: Phillip Kearns, Aaron Brown, Annabel Li-Pershing
Dictionary Learning Demo
"""
import numpy as np
from numpy.random import permutation
from scipy.sparse.linalg import svds
from scipy.linalg import lstsq, pinv
from sklearn.preprocessing import normalize

def kSVD(Y,M,K):
    # % K-SVD algorithm as described in paper by Aharon, Elad Bruckstein:
    # % "K-SVD: An Algorithm for Designing Overcomplete Dictionaries for 
    # % Sparse Representation." IEEE Trans. on Sig. Proc., Vol. 54, No. 11, 2006.
    # %
    # % Syntax:   [D,X] = K-SVD(Y)
    # % 
    # % Inputs:   Y   data matrix, size n X N (N examples each of size n)
    # %           M   desired number of atoms in the dictionary
    # %
    # % Outputs:  D   dictionary of size n x K
    # %           X   sparse representation matrix of size K x N

    n, N = Y.shape

#   algorithm parameters
    max_iters = 80 

#   initialize D to M randomly chosen signals and normalize columns
    randYind = permutation(N)[:M]
    D = Y[:,randYind] 
    D = normalize(D,norm='l2',axis=0)

#   intialization of sparse representation X
#     X = np.zeros((M,N))

    for it in range(max_iters):
#       sparse coding stage (orthogonal matching pursuit)
#       invoke OMP algorithm 
        X = OMP(D,Y,K) 

#       dictionary update stage
        for k in range(M):
            omega = np.nonzero(X[k,:])[0]
            x_temp = X[:,omega]
            x_temp[k,:] = 0
            Ek = Y[:,omega] - D @ x_temp
            U,S,Vt = svds(Ek,1)
            D[:,k] = U.squeeze()
            X[k,omega] = np.dot(S,Vt)

    return D, X


def OMP(D,Y,K):
    # % Orthogonal Matching Pursuit (OMP) algorithm as described in paper by 
    # % Rubinstein, Zibulevsky, Elad:
    # % "Efficient Implementation of the K-SVD Algorithm using Batch 
    # % Orthogonal Matching Pursuit."
    # %
    # % Syntax:   Xhat = OMP(D,Y,K)
    # % 
    # % Inputs:   D       dictionary matrix, size n X M
    # %           Y       signal matrix to find representation
    # %           K       target sparsity level
    # %
    # % Output:   Xhat    spase representation matrix: Y \approx D*Xhat

    _, P = Y.shape
    _, M = D.shape
    Xhat = np.zeros((M,P))

    for ii in range(P):
        x = Y[:,ii].reshape(-1,1)
        resid = x
        for jj in range(K):
            places = np.argmax(np.abs(D.T@resid))
            if jj == 0:
                I = places
                d = D[:,I].reshape(-1,1)
            else:
                I = np.append(I, places)
                d = D[:,I]
#             xhat = lstsq(d,x)[0]
            xhat = pinv(d)@x
            resid = x - d @ xhat

        tmp = np.zeros((M,1))
        tmp[I] = xhat
        Xhat[:,ii] = tmp.squeeze()
    return Xhat