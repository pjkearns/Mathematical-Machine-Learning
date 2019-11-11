"""
Authors: Phillip Kearns, Aaron Brown, Annabel Li-Pershing
Dictionary Learning Demo
"""
import numpy as np
from numpy.random import permutation
from scipy.sparse.linalg import svds
from scipy.linalg import lstsq

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

#   initialization of dictionary D
#   set initial D to M randomly chosen signals and normalize columns
    randYind = permutation(N)[:M]
    D = Y[:,randYind] 
    D = D/(np.sum(np.abs(D)**2,axis=0))**(1/2)

#   intialization of sparse representation X
    X = np.zeros((M,N))

    for it in range(max_iters):
#       sparse coding stage (orthogonal matching pursuit)
#       invoke OMP algorithm 
        X = OMP(D,Y,K) 

#       dictionary update stage
        for k in range(M):
            omega = np.nonzero(X[k,:])
            x_temp = X[:,omega]
            x_temp[k,:] = 0
            Ek = Y[:,omega] - D @ x_temp
            U,S,Vt = svds(Ek,1)
            D[:,k] = U
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
        x = Y[:,ii]
        resid = x
        for jj in range(K):
            places = np.argmax(np.abs(D.T@resid))
            if not ii:
                I = places[0]
            else:
                I = np.append(I, places[0])
#             xhat = pinv(D[:,I]) @ x
            xhat = lstsq(D[:,I],x)[0]
            resid = x - D[:,I] @ xhat

        tmp = np.zeros(M)
        tmp[I] = xhat
        Xhat[:,ii] = tmp
    return Xhat
                             
