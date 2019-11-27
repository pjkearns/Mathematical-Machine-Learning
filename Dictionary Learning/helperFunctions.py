"""
Authors: Phillip Kearns, Aaron Brown, Annabel Li-Pershing
Dictionary Learning Demo
"""
import numpy as np
from numpy.random import permutation
import scipy.sparse as spsp
from scipy.sparse.linalg import svds
from scipy.linalg import lstsq, pinv
from sklearn.preprocessing import normalize

def kSVD(Y,M,K):
    # % K-SVD algorithm as described in paper by Aharon, Elad Bruckstein:
    # % "K-SVD: An Algorithm for Designing Overcomplete Dictionaries for 
    # % Sparse Representation." IEEE Trans. on Sig. Proc., Vol. 54, No. 11, 2006.
    # % 
    # % Inputs:   Y   data matrix, size n X N (N examples each of size n)
    # %           M   desired number of atoms in the dictionary
    # %           K   target sparsity level
    # %
    # % Outputs:  D   dictionary of size n x M
    # %           X   sparse representation matrix of size M x N
    n, N = Y.shape

    # algorithm parameters
    max_iters = 80 

    # initialize D to M randomly chosen signals and normalize columns
    randYind = permutation(N)[:M]
    D = Y[:,randYind] 
    D = normalize(D,norm='l2',axis=0)

    for it in range(max_iters):
        # sparse coding stage (orthogonal matching pursuit)
        X = OMP(D,Y,K) 

        # dictionary update stage
        for k in range(M):
            omega = np.nonzero(X[k,:])[0]
            if len(omega) <= 1:
                continue
            x_temp = X[:,omega]
            x_temp[k,:] = 0
            Ek = Y[:,omega] - D @ x_temp
            U,S,Vt = svds(Ek,1)
            D[:,k] = U.squeeze()
            X[k,omega] = np.dot(S,Vt)

    return D, X


def OMP(D,Y,K):
    # % Orthogonal Matching Pursuit (OMP) algorithm as in Rubinstein, Zibulevsky, Elad:
    # % "Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit."
    # %
    # % Syntax:   Xhat = OMP(D,Y,K)
    # % 
    # % Inputs:   D       dictionary matrix, size n X M
    # %           Y       signal matrix to find representation
    # %           K       target sparsity level
    # %
    # % Output:   Xhat    spase representation matrix: Y \approx D*Xhat
    _, N = Y.shape
    _, M = D.shape
    Xhat = np.zeros((M,N))

    for ii in range(N):
        x = Y[:,ii].reshape(-1,1)
        resid = x
        for jj in range(K):
            places = np.argmax(np.abs(D.T @ resid))
            if jj == 0:
                I = places
                d = D[:,I].reshape(-1,1)
            else:
                I = np.append(I, places)
                d = D[:,I]
            xhat  = pinv(d) @ x
            resid = x - d @ xhat
        tmp = np.zeros((M,1))
        tmp[I] = xhat
        Xhat[:,ii] = tmp.squeeze()
    return Xhat


def ImgBlockPartition(img,row,col):
# % partition the pixels into blocks
# % syntax img_block = ImgBlockPartition(img,row,col)
# % INPUT : img         - image in grey scale
# %         row         - size of row of the image block
# %         col         - size of column of the image block
# %         noise       - noise level, from 0.1 to 0.9
# % OUTPUT: img_block   - block output of the image block matrix 
# %                     (row x col x [num blocks in row] x [num blocks in col])
# %         noise_block - block output of the image block matrix with noise
# %                     (row x col x [num blocks in row] x [num blocks in col])
    M, N = img.shape
    img_block = np.zeros((row*col,M//row,N//col))
    
    for ii in range(M//row):
        tempseg = img[ii*row:(ii+1)*row,:]
        for jj in range(N//col):
            tempblock = tempseg[:,jj*col:(jj+1)*col]
            img_block[:,ii,jj] = tempblock.flatten()

    return img_block

def ImgBlockReconstruct(img_hat,row,col,H,W):
# % reconstruct image from signals
# % INPUT : img_hat     - image signals in grey scale
# %         row         - size of row of the image block
# %         col         - size of column of the image block
# % OUTPUT: img_block   - output of the image 

    img = np.zeros((H,W))
    kk = 0
    for ii in range(H//row):
        for jj in range(W//col):
            img[ii*row:(ii+1)*row,jj*col:(jj+1)*col] = img_hat[:,kk].reshape((row,col))
            kk += 1
            
    return img