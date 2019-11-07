"""
Author: Phillip Kearns
KRR demo - kernels.py
python version: 3.7.2
"""
import numpy as np

def linear(x,y):
    """Linear Kernel"""
    return np.dot(x,y)
    
def polynomial(x,y,r=1,n=2):
    """Polynomial Kernel"""
    return (r + np.dot(x,y))**n
    
def rbf(x,y,sig=1):
    """RBF (Gaussian) Kernel"""
    diff = np.subtract(x,y)
    return np.exp(-np.dot(diff,diff)/(2*sig**2))