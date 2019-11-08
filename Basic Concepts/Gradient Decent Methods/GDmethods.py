import numpy as np

def lsgd(A,b,mu,x0,numIter):
# % Gradient descent for solving least-squares problems
# %
# % Syntax:       xgd = lsgd(A,b,mu,x0,maxIters)
# %               
# % Inputs:       A is an (m x n) matrix              
# %               b is a vector of length m
# %               mu is the step size to use and must satisfy
# %                      0 < mu < 2/norm(A,2)^2 to guarantee convergence
# %               x0 is the initial starting vector (of length n) to use
# %               numIter is the number of iterations to perform
# %               
# % Outputs:      xgd is a vector of length n containing the approximate solution
    x = x0
    
    for ii in range(numIter):
        x = x - mu*A.T@(A@x - b)
    
    xgd = x
    return xgd

def lsgdNesterov(A,b,mu,x0,numIter):
# % Nesterov's gradient descent for solving least-squares problems
# % Same parameters as lsgd
    t_k  = 0
    x_k  = x0
    x_km = x0
    x_kp = 0

    for ii in range(numIter):
        t_kp = (1+np.sqrt(1+4*t_k**2))/2

        z_kp = x_k + ((t_k - 1) / t_kp)*(x_k - x_km)
        x_kp = z_kp - mu*A.T@(A@z_kp - b)

        t_k  = t_kp
        x_km = x_k
        x_k  = x_kp
        
    xgdNesterov = x_kp
    return xgdNesterov

def lsgdNonNeg(A,b,mu,x0,numIter):
# % Non-negative least-squares w/ gradient descent
# % Same parameters as above
    zero_vec = np.zeros(x0.shape)
    x = x0

    for ii in range(numIter):
        x = np.maximum(zero_vec, x - mu*A.T@(A@x - b));

    xnn = x
    return xnn