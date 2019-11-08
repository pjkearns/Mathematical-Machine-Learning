# Robust Principal Component Analysis via Principle Component Pursuit (PCP)

This demo implements the *Principal Component Pursuit* (PCP) algorithm for robust principal component analysis using the alternating direction method of multipliers (ADMM) framework. 

PCA performs a low-rank approximation to the data matrix *X*, formulated as
    *L = arg min ||X - L||*{2}
    \text{subject to} && \text{rank}(L) \leq r.
\end{eqnarray*}
The above formulation assumes we know an upper bound on the desired rank. If this value is unknown, we could solve the alternative formulation
\begin{eqnarray*}
    \hat{L} &=& \argmin_{L \in \bR^{D \times N}} \; \text{rank}(L) \\
    \text{subject to} && \norm{X - L}_{F}^{2} \leq \varepsilon
\end{eqnarray*}
for some small value of $\varepsilon$ that determines our reconstruction error. Unfortunately, as with minimizing the number of nonzero elements in a vector, minimizing the rank of a matrix is a non-convex problem that is NP-hard to solve. The solution is to again use a convex relaxation! Recall that the relaxation of $\norm{\cdot}_{0}$ in the Lasso is $\norm{\cdot}_{1}$, i.e., we went from minimizing the \emph{total} number of nonzero elements to minimizing the \emph{sum} of absolute values.
In the case of matrices, we can view the rank as the total number of directions in which there is a nonzero component, which can be expressed as
\begin{equation*}
    \text{rank}(X) = \norm{\begin{bmatrix} \sigma_{1} \\ \sigma_{2} \\ \vdots \\ \sigma_{\min(D,N)} \end{bmatrix}}_{0}.
\end{equation*}
In words, since the rank of a matrix is the number of nonzero singular values, it is equivalent to the $\ell_{0}$-``norm'' of the vector of all singular values. With this in mind, the natural convex relaxation of rank is the $\ell_{1}$-norm of the singular values, which is known as the \emph{nuclear norm}
\begin{equation*}
    \norm{X}_{*} = \sum_{i = 1}^{\min(D,N)} \sigma_{i}.
\end{equation*}
Note that since $\sigma_{i} \geq 0$, this is exactly the $\ell_{1}$-norm of the vector of singular values. With this definition, we can reformulate PCA as
\begin{eqnarray*}
    \hat{L} &=& \argmin_{L \in \bR^{D \times N}} \; \norm{L}_{*} \\
    \text{subject to} && \norm{X - L}_{F}^{2} \leq \varepsilon,
\end{eqnarray*}
which is a convex problem that we do have a hope of solving.

\subsection*{Sparse Plus Low-Rank Models}

Suppose the matrix $X$ is corrupted by \emph{sparse} outliers stored in a matrix $S \in \bR^{D \times N}$ and we observe $Y = X + S$. In this case, we may wish to formulate an optimization problem that solves for the sparse and low-rank components separately
\begin{eqnarray*}
    \hat{L}, \hat{S} &=& \argmin_{L,S \in \bR^{D \times N}} \text{rank}(L) + \lambda \norm{S}_{0} \\
    \text{subject to} && Y = L + S.
\end{eqnarray*}
This formulation has both the rank and $\ell_{0}$-``norm'' issues, so we again solve the convex relaxation of the problem, which is written as
\begin{eqnarray}
    \label{eq:pcp}
    \hat{L}, \hat{S} &=& \argmin_{L,S \in \bR^{D \times N}} \norm{L}_{*} + \lambda \norm{S}_{1} \\
    \text{subject to} && Y = L + S \nonumber.
\end{eqnarray}
This problem is known as \emph{principal component pursuit} and is probably the most widely-considered formulation for robust PCA. In fact, nearly all robust PCA algorithms are variants of \eqref{eq:pcp}. In this demo, we will code the solution to the PCP problem using ADMM.

\subsection*{ADMM Iterations}

The ADMM iterations for solving \eqref{eq:pcp} are given below. In a homework, you may be asked to derive these, but for now simply note that the update on $L$ involves another soft-thresholding operator that is performed only on the singular values of the appropriate matrix. This should reinforce the similarity between nulcear norm minimization and $\ell_{1}$-norm minimization.

The augmented Lagrangian for this problem is
\begin{equation*}
    \sL(L,S,Z) = \norm{L}_{*} + \lambda \norm{S}_{1} + \ip{Z}{Y - L - S} + \frac{\rho}{2} \norm{Y - L - S}_{F}^{2},
    \label{eq:augLag}
\end{equation*}
where the matrix inner product above is defined as $\ip{A}{B} = \tr(A^{T}B)$ and $Z$ is the matrix of Lagrange multipliers.

The update for the low-rank component $L$ is
\begin{equation}
    L_{k+1} = \sD_{1/\rho}(Y - S_{k} + \frac{1}{\rho} Z_{k})
    \label{eq:lUpdate}
\end{equation}
which is the singular value thresholding operator defined as
\begin{equation*}
    \sD_{\tau}(X) = U \sS_{\tau}(\text{diag}(\Sigma)) V^{T}
\end{equation*}
where $X = U \Sigma V^{T}$ and $\sS_{\tau}(x)$ is the soft-thresholding operator.

The update for the sparse component $S$ is the soft-thresholding operator applied to the entire matrix
\begin{equation}
    S_{k+1} = \sS_{\lambda/\rho}(Y - L_{k+1} + \frac{1}{\rho} Z_{k}).
    \label{eq:sUpdate}
\end{equation}

Finally, the update for the Lagrange multipliers is the usual gradient-descent type update
\begin{equation}
    Z_{k+1} = Z_{k} + \rho(Y - L_{k+1} - S_{k+1}).
    \label{eq:zUpdate}
\end{equation}

\section*{Task 1: Implement PCP}

Your first task is to implement the above ADMM iterations to complete the \texttt{pcp.m} function. Follow the steps below in order.
\begin{itemize}
    \item Complete the \texttt{st.m} (soft thresholding) and \texttt{svt.m} (singular value thresholding) files and test them using the script \texttt{threshTest.m}. \textbf{Hint:} Your singular value thresholding function can and should call your soft thresholding function.
    \item Raise your hand when you have successfully completed the thresholding functions.
    \item Integrate your \texttt{st} and \texttt{svt} functions to complete \texttt{pcp.m}. Test this on the script \texttt{syntheticTest.m}
    \item Raise your hand when you have successfully completed the PCP function.
\end{itemize}

\section*{Task 2: Test PCP on Benchmark Data}

Your final task is to run your \texttt{pcp.m} algorithm on the included \texttt{lobby.mat} dataset using the script \texttt{lobbyTest.m}. This dataset includes a video of an office lobby in which a person walks through near the end. The stationary/background portion of the video is modeled as the low-rank component, while the person walking through is the sparse component. You should see that the second and third plots show the background only and person only. This is an example of using robust PCA for foreground-background
separation.

