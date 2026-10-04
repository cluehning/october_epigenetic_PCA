# Mathematical Appendix: PCA from First Principles

Let there be $n$ samples and $p$ genomic features. The measurement matrix is
$X\in\mathbb{R}^{n\times p}$, with sample $i$ in row $i$ and feature $j$ in
column $j$. The feature mean is

$$
\bar{x}_j=\frac{1}{n}\sum_{i=1}^n X_{ij},
\qquad
X_c=X-\mathbf{1}\bar{x}^{\mathsf T}.
$$

Centering makes each feature column sum to zero. PCA is applied to $X_c$; the
project seeks a low-dimensional linear subspace that minimizes squared
reconstruction error.

## Variance-maximizing direction

For a unit direction $v\in\mathbb{R}^p$, the projected sample scores are
$X_cv$. Their sample variance is

$$
\operatorname{Var}(X_cv)=\frac{1}{n-1}\|X_cv\|_2^2
=v^{\mathsf T}\left(\frac{X_c^{\mathsf T}X_c}{n-1}\right)v.
$$

Define the sample covariance matrix $S=X_c^{\mathsf T}X_c/(n-1)$. To maximize
$v^{\mathsf T}Sv$ subject to $v^{\mathsf T}v=1$, form
$L(v,\lambda)=v^{\mathsf T}Sv-\lambda(v^{\mathsf T}v-1)$. Stationarity gives
$Sv=\lambda v$. Thus the first principal direction is a unit eigenvector of
$S$ with largest eigenvalue; subsequent directions are orthogonal eigenvectors
in descending eigenvalue order.

The eigenvalue $\lambda_k$ is the variance of scores along direction $v_k$.
The explained-variance ratio is

$$
r_k=\frac{\lambda_k}{\sum_{j=1}^{q}\lambda_j},
\qquad q=\operatorname{rank}(X_c)\leq\min(n-1,p).
$$

The cumulative ratio through $k$ components is $\sum_{j=1}^k r_j$.

## SVD and PCA

Take a thin singular value decomposition

$$
X_c=U\Sigma V^{\mathsf T},
$$

where columns of $U$ and $V$ are orthonormal and the diagonal entries
$\sigma_1\geq\cdots\geq\sigma_q\geq0$ are singular values. Then

$$
S=V\frac{\Sigma^2}{n-1}V^{\mathsf T},
\qquad
\lambda_k=\frac{\sigma_k^2}{n-1},
\qquad
T_k=X_cv_k=\sigma_k u_k.
$$

So PCA via SVD and covariance eigendecomposition gives the same principal
subspace. SVD is often numerically preferable: explicitly forming $X_c^TX_c$
can square the condition number. When $p$ is much larger than $n$, the dual
Gram matrix $X_cX_c^T/(n-1)$ is smaller. If
$X_cX_c^Tu_k=\sigma_k^2u_k$, then for nonzero $\sigma_k$,
$v_k=X_c^Tu_k/\sigma_k$.

The sign of each eigenvector is arbitrary: $(v_k,T_k)$ and $(-v_k,-T_k)$ are
the same axis. If eigenvalues are repeated, any orthonormal basis of the
corresponding eigenspace is valid. Hence compare eigenvalues, projections, or
reconstructions rather than signed loading columns alone.

## Projection and reconstruction geometry

Let $V_k=[v_1,\ldots,v_k]$. Since $V_k^TV_k=I_k$, the matrix
$P_k=V_kV_k^T$ is the orthogonal projector onto the retained feature-space
subspace. The rank-$k$ reconstruction and scores are

$$
\widehat{X}_k=X_cP_k=(X_cV_k)V_k^T=T_kV_k^T.
$$

Among all rank-$k$ approximations, this truncated-SVD reconstruction minimizes
the Frobenius error. Its squared error is $\sum_{j>k}\sigma_j^2$, and the
relative error reported by the code is

$$
\frac{\|X_c-\widehat{X}_k\|_F}{\|X_c\|_F}.
$$

## Scaling and effective dimension

The project uses feature-wise z-scores after signal compression. This replaces
the covariance matrix with the correlation matrix of the transformed features,
so low-variance and high-variance genomic windows receive equal marginal
variance. This is a scientific choice, not a neutral numerical step.

The participation ratio

$$
d_{\mathrm{eff}}=
\frac{(\sum_k\lambda_k)^2}{\sum_k\lambda_k^2}
$$

is near one when one eigenvalue dominates and equals $q$ when all $q$ nonzero
eigenvalues are equal. It is a continuous summary of spectral concentration.
It is not the number of components required for a chosen reconstruction
tolerance.

## What PCA does not establish

PCA finds orthogonal directions of high linear variance. It does not infer
causal regulatory mechanisms, optimize class separation, remove batch effects,
or guarantee that large-variance directions are biologically important. A
cell-type association is descriptive evidence that labels align with scores;
it needs careful validation against study design, batch, donor, and cell mixture.