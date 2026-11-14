# Appendix: Why k-Means Cannot Be Solved by Differentiation Alone — and How We Arrive at the Two-Step EM-Like Procedure

## 1. The objective and its two types of unknowns

Let \(X\in\mathbb{R}^{m\times n}\) contain the data points \(x_j\in\mathbb{R}^m\) as columns, let \(M\in\mathbb{R}^{m\times k}\) contain the cluster centroids \(\mu_i\) as columns, and let \(Z\in\mathbb{R}^{k\times n}\) be the assignment matrix with

\[
z_{ij}=
\begin{cases}
1, & \text{if } x_j \text{ is assigned to cluster } i,\\
0, & \text{otherwise}.
\end{cases}
\]

The k-means objective is

\[
J(M,Z)
=
\sum_{i=1}^{k}\sum_{j=1}^{n} z_{ij}\|x_j-\mu_i\|^2
=
\|X-MZ\|_F^2,
\]

subject to

\[
z_{ij}\in\{0,1\},
\qquad
\sum_{i=1}^{k} z_{ij}=1
\quad\text{for every }j.
\]

Thus, k-means has two very different kinds of unknowns:

1. **The centroids \(M\)** are continuous variables in \(\mathbb{R}^{m\times k}\).
2. **The assignments \(Z\)** are discrete, binary, one-hot variables.

This mixed continuous–discrete structure is the mathematical source of the difficulty. The objective is differentiable with respect to \(M\), but the assignment part is a combinatorial choice.

---

## 2. Differentiation with respect to the centroids works

If we fix the assignment matrix \(Z\), then \(J(M,Z)\) is a quadratic function of \(M\). Using

\[
J(M,Z)
=
\operatorname{tr}\bigl[(X-MZ)^T(X-MZ)\bigr],
\]

we obtain

\[
\frac{\partial J}{\partial M}
=
2\left(MZZ^T-XZ^T\right).
\]

Setting this derivative to zero gives

\[
MZZ^T = XZ^T,
\]

and therefore

\[
M = XZ^T(ZZ^T)^{-1}.
\]

Since \(ZZ^T\) is diagonal with the cluster sizes on its diagonal, this is exactly the centroid update:

\[
\mu_i
=
\frac{1}{n_i}\sum_{x_j\in C_i}x_j,
\qquad
n_i=|C_i|.
\]

So differentiation gives a clean closed-form solution for \(M\), **provided \(Z\) is already known**.

---

## 3. Differentiation with respect to the assignments does not solve the problem

Now suppose \(M\) is fixed. Then

\[
J(M,Z)
=
\sum_{j=1}^{n}\sum_{i=1}^{k} z_{ij}\|x_j-\mu_i\|^2.
\]

For fixed \(M\), this is **linear** in the entries \(z_{ij}\). The partial derivative is

\[
\frac{\partial J}{\partial z_{ij}}
=
\|x_j-\mu_i\|^2.
\]

Setting this derivative to zero would require

\[
\|x_j-\mu_i\|^2=0
\]

for all \(i,j\), which is impossible in general. More importantly, the assignment variables are constrained by

\[
z_{ij}\in\{0,1\},
\qquad
\sum_i z_{ij}=1.
\]

Because the objective is linear in each column of \(Z\), its minimum over the simplex occurs at a vertex. For each data point \(x_j\), the optimal assignment is

\[
z_{ij}=1
\quad\text{for}\quad
i=\arg\min_i \|x_j-\mu_i\|^2.
\]

This is the nearest-centroid assignment rule. It is a discrete **argmin**, not a solution obtained by setting a derivative to zero. The \(\arg\min\) operation is not differentiable in the usual sense, and at points where two centroids are equally close it is not even uniquely defined.

If we ignore the binary constraints and differentiate the matrix expression with respect to \(Z\), we get

\[
\frac{\partial}{\partial Z}\|X-MZ\|_F^2
=
-2M^TX+2M^TMZ.
\]

Setting this to zero gives

\[
Z=(M^TM)^{-1}M^TX.
\]

This is a continuous least-squares solution. In general it is **not binary**, its columns do not sum to one, and it does not represent a valid clustering. Thus, naive differentiation of the assignment matrix solves a relaxed regression problem, not k-means.

---

## 4. Matrix-factorization view

The same difficulty appears in the matrix-factorization form. Since

\[
J(M,Z)=\|X-MZ\|_F^2,
\]

we can eliminate \(M\) using the optimal centroid formula. Substituting

\[
M=XZ^T(ZZ^T)^{-1}
\]

into the objective gives the reduced problem

\[
\min_{Z}
\left\|
X-XZ^T(ZZ^T)^{-1}Z
\right\|_F^2
\]

subject to

\[
z_{ij}\in\{0,1\},
\qquad
\sum_i z_{ij}=1.
\]

This shows that k-means is a **constrained low-rank matrix factorization problem**. However, the remaining variable \(Z\) is still discrete and one-hot. The reduced objective is nonconvex, and the feasible set is combinatorial. In general, finding the globally optimal assignment is NP-hard. Therefore, even though the matrix expression is compact and differentiable in a relaxed sense, differentiation alone does not produce the clustering.

---

## 5. The two-step alternating solution

Because the problem has two blocks of variables, the natural mathematical strategy is **block coordinate descent**, also called **alternating minimization**. We alternate between optimizing one block while holding the other fixed.

### Step 1: Assignment step — fix \(M\), optimize \(Z\)

Given the current centroids \(M\), assign each point to its nearest centroid:

\[
z_{ij}=
\begin{cases}
1, & \text{if } i=\arg\min_{i'} \|x_j-\mu_{i'}\|^2,\\
0, & \text{otherwise}.
\end{cases}
\]

This is the discrete optimization over \(Z\). It is the **hard assignment** step.

### Step 2: Update step — fix \(Z\), optimize \(M\)

Given the current assignment \(Z\), update each centroid to the mean of its assigned points:

\[
\mu_i
=
\frac{1}{n_i}\sum_{x_j\in C_i}x_j,
\qquad
n_i=|C_i|.
\]

Equivalently,

\[
M=XZ^T(ZZ^T)^{-1}.
\]

This is the differentiable part of the problem.

The algorithm iterates these two steps until the assignments no longer change or the objective change is below a tolerance. Each step cannot increase the objective:

\[
J(M^t,Z^t)
\ge
J(M^t,Z^{t+1})
\ge
J(M^{t+1},Z^{t+1}).
\]

Since there are only finitely many partitions of \(n\) points into \(k\) clusters, the procedure terminates in finitely many steps at a local optimum.

---

## 6. Why this is an EM-like procedure

The two-step structure is exactly the same as in the EM algorithm:

- The **assignment step** is like the **E-step**: it infers the hidden cluster membership of each data point given the current model parameters.
- The **centroid update step** is like the **M-step**: it updates the model parameters given the current inferred assignments.

In a Gaussian mixture model with isotropic Gaussians of equal variance \(\sigma^2 I\), the E-step computes soft responsibilities

\[
r_{ij}
\propto
\exp\left(-\frac{\|x_j-\mu_i\|^2}{2\sigma^2}\right).
\]

As \(\sigma^2\to 0\), these soft responsibilities become concentrated on the nearest centroid and converge to the one-hot assignments used by k-means. In this limiting sense, k-means is a **hard-assignment EM** or **classification EM** procedure.

Therefore, the mathematical reason for the two-step algorithm is not an arbitrary algorithmic choice. It follows directly from the mixed discrete–continuous structure of the objective:

- differentiation gives the optimal centroids for fixed assignments;
- discrete optimization gives the optimal assignments for fixed centroids;
- alternating between the two yields a monotone, EM-like algorithm.

In short: **k-means cannot be solved by a single differentiation step because the assignment variables are discrete and combinatorial. Differentiation solves the centroid half of the problem; the assignment half requires a nearest-centroid argmin. Alternating these two solutions is precisely the k-means / hard-EM procedure.**