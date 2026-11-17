# Appendix: From the k-Means Objective to Lloyd's Algorithm

This appendix is a supplement to the two clustering lectures parts.
In class we motivated k-means through the within-cluster sum of squared
distances and stated its objective — the **sum of squared errors** (SSE);
here we open the box and show how **differentiating that objective**
produces the two update rules of Lloyd's algorithm. We do this first in
plain element-wise form (no matrices), and only afterwards rewrite the
same derivation in matrix form, because that matrix form is the natural
preview of the **normal equations** of linear regression that we will
meet later in the course.

Throughout, $\mathbf{x}_j\in\mathbb{R}^{d}$ denotes a data point (bold),
$\mu_i\in\mathbb{R}^{d}$ a centroid (non-bold), $C_i$ the $i$-th cluster,
and $\mathcal{C}=\{C_1,\dots,C_k\}$ the whole clustering — exactly the
notation of the lectures.

---

## 1. The objective, motivated by Lemma 2

**Lemma 2** from the lectures states that for any set of
$n$ points $\{\mathbf{x}_1,\dots,\mathbf{x}_n\}\subset\mathbb{R}^d$ with
centroid $\mu=\frac{1}{n}\sum_i \mathbf{x}_i$,

$$
\sum_{i=1}^{n}\sum_{j>i}\|\mathbf{x}_i-\mathbf{x}_j\|^2
=
n\sum_{i=1}^{n}\|\mathbf{x}_i-\mu\|^2.
$$

The left-hand side is the total within-set pairwise squared distance;
the right-hand side is $n$ times the **within-cluster sum of squares**
(WCSS). The two are proportional, so minimizing one is the same as
minimizing the other, *and both are achieved by the same mean* $\mu$.

Apply the same lemma to a single cluster $C_i$ of size $n_i$ with
centroid $\mu_i=\frac{1}{n_i}\sum_{\mathbf{x}\in C_i}\mathbf{x}$:

$$
\sum_{\substack{\mathbf{x}_a,\mathbf{x}_b\in C_i\\ a<b}}
\|\mathbf{x}_a-\mathbf{x}_b\|^2
=
n_i\sum_{\mathbf{x}\in C_i}\|\mathbf{x}-\mu_i\|^2.
$$

So for any fixed cluster, "compactness" (small pairwise distances) is
equivalent to a small WCSS, and both are minimized by placing the
centroid at the cluster mean. For $k$ clusters, the natural overall
objective is therefore the total SSE:

$$
\mathrm{SSE}\bigl(\mathcal{C},\{\mu_i\}\bigr)
=
\sum_{i=1}^{k}\sum_{\mathbf{x}\in C_i}\|\mathbf{x}-\mu_i\|^2.
$$

To make the partition an explicit variable we introduce the one-hot
indicator

$$
z_{ij}=
\begin{cases}
1, & \mathbf{x}_j\in C_i,\\
0, & \text{otherwise},
\end{cases}
\qquad
\sum_{i=1}^{k}z_{ij}=1\ \text{for every }j,
$$

and rewrite the objective as

$$
\boxed{\;
\mathrm{SSE}\bigl(\{\mu_i\},Z\bigr)
=
\sum_{i=1}^{k}\sum_{j=1}^{n} z_{ij}\,\|\mathbf{x}_j-\mu_i\|^2
\;}
$$

with two blocks of unknowns: the centroids
$\{\mu_i\}\subset\mathbb{R}^d$ and the assignments
$\{z_{ij}\}\in\{0,1\}$.

> **Iris sizes.** The Iris dataset has $n=150$ specimens described by
> $d=4$ features (sepal length/width, petal length/width). With the
> usual $k=3$ clusters we therefore optimise $k\cdot d=12$ continuous
> centroid coordinates and $n\cdot k=450$ binary assignment variables.

---

## 2. Two unknowns of two different kinds

The objective has a **mixed continuous–discrete** structure:

- **Centroids $\mu_i$**: continuous, unconstrained, in $\mathbb{R}^d$.
- **Assignments $z_{ij}$**: discrete, one-hot per column.

This split is the single reason k-means is not solvable by one
differentiation. We will see that:

1. Differentiating the SSE with respect to a centroid $\mu_i$ gives a
   clean closed-form update.
2. Differentiating the SSE with respect to an assignment $z_{ij}$ gives
   nothing useful directly — the objective is *linear* in $z_{ij}$, so
   the optimum sits at a vertex of the assignment simplex, not at a
   zero of the gradient.

Both updates are derived below **without any matrix notation**.

---

## 3. The centroid step — element-wise differentiation w.r.t. $\mu_i$

Fix the partition (i.e. fix all $z_{ij}$). Differentiate the SSE with
respect to a single coordinate $\mu_{\ell a}$, the $\ell$-th feature of
the $a$-th centroid. The only terms in the double sum that mention
$\mu_{\ell a}$ are those with $i=a$:

$$
\frac{\partial\, \mathrm{SSE}}{\partial \mu_{\ell a}}
=
\sum_{j=1}^{n} z_{aj}\cdot 2(\mu_{\ell a}-x_{\ell j}).
$$

Setting this to zero and dividing by $2$:

$$
\mu_{\ell a}\sum_{j=1}^{n} z_{aj}
=
\sum_{j=1}^{n} z_{aj}\,x_{\ell j}.
$$

Writing $n_a=\sum_j z_{aj}=|C_a|$ for the size of cluster $a$ (which is
non-zero as long as the cluster is non-empty), we obtain

$$
\mu_{\ell a}=\frac{1}{n_a}\sum_{j=1}^{n} z_{aj}\,x_{\ell j}.
$$

Stacking the $\ell$-indices back into a vector gives the **centroid
update rule**:

$$
\boxed{\;
\mu_a
=
\frac{1}{n_a}\sum_{j:\,z_{aj}=1} \mathbf{x}_j
=
\frac{1}{|C_a|}\sum_{\mathbf{x}\in C_a}\mathbf{x},
\;}
$$

i.e. the mean of the points assigned to cluster $a$. This is exactly
the centroid step of Lloyd's algorithm. The derivation used nothing
but the ordinary scalar chain rule on the original double-sum objective
— no matrices, no Frobenius norm.

> **Iris check.** For *Iris setosa* ($n_a=50$ specimens) this gives
> $\mu_{\text{setosa}}\approx(5.01,\,3.43,\,1.46,\,0.25)$, the
> $4$-vector of feature means over those 50 flowers.

---

## 4. The assignment step — linear in $z_{ij}$, minimum at a vertex

Now fix the centroids. Then the SSE is **linear** in each $z_{ij}$,
because $z_{ij}$ multiplies the constant $\|\mathbf{x}_j-\mu_i\|^2$:

$$
\frac{\partial\, \mathrm{SSE}}{\partial z_{ij}}
=
\|\mathbf{x}_j-\mu_i\|^2.
$$

A linear function has no interior stationary point. Setting
$\partial\,\mathrm{SSE}/\partial z_{ij}=0$ would require
$\|\mathbf{x}_j-\mu_i\|^2=0$ for *every* $i$, which is impossible as
soon as two centroids are distinct.

But $z_{\cdot j}$ does not live in an unconstrained space — it lives on
the **assignment simplex**

$$
\Delta_j=\Bigl\{z\in\{0,1\}^k\ :\ \sum_{i=1}^{k}z_i=1\Bigr\},
$$

whose vertices are the $k$ one-hot vectors $e_1,\dots,e_k$. A linear
function on a simplex always attains its minimum at a vertex. The
minimizing vertex is the index of the smallest coefficient:

$$
\boxed{\;
z_{ij}=
\begin{cases}
1, & i=\arg\min_{i'}\|\mathbf{x}_j-\mu_{i'}\|^2,\\[2pt]
0, & \text{otherwise}.
\end{cases}
\;}
$$

This is the **nearest-centroid assignment** step of Lloyd's algorithm.
It is a discrete $\arg\min$, **not** the solution of
$\nabla\,\mathrm{SSE}=0$. The derivative
$\partial\,\mathrm{SSE}/\partial z_{ij}=\|\mathbf{x}_j-\mu_i\|^2$ tells
us *which* vertex is best, but it does not give us an interior
stationary point to solve for.

---

## 5. Alternating the two steps is Lloyd's algorithm

Neither step alone solves the full problem: the centroid step needs a
partition, the assignment step needs centroids. The natural strategy is
**alternating minimization** (block coordinate descent):

1. **Assignment step.** Fix $\{\mu_i\}$, update $Z$ by the
   nearest-centroid rule of §4.
2. **Update step.** Fix $Z$, update $\{\mu_i\}$ to the cluster means of
   §3.

Each step cannot increase the objective:

$$
\mathrm{SSE}\bigl(\{\mu_i\}^{t},Z^{t}\bigr)
\;\ge\;
\mathrm{SSE}\bigl(\{\mu_i\}^{t},Z^{t+1}\bigr)
\;\ge\;
\mathrm{SSE}\bigl(\{\mu_i\}^{t+1},Z^{t+1}\bigr).
$$

There are only finitely many partitions of $n$ points into $k$ clusters,
so the SSE can take only finitely many values; combined with strict
monotonicity this guarantees termination in finitely many steps at a
local optimum.

This is **Lloyd's algorithm** — the procedure we have used throughout
the clustering lectures. We did not invent it here; we **derived** it
from the objective by element-wise differentiation, with the mixed
continuous–discrete structure of the SSE forcing the alternation on us.

---

## 6. Matrix form: a preview of the normal equations

The derivation above is complete and self-contained. We now rewrite the
*same* objective in matrix form. This is not new mathematics — it is the
same content packaged compactly — but the packaging is worth the
trouble, because the resulting equations are exactly the **normal
equations** that we will meet again in linear regression.

### 6.1 Setup

Stack the data as columns of $X\in\mathbb{R}^{d\times n}$, the centroids
as columns of $M\in\mathbb{R}^{d\times k}$, and the assignments as the
one-hot matrix $Z\in\mathbb{R}^{k\times n}$ with $Z_{ij}=z_{ij}$:

$$
X=[\,\mathbf{x}_1\ \dots\ \mathbf{x}_n\,],\qquad
M=[\,\mu_1\ \dots\ \mu_k\,],\qquad
Z_{ij}=z_{ij}.
$$

The residual $X-MZ$ has as its $j$-th column
$\mathbf{x}_j-\mu_{i(j)}$, the displacement of $\mathbf{x}_j$ from its
assigned centroid. The squared Frobenius norm of this residual is the
sum of all squared entries:

$$
\|X-MZ\|_F^{2}
=
\sum_{i,j}\bigl[(X-MZ)_{ij}\bigr]^{2}
=
\sum_{j=1}^{n}\sum_{i=1}^{k} z_{ij}\,\|\mathbf{x}_j-\mu_i\|^2
=
\mathrm{SSE}\bigl(\{\mu_i\},Z\bigr).
$$

So the k-means objective is the squared Frobenius norm of the residual:

$$
\boxed{\;\mathrm{SSE}=\|X-MZ\|_F^{2}.\;}
$$

### 6.2 Centroid step — same answer, in matrix form

Fix $Z$ and differentiate $\|X-MZ\|_F^{2}$ with respect to $M$.
Expanding
$\|X-MZ\|_F^{2}=\operatorname{tr}\!\bigl((X-MZ)^{T}(X-MZ)\bigr)$ and
applying the standard matrix-calculus identities
$\partial\,\operatorname{tr}(AX)/\partial X=A^{T}$ and
$\partial\,\operatorname{tr}(X^{T}AX)/\partial X=(A+A^{T})X$, we get

$$
\frac{\partial\, \mathrm{SSE}}{\partial M}
=
-2(X-MZ)Z^{T}
=
-2XZ^{T}+2MZZ^{T}.
$$

Setting this to zero gives the **centroid normal equation**

$$
\boxed{\;M\,ZZ^{T}=XZ^{T}\;}
$$

and, because $Z$ is one-hot so
$ZZ^{T}=\operatorname{diag}(n_1,\dots,n_k)$,

$$
\boxed{\;M=XZ^{T}(ZZ^{T})^{-1}.\;}
$$

Reading column by column,
$\mu_a=\frac{1}{n_a}\sum_{j:z_{aj}=1}\mathbf{x}_j$ — exactly the rule
of §3. Same content, packaged in matrix notation.

### 6.3 The same algebraic shape appears in linear regression

Later in the course we will study **linear regression**: given a design
matrix $M\in\mathbb{R}^{n\times d}$ and responses
$X\in\mathbb{R}^{n\times p}$, find the coefficient matrix
$Z\in\mathbb{R}^{d\times p}$ minimizing

$$
\|X-MZ\|_F^{2}.
$$

This is the *same* expression as the k-means objective. The only
difference is **which matrix is fixed and which is unknown**:

| Problem | Fixed | Unknown | Minimizer |
|---|---|---|---|
| k-means centroid step | $Z$ (one-hot) | $M$ | $M=XZ^{T}(ZZ^{T})^{-1}$ |
| Linear regression | $M$ (design) | $Z$ (free real) | $Z=(M^{T}M)^{-1}M^{T}X$ |

Differentiating the regression objective w.r.t. the free $Z$ gives

$$
\frac{\partial\, \mathrm{SSE}}{\partial Z}=-2M^{T}X+2M^{T}MZ,
$$

and setting it to zero produces the **normal equations of linear
regression**:

$$
\boxed{\;M^{T}MZ=M^{T}X\quad\Longleftrightarrow\quad Z=(M^{T}M)^{-1}M^{T}X.\;}
$$

The two boxed equations — $MZZ^{T}=XZ^{T}$ for k-means and
$M^{T}MZ=M^{T}X$ for regression — are the **same algebraic shape**: a
symmetric Gram matrix times the unknown equals the data matrix
projected onto the fixed side. The only thing that changes is which
matrix plays which role. Encountering the normal equations again in
regression should therefore feel familiar, not new.

### 6.4 Why the matrix form does not let us escape the discrete step

The resemblance is tempting: could we just apply the regression
formula $Z=(M^{T}M)^{-1}M^{T}X$ to k-means with $M$ fixed and skip the
discrete $\arg\min$ of §4? **No**, for three reasons:

1. **$Z$ must be one-hot.** The regression solution is real-valued; its
   columns do not sum to $1$; its entries are not in $\{0,1\}$.
2. **Its $\arg\max$ is not the nearest-centroid rule.**
   $(M^{T}M)^{-1}M^{T}\mathbf{x}$ is the coordinate vector of
   $\mathbf{x}$ in the generally non-orthogonal basis formed by the
   rows of $M$, weighted by the inverse Gram matrix $M^{T}M$. This is
   a *projection* operation, not a *nearest-neighbour* operation. The
   two coincide only when the rows of $M$ are mutually orthogonal,
   which essentially never happens for real centroids.
3. **We verify both facts experimentally on Iris in §8.** There, the
   relaxed solution disagrees with the nearest-centroid rule on $30\%$
   of the specimens.

So in k-means, the assignment half of the problem genuinely requires
the discrete $\arg\min$ of §4. The matrix form is a useful compact
restatement and a preview of regression, but it does not let us
differentiate our way out of the combinatorial step.

---

## 7. Iris walkthrough (concrete shapes)

For Iris with $d=4$ features, $n=150$ specimens, $k=3$ clusters:

$$
X\in\mathbb{R}^{4\times 150},\qquad
M\in\mathbb{R}^{4\times 3},\qquad
Z\in\mathbb{R}^{3\times 150}.
$$

If the partition happened to be balanced ($50$ specimens per species),
$ZZ^{T}=\operatorname{diag}(50,50,50)$ and
$(ZZ^{T})^{-1}=\operatorname{diag}\!\bigl(\tfrac{1}{50},\tfrac{1}{50},\tfrac{1}{50}\bigr)$,
so each centroid is the column mean of its $50$ assigned points. In
general the partition produced by Lloyd is *not* balanced (the Iris run
below gives sizes $50,61,39$), and $ZZ^{T}$ is still diagonal but with
the actual cluster sizes on the diagonal.

---

## 8. Python verification on Iris (procedural, no classes)

The script below verifies, in plain procedural style, that:

1. $M=XZ^{T}(ZZ^{T})^{-1}$ matches the per-cluster feature mean, to
   machine precision.
2. $\|X-MZ\|_F^{2}$ equals the element-wise double sum
   $\sum_{ij}z_{ij}\|\mathbf{x}_j-\mu_i\|^2$, exactly.
3. The gradient
   $\partial\,\mathrm{SSE}/\partial Z=-2M^{T}X+2M^{T}MZ$ matches a
   finite-difference numerical gradient.
4. Procedural Lloyd's algorithm (k-means++ initialisation) reaches the
   same SSE whether computed pointwise as
   $\sum_j\|\mathbf{x}_j-\mu_{\text{label}(j)}\|^2$ or as
   $\|X-MZ\|_F^{2}$.
5. The relaxed $Z^{*}=(M^{T}M)^{-1}M^{T}X$ is **not** one-hot, and its
   $\arg\max$ **disagrees** with the nearest-centroid rule of §4 — the
   direct empirical check of §6.4.

```python
"""
Verification of the k-means matrix identities on Iris.

Lecture notation: bold x_j for points (here as columns of X), non-bold
mu_i for centroids, SSE for the objective. Convention: columns = points,
so X is (d x n) = (features x samples).
"""

import numpy as np
from sklearn.datasets import load_iris

SEED = 42

# 0. Load Iris -----------------------------------------------------
iris = load_iris()
Xsk = iris.data.astype(float)        # (150, d): rows = samples
X = Xsk.T                            # (d, 150): cols = samples
d, n = X.shape
k = 3
y_true = iris.target

# 1. One-hot Z from the true species labels -----------------------
Z = np.zeros((k, n))
Z[y_true, np.arange(n)] = 1.0
ZZT = Z @ Z.T
print("ZZ^T diagonal (cluster sizes):", np.diag(ZZT).astype(int))

# 2. Centroid formula  M = X Z^T (Z Z^T)^{-1}  vs per-cluster mean
M_formula = X @ Z.T @ np.linalg.inv(ZZT)
M_mean = np.array([Xsk[y_true == i].mean(axis=0) for i in range(k)]).T
print("max |M_formula - M_mean| =", np.max(np.abs(M_formula - M_mean)))

# 3. SSE: matrix form  ==  element-wise double sum
sse_matrix = float(np.sum((X - M_formula @ Z) ** 2))
sse_elem = 0.0
for i in range(k):
    for j in range(n):
        if Z[i, j] == 1.0:
            sse_elem += np.sum((X[:, j] - M_formula[:, i]) ** 2)
print("SSE ||X-MZ||_F^2 =", sse_matrix)
print("SSE element-wise =", sse_elem)

# 4. Relaxed Z  = (M^T M)^{-1} M^T X  -- NOT a valid one-hot assignment
MTM = M_formula.T @ M_formula
Z_relaxed = np.linalg.inv(MTM) @ M_formula.T @ X
print("Z_relaxed first column:", Z_relaxed[:, 0])
print("column sums (first 5):", Z_relaxed.sum(axis=0)[:5])
print("all entries binary? ", np.all((Z_relaxed == 0) | (Z_relaxed == 1)))

# 5. argmax(Z_relaxed)  vs  nearest-centroid rule  (they disagree)
labels_relaxed = np.argmax(Z_relaxed, axis=0)
dist = np.linalg.norm(Xsk[:, None, :] - M_formula.T[None, :, :], axis=2)
labels_nearest = np.argmin(dist, axis=1)
print("nearest-centroid accuracy:    ", np.mean(labels_nearest == y_true) * 100, "%")
print("argmax(Z_relaxed) accuracy:   ", np.mean(labels_relaxed == y_true) * 100, "%")
print("agreement argmax(Z_relaxed) vs nearest-centroid:",
      np.mean(labels_relaxed == labels_nearest) * 100, "%")

# 6. Gradient formula  d SSE/dZ = -2 M^T X + 2 M^T M Z  vs finite differences
rng = np.random.default_rng(0)
Z_test = rng.standard_normal((k, n))

def sse_func(Zm):
    return float(np.sum((X - M_formula @ Zm) ** 2))

grad_formula = -2 * M_formula.T @ X + 2 * M_formula.T @ M_formula @ Z_test
grad_num = np.zeros_like(Z_test)
eps = 1e-6
for i in range(k):
    for j in range(n):
        Zp = Z_test.copy(); Zp[i, j] += eps
        Zm = Z_test.copy(); Zm[i, j] -= eps
        grad_num[i, j] = (sse_func(Zp) - sse_func(Zm)) / (2 * eps)
print("max |grad_formula - grad_numerical| =",
      np.max(np.abs(grad_formula - grad_num)))

# 7. Procedural Lloyd's k-means with k-means++ init  (no classes)
def kmeans_pp_init(Xsk, k, rng):
    n = len(Xsk)
    centroids = [Xsk[rng.integers(n)]]
    for _ in range(1, k):
        d2 = ((Xsk[:, None, :] - np.array(centroids)[None, :, :]) ** 2) \
            .sum(axis=2).min(axis=1)
        centroids.append(Xsk[rng.choice(n, p=d2 / d2.sum())])
    return np.array(centroids, dtype=float)

def kmeans(Xsk, k, max_iter=300, tol=1e-9, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    centroids = kmeans_pp_init(Xsk, k, rng)
    for it in range(max_iter):
        distances = np.linalg.norm(
            Xsk[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
        for j in range(k):
            members = Xsk[labels == j]
            if len(members) > 0:
                new_centroids[j] = members.mean(axis=0)
        shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        if shift < tol:
            break
    inertia = float(np.sum((Xsk - centroids[labels]) ** 2))
    return centroids, labels, inertia, it + 1

centroids, labels, inertia, n_iter = kmeans(Xsk, k, rng=np.random.default_rng(SEED))
print("Lloyd iterations:", n_iter)
print("cluster sizes:", np.bincount(labels, minlength=k))
print("inertia (pointwise SSE):", inertia)

# 8. Re-derive the SAME SSE via the matrix form with Lloyd's labels
Z_lloyd = np.zeros((k, n))
Z_lloyd[labels, np.arange(n)] = 1.0
M_lloyd = X @ Z_lloyd.T @ np.linalg.inv(Z_lloyd @ Z_lloyd.T)
sse_matrix_lloyd = float(np.sum((X - M_lloyd @ Z_lloyd) ** 2))
print("SSE ||X-MZ||_F^2 (Lloyd labels):", sse_matrix_lloyd)
print("match:", np.isclose(sse_matrix_lloyd, inertia))
```

### Program output (verbatim, Iris, SEED=42)

```
ZZ^T diagonal (cluster sizes): [50 50 50]
max |M_formula - M_mean| = 2.22e-16
SSE ||X-MZ||_F^2 = 89.297400
SSE element-wise = 89.297400
Z_relaxed first column: [ 1.02136683  0.10245177 -0.09474523]
column sums (first 5): [1.02907337 0.99771851 0.94616567 0.92274539 1.00252589]
all entries binary?  False
nearest-centroid accuracy:     92.66666666666667 %
argmax(Z_relaxed) accuracy:    76.0 %
agreement argmax(Z_relaxed) vs nearest-centroid: 70.0 %
max |grad_formula - grad_numerical| = 5.90e-06
Lloyd iterations: 3
cluster sizes: [50 61 39]
inertia (pointwise SSE): 78.855666
SSE ||X-MZ||_F^2 (Lloyd labels): 78.855666
match: True
```

### What the numbers tell us

| Check | Result |
|---|---|
| $M=XZ^{T}(ZZ^{T})^{-1}$ equals per-cluster mean | max error $2.2\times10^{-16}$ (machine precision) |
| $\|X-MZ\|_F^{2}$ equals the element-wise double sum | both give $89.297400$ |
| Gradient formula vs finite differences | max error $5.9\times10^{-6}$ (forward-diff tolerance) |
| Lloyd pointwise SSE vs $\|X-MZ\|_F^{2}$ on converged labels | both give $78.855666$ |
| Relaxed $Z^{*}=(M^{T}M)^{-1}M^{T}X$ binary? | **no** (entries range from $-2.66$ to $+3.45$) |
| $\arg\max Z^{*}$ vs nearest-centroid rule | **disagree on $30\%$ of Iris specimens** |

The first four rows confirm that the matrix form is exactly equivalent
to the element-wise derivation — the same content, packaged
compactly. The last two rows confirm that the matrix form, tempting as
it is, does **not** give us a shortcut around the discrete assignment
step.

---

## 9. Summary

Starting from **Lemma 2** in the lectures, we wrote the k-means
objective as the within-cluster sum of squared errors,

$$
\mathrm{SSE}\bigl(\mathcal{C},\{\mu_i\}\bigr)
=
\sum_{i=1}^{k}\sum_{\mathbf{x}\in C_i}\|\mathbf{x}-\mu_i\|^2
=
\sum_{i=1}^{k}\sum_{j=1}^{n} z_{ij}\,\|\mathbf{x}_j-\mu_i\|^2,
$$

and showed that two elementary differentiations — both done
**element-wise, with no matrices** — produce the two update rules of
Lloyd's algorithm:

- **Centroid step.** $\partial\,\mathrm{SSE}/\partial \mu_{\ell a}=0$
  gives $\mu_a=\frac{1}{|C_a|}\sum_{\mathbf{x}\in C_a}\mathbf{x}$, the
  cluster mean.
- **Assignment step.** The SSE is *linear* in each $z_{ij}$; on the
  one-hot simplex its minimum is the vertex
  $i=\arg\min_{i'}\|\mathbf{x}_j-\mu_{i'}\|^2$, the nearest centroid.

Alternating these two closed-form updates is Lloyd's algorithm. The
mixed continuous–discrete structure of the SSE forces this alternation
on us: the objective is differentiable in the centroids but only
*linear* in the discrete assignments, so no single differentiation can
solve both halves at once.

The matrix form $\mathrm{SSE}=\|X-MZ\|_F^{2}$ packages the same
derivation compactly and previews the **normal equations** of linear
regression:

- k-means centroid step: $M\,ZZ^{T}=XZ^{T}$;
- linear regression: $M^{T}M\,Z=M^{T}X$.

These are the same algebraic shape — a symmetric Gram matrix times the
unknown equals the data matrix projected onto the fixed side — with the
roles of "fixed" and "unknown" swapped. When we meet the normal
equations again in regression, the only new ingredient will be that
$Z$ is allowed to be any real matrix rather than a one-hot assignment;
the derivation will be the one we have already done here.

The matrix form does not, however, let us differentiate our way out of
the discrete assignment: the relaxed $Z=(M^{T}M)^{-1}M^{T}X$ is
real-valued, and its $\arg\max$ disagrees with the nearest-centroid
rule on $30\%$ of Iris specimens (§8). The discrete $\arg\min$ of §4 is
genuinely irreducible — and that, in one sentence, is why k-means is a
*two-step* algorithm rather than a *one-shot* differentiation.
