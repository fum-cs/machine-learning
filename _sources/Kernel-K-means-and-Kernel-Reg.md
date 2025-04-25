# Kernel K-means Clustering

## Review: Standard K-means Algorithm

K-means is a widely used clustering algorithm that partitions a dataset into $k$ clusters by minimizing the sum of squared distances between each point and its assigned cluster centroid.

**Algorithm Steps:**
1. **Initialization:** Randomly select $k$ initial centroids $\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_k$.
2. **Assignment Step:** Assign each data point $\mathbf{x}_j$ to the nearest centroid:
   $$
   i^* = \arg\min_{i} \left\{ \|\mathbf{x}_j - \boldsymbol{\mu}_i\|^2 \right\}
   $$
3. **Update Step:** Recompute each centroid as the mean of the points assigned to it:
   $$
   \boldsymbol{\mu}_i = \frac{1}{|C_i|} \sum_{\mathbf{x}_j \in C_i} \mathbf{x}_j
   $$
4. **Repeat** steps 2 and 3 until convergence (i.e., assignments no longer change or centroids stabilize).

**Objective Function:**
$$
SSE(\mathcal{C}) = \sum_{i=1}^{k} \sum_{\mathbf{x}_j \in C_i} \|\mathbf{x}_j - \boldsymbol{\mu}_i\|^2
$$

**Limitation:**  
K-means can only find clusters separated by linear boundaries. It fails when clusters have nonlinear structure.

---

## Kernel K-means: Motivation

To overcome the linearity limitation, Kernel K-means uses the **kernel trick** to implicitly map data into a higher-dimensional feature space where clusters may become linearly separable.

- Let $\phi(\mathbf{x})$ be a mapping from the input space to a (possibly infinite-dimensional) feature space.
- Instead of computing $\phi(\mathbf{x})$ explicitly, we use a kernel function $K(\mathbf{x}, \mathbf{z}) = \langle \phi(\mathbf{x}), \phi(\mathbf{z}) \rangle$.

---

## Kernel K-means Algorithm

**Definitions:**
- Let $\mathbf{K}$ be the $n \times n$ kernel matrix with $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$.
- The centroid of cluster $C_i$ in feature space is:
  $$
  \boldsymbol{\mu}_i^\phi = \frac{1}{n_i} \sum_{\mathbf{x}_j \in C_i} \phi(\mathbf{x}_j)
  $$

**Distance in Feature Space:**
The squared distance between a point and a cluster centroid in feature space can be computed using only kernel values:
$$
\|\phi(\mathbf{x}_j) - \boldsymbol{\mu}_i^\phi\|^2 = K(\mathbf{x}_j, \mathbf{x}_j) - \frac{2}{n_i} \sum_{\mathbf{x}_a \in C_i} K(\mathbf{x}_a, \mathbf{x}_j) + \frac{1}{n_i^2} \sum_{\mathbf{x}_a, \mathbf{x}_b \in C_i} K(\mathbf{x}_a, \mathbf{x}_b)
$$

**Algorithm Steps:**
1. **Initialization:** Randomly assign points to $k$ clusters.
2. **For each cluster $C_i$ and each point $\mathbf{x}_j$:**
   - Compute:
     - $\text{sqnorm}_i = \frac{1}{n_i^2} \sum_{\mathbf{x}_a, \mathbf{x}_b \in C_i} K(\mathbf{x}_a, \mathbf{x}_b)$
     - $\text{avg}_{ji} = \frac{1}{n_i} \sum_{\mathbf{x}_a \in C_i} K(\mathbf{x}_a, \mathbf{x}_j)$
   - Compute the distance:
     $$
     d(\mathbf{x}_j, C_i) = \text{sqnorm}_i - 2 \cdot \text{avg}_{ji}
     $$
3. **Assignment Step:** Assign each point to the cluster with the minimum $d(\mathbf{x}_j, C_i)$.
4. **Repeat** steps 2 and 3 until convergence.

**Objective Function in Kernel Space:**
$$
\min_{\mathcal{C}} \; SSE(\mathcal{C}) =
\sum_{i=1}^k \sum_{\mathbf{x}_j \in C_i}
\|\phi(\mathbf{x}_j) - \boldsymbol{\mu}_i^\phi\|^2
$$

---

## Pseudocode

> If you cannot render the LaTeX pseudocode, you may include an image of the algorithm from the slides.

---

## Notes

- If the kernel is linear ($K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}$), Kernel K-means reduces to standard K-means.
- Using nonlinear kernels (e.g., Gaussian/RBF), Kernel K-means can find clusters with nonlinear boundaries.
- All computations are performed using kernel values; the explicit mapping $\phi(\mathbf{x})$ is never needed.

---

## Example: Gaussian Kernel

With the Gaussian (RBF) kernel:
$$
K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\sigma^2}\right)
$$
Kernel K-means can separate concentric or otherwise nonlinearly separable clusters.

---

# Kernel Regression

## Review: Linear Regression

**Regression** is a fundamental supervised learning task where the goal is to predict a continuous response variable $Y$ from one or more predictor (independent) variables $X_1, X_2, \ldots, X_d$. The aim is to learn a regression function $f$ such that:
$$
Y = f(X_1, X_2, \ldots, X_d) + \varepsilon
$$
where $\varepsilon$ is a random error term.

In **linear regression**, $f$ is assumed to be a linear function of the predictors:
$$
f(\mathbf{X}) = b + w_1 X_1 + w_2 X_2 + \cdots + w_d X_d = b + \mathbf{w}^T \mathbf{X}
$$
The parameters $b$ (bias) and $\mathbf{w}$ (weights) are estimated from data by minimizing the sum of squared errors (SSE):
$$
SSE = \sum_{i=1}^n (y_i - b - \mathbf{w}^T \mathbf{x}_i)^2
$$

**Ridge regression** adds an $L_2$ regularization term to prevent overfitting:
$$
J(\mathbf{w}) = \sum_{i=1}^n (y_i - b - \mathbf{w}^T \mathbf{x}_i)^2 + \alpha \|\mathbf{w}\|^2
$$

However, linear regression can only model linear relationships between predictors and response. For complex, nonlinear relationships, its performance is limited.

---

## Kernel Regression: Motivation

**Kernel regression** extends linear regression to handle nonlinear relationships by leveraging the **kernel trick**. The idea is to implicitly map the input data into a high-dimensional feature space where a linear model can fit complex, nonlinear patterns.

- Let $\phi(\mathbf{x})$ be a mapping from the input space to a feature space.
- Instead of explicitly computing $\phi(\mathbf{x})$, we use a kernel function $K(\mathbf{x}, \mathbf{z}) = \langle \phi(\mathbf{x}), \phi(\mathbf{z}) \rangle$.

---

## Kernel Ridge Regression: Formulation

Given training data $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$, kernel ridge regression solves:
$$
\min_{\mathbf{w}} \sum_{i=1}^n (y_i - \mathbf{w}^T \phi(\mathbf{x}_i))^2 + \alpha \|\mathbf{w}\|^2
$$

The solution can be expressed in terms of the kernel matrix $\mathbf{K}$, where $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$:
1. Compute the kernel matrix $\mathbf{K} \in \mathbb{R}^{n \times n}$.
2. Compute the coefficients:
   $$
   \mathbf{c} = (\mathbf{K} + \alpha \mathbf{I})^{-1} \mathbf{y}
   $$
3. The predicted response for a new point $\mathbf{z}$ is:
   $$
   \hat{y} = \sum_{i=1}^n c_i K(\mathbf{z}, \mathbf{x}_i)
   $$

**Key points:**
- The model is nonlinear in the original input space, but linear in the feature space.
- All computations are performed using kernel values; the explicit mapping $\phi(\mathbf{x})$ is never needed.
- Common kernels: linear, polynomial, Gaussian (RBF), etc.

---

## Kernel Regression Algorithm (Pseudocode)

1. **Input:** Training data $\{\mathbf{x}_i, y_i\}_{i=1}^n$, kernel function $K$, regularization parameter $\alpha$.
2. **Compute** the kernel matrix $\mathbf{K}$ with $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$.
3. **Compute** $\mathbf{c} = (\mathbf{K} + \alpha \mathbf{I})^{-1} \mathbf{y}$.
4. **Prediction:** For a new input $\mathbf{z}$, compute $\hat{y} = \sum_{i=1}^n c_i K(\mathbf{z}, \mathbf{x}_i)$.

---

## Example

Suppose the relationship between $X$ and $Y$ is nonlinear (e.g., quadratic). Linear regression will not fit the data well. By using a quadratic or Gaussian kernel, kernel regression can capture the nonlinear pattern, resulting in a much better fit and lower error.

---

## Summary

- **Kernel regression** generalizes linear regression to nonlinear settings using the kernel trick.
- It enables fitting complex relationships without explicitly mapping data to high-dimensional spaces.
- The choice of kernel and regularization parameter $\alpha$ are crucial for model performance.

---

## Kernel Regression

**Kernel regression** is a non-parametric technique that extends linear regression to model nonlinear relationships between input variables and the response. Unlike kernel ridge regression, which is a regularized version of linear regression in feature space, kernel regression directly estimates the regression function using a weighted average of observed responses, where the weights are determined by a kernel function measuring similarity.

---

### Nadaraya-Watson Kernel Regression

The most common form of kernel regression is the **Nadaraya-Watson estimator**. Given training data $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$, the predicted value for a new input $\mathbf{z}$ is:
$$
\hat{y}(\mathbf{z}) = \frac{\sum_{i=1}^n K(\mathbf{z}, \mathbf{x}_i) y_i}{\sum_{i=1}^n K(\mathbf{z}, \mathbf{x}_i)}
$$
where $K(\mathbf{z}, \mathbf{x}_i)$ is a kernel function (e.g., Gaussian kernel) that measures the similarity between $\mathbf{z}$ and $\mathbf{x}_i$.

- If $\mathbf{z}$ is close to $\mathbf{x}_i$, $K(\mathbf{z}, \mathbf{x}_i)$ is large, so $y_i$ has more influence on the prediction.
- If $\mathbf{z}$ is far from $\mathbf{x}_i$, $K(\mathbf{z}, \mathbf{x}_i)$ is small, so $y_i$ has less influence.

**Common kernels:**
- **Gaussian (RBF):** $K(\mathbf{z}, \mathbf{x}_i) = \exp\left(-\frac{\|\mathbf{z} - \mathbf{x}_i\|^2}{2h^2}\right)$
- **Polynomial:** $K(\mathbf{z}, \mathbf{x}_i) = (\mathbf{z}^T \mathbf{x}_i + c)^d$

---

### Example: Kernel Regression in Python

Below is a simple example of Nadaraya-Watson kernel regression using the Gaussian kernel:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic nonlinear data
np.random.seed(0)
X = np.linspace(-3, 3, 50)
y = np.sin(X) + 0.3 * np.random.randn(50)

# Define the Gaussian kernel function
def gaussian_kernel(x, xi, h):
    return np.exp(-0.5 * ((x - xi) / h) ** 2)

# Nadaraya-Watson kernel regression estimator
def kernel_regression(x_query, X, y, h):
    weights = gaussian_kernel(x_query[:, None], X[None, :], h)
    weights_sum = np.sum(weights, axis=1)
    y_pred = np.dot(weights, y) / weights_sum
    return y_pred

# Predict on a grid
X_test = np.linspace(-3, 3, 200)
h = 0.5  # bandwidth parameter
y_pred = kernel_regression(X_test, X, y, h)

# Plot
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, y_pred, color='red', label='Kernel Regression')
plt.title('Nadaraya-Watson Kernel Regression')
plt.legend()
plt.show()
```

**Explanation:**
- The kernel regression estimator computes a weighted average of the observed $y_i$ values, where the weights depend on the distance between the query point and each training point.
- The bandwidth parameter $h$ controls the smoothness of the fit: smaller $h$ leads to a more flexible fit, larger $h$ leads to a smoother fit.

---

### Key Points

- **Kernel regression** is non-parametric: it does not assume a specific form for the regression function.
- It can model complex, nonlinear relationships.
- The choice of kernel and bandwidth parameter $h$ is crucial for performance.
- Unlike kernel ridge regression, kernel regression does not involve solving a linear system or regularization.

---

**References:**
- Mohammad Zaki & Wagner Meira Jr., *Data Mining and Machine Learning: Fundamental Concepts and Algorithms*
- See also: [Kernel Trick](./Kernel-Trick.md)

## References

- Mohammad Zaki & Wagner Meira Jr., *Data Mining and Machine Learning: Fundamental Concepts and Algorithms*
- See also: [Kernel Trick](./Kernel-Trick.md)