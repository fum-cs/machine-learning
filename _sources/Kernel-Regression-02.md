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
## Kernel Regression

**Kernel regression** is a non-parametric technique that extends linear regression to model nonlinear relationships between input variables and the response. Unlike kernel ridge regression, which is a regularized version of linear regression in feature space, kernel regression directly estimates the regression function using a weighted average of observed responses, where the weights are determined by a kernel function measuring similarity.

---

## Kernel Regression: Primal and Dual Form

Kernel regression can be understood from the perspective of the **primal** and **dual** forms, similar to other kernelized algorithms. This approach provides insight into how the kernel trick allows us to perform nonlinear regression without explicitly mapping data to a high-dimensional feature space.

---

### Primal Form

In the **primal form** of (linear) regression, we seek a function of the form:
$$
f(\mathbf{x}) = \mathbf{w}^T \phi(\mathbf{x})
$$
where $\phi(\mathbf{x})$ is a (possibly nonlinear) mapping from the input space to a feature space, and $\mathbf{w}$ is the weight vector to be learned.

Given training data $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$, the objective (for least squares regression) is:
$$
\min_{\mathbf{w}} \sum_{i=1}^n (y_i - \mathbf{w}^T \phi(\mathbf{x}_i))^2
$$

---

### Dual Form

The **dual form** expresses the solution as a linear combination of the mapped training points:
$$
f(\mathbf{x}) = \sum_{i=1}^n \alpha_i \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}) \rangle = \sum_{i=1}^n \alpha_i K(\mathbf{x}_i, \mathbf{x})
$$
where $K(\mathbf{x}_i, \mathbf{x})$ is the kernel function, and $\alpha_i$ are the dual coefficients.

To find the optimal $\boldsymbol{\alpha}$, we solve:
$$
\min_{\boldsymbol{\alpha}} \| \mathbf{y} - \mathbf{K} \boldsymbol{\alpha} \|^2
$$
where $\mathbf{K}$ is the $n \times n$ kernel matrix with entries $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$.

The solution is:
$$
\boldsymbol{\alpha} = (\mathbf{K})^{-1} \mathbf{y}
$$
(In practice, a regularization term $\lambda \mathbf{I}$ is often added for numerical stability.)

---

### Prediction

Given a new input $\mathbf{z}$, the prediction is:
$$
\hat{y}(\mathbf{z}) = \sum_{i=1}^n \alpha_i K(\mathbf{x}_i, \mathbf{z})
$$

---

### Summary Table

| Form   | Model Representation                      | Solution                        | Prediction                        |
|--------|-------------------------------------------|----------------------------------|------------------------------------|
| Primal | $f(\mathbf{x}) = \mathbf{w}^T \phi(\mathbf{x})$ | $\mathbf{w} = \ldots$ (in feature space) | $f(\mathbf{z}) = \mathbf{w}^T \phi(\mathbf{z})$ |
| Dual   | $f(\mathbf{x}) = \sum_i \alpha_i K(\mathbf{x}_i, \mathbf{x})$ | $\boldsymbol{\alpha} = (\mathbf{K})^{-1} \mathbf{y}$ | $f(\mathbf{z}) = \sum_i \alpha_i K(\mathbf{x}_i, \mathbf{z})$ |

---

### Example: Dual Kernel Regression in Python

```python
import numpy as np

# Training data
X = np.linspace(-3, 3, 30)[:, None]
y = np.sin(X).ravel() + 0.1 * np.random.randn(30)

# Gaussian kernel function
def gaussian_kernel(x1, x2, sigma=1.0):
    return np.exp(-0.5 * ((x1 - x2.T) ** 2) / sigma**2)

# Compute kernel matrix
K = gaussian_kernel(X, X, sigma=0.5)

# Regularization for numerical stability
lambda_reg = 1e-3
alpha = np.linalg.solve(K + lambda_reg * np.eye(len(X)), y)

# Prediction for new points
X_test = np.linspace(-3, 3, 100)[:, None]
K_test = gaussian_kernel(X, X_test, sigma=0.5)
y_pred = np.dot(alpha, K_test)

# Plot
import matplotlib.pyplot as plt
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, y_pred, color='red', label='Kernel Regression (Dual)')
plt.legend()
plt.show()
```

---

### Key Points

- The **primal form** works in the (possibly high-dimensional) feature space.
- The **dual form** uses only kernel evaluations and avoids explicit computation in feature space.
- The kernel trick enables nonlinear regression using only inner products in the input space.

---

# Kernel Regression using Primal-Dual Form

## Introduction
Kernel regression is a method that enables nonlinearity in regression models by mapping input data into a higher-dimensional space using kernel functions.

## Mathematical Background
### Linear Regression (Primal Form)
Given a dataset with input **X** and output **y**, the linear regression model is:
$$ b = (A^T A + \lambda I)^{-1} A^T y $$
where **A** represents the feature matrix, and **λ** is a regularization parameter.

### Dual Form of Linear Regression
Using the dual formulation, the model can be rewritten as:
$$ b = A^T (\lambda I + AA^T)^{-1} y $$
This expresses **b** in terms of the training samples rather than the feature columns.

## Kernel Trick
To introduce nonlinearity, we replace the inner product **⟨x, x'⟩** with a kernel function **k(x, x')**, leading to:
$$ y = \sum_{n=1}^{N} \alpha_n k(x, x_n) $$
where **α_n** are the learned coefficients.

## Common Kernel Functions
- **Linear Kernel**: \( k(x, x') = x^T x' \)
- **Polynomial Kernel**: \( k(x, x') = (x^T x' + c)^d \)
- **Radial Basis Function (RBF) Kernel**: \( k(x, x') = \exp(-\gamma \|x - x'\|^2) \)

## Conclusion
Kernel regression allows for flexible, nonlinear modeling by leveraging kernel functions in the dual formulation. This approach is widely used in machine learning applications such as support vector machines and Gaussian processes.



**References:**
- [Lecture03_kernel.pdf, Purdue University](https://engineering.purdue.edu/ChanGroup/ECE595/files/Lecture03_kernel.pdf)
- Mohammad Zaki & Wagner Meira Jr., *Data Mining and Machine Learning: Fundamental Concepts and Algorithms*


## References

- Mohammad Zaki & Wagner Meira Jr., *Data Mining and Machine Learning: Fundamental Concepts and Algorithms*
- See also: [Kernel Trick](./Kernel-Trick.md)