# Selected Problems: Chapter 2 (Bayesian Decision Theory)

Based on Duda, Hart & Stork, "Pattern Classification" (2nd Ed.)

Here is a selection of problems relevant to the core concepts of Bayesian decision making, discriminant functions, classification with Gaussian distributions, and error analysis.

---

## Problem 2.2

Consider two one-dimensional density functions $p(x|\omega_1)$ and $p(x|\omega_2)$ shown below.

*[Insert Figure for Problem 2.2 here - showing two overlapping uniform distributions]*

(a) Find the value $x^*$ that minimizes the probability of error $P(\text{error})$ if the prior probabilities are $P(\omega_1) = P(\omega_2) = 1/2$. Describe the optimal decision rule.
(b) Calculate the minimum probability of error $P(\text{error})$.
(c) Repeat parts (a) and (b) for the case $P(\omega_1) = 2/3$ and $P(\omega_2) = 1/3$.

---

## Problem 2.3

Let $x$ be a continuous random variable with distributions conditioned on the state of nature $\omega_i$:
$$p(x|\omega_1) = \begin{cases} \frac{1}{3} e^{-x/3} & x \ge 0 \\ 0 & x < 0 \end{cases}$$
$$p(x|\omega_2) = \begin{cases} \frac{1}{5} e^{-x/5} & x \ge 0 \\ 0 & x < 0 \end{cases}$$
Assume $P(\omega_1) = P(\omega_2) = 1/2$.

(a) Find the decision boundary $x^*$ that minimizes the probability of error.
(b) Calculate the minimum probability of error.

---

## Problem 2.4

Consider a two-category classification problem with the following loss function:

| True Class | Decision $\alpha_1$ (decide $\omega_1$) | Decision $\alpha_2$ (decide $\omega_2$) |
| :--------- | :----------------------------------- | :----------------------------------- |
| $\omega_1$ | $\lambda_{11} = 0$                   | $\lambda_{12} = 1$                   |
| $\omega_2$ | $\lambda_{21} = 5$                   | $\lambda_{22} = 0$                   |

Let the class-conditional densities be uniform:
$$p(x|\omega_1) = \begin{cases} 1 & 0 \le x \le 1 \\ 0 & \text{otherwise} \end{cases}$$
$$p(x|\omega_2) = \begin{cases} 1/2 & 0 \le x \le 2 \\ 0 & \text{otherwise} \end{cases}$$
Assume $P(\omega_1) = P(\omega_2) = 1/2$.

(a) Determine the decision rule that minimizes the Bayes risk.
(b) Calculate the resulting minimum Bayes risk.

---

## Problem 2.5

Show that for minimum error rate classification, the decision boundary where $P(\omega_1|\mathbf{x}) = P(\omega_2|\mathbf{x})$ corresponds to the boundary where the likelihood ratio equals the ratio of priors:
$$\frac{p(\mathbf{x}|\omega_1)}{p(\mathbf{x}|\omega_2)} = \frac{P(\omega_2)}{P(\omega_1)}$$

---

## Problem 2.6

Consider a fish classification problem based on two discrete features: lightness ($x_1$) and thickness ($x_2$). Assume there are two types of fish, salmon ($\omega_1$) and sea bass ($\omega_2$), with equal prior probabilities $P(\omega_1) = P(\omega_2) = 1/2$. The conditional probabilities $P(x_1, x_2 | \omega_i)$ are given in the tables below:

**Salmon ($\omega_1$)**

| $P(x_1, x_2|\omega_1)$ | $x_2$ = light | $x_2$ = medium | $x_2$ = dark |
| :--------------------- | :------------ | :------------- | :----------- |
| $x_1$ = thin           | 0.10          | 0.15           | 0.05         |
| $x_1$ = medium         | 0.15          | 0.25           | 0.10         |
| $x_1$ = thick          | 0.05          | 0.10           | 0.05         |
| **Sum**                | 0.30          | 0.50           | 0.20         |

**Sea Bass ($\omega_2$)**

| $P(x_1, x_2|\omega_2)$ | $x_2$ = light | $x_2$ = medium | $x_2$ = dark |
| :--------------------- | :------------ | :------------- | :----------- |
| $x_1$ = thin           | 0.05          | 0.10           | 0.10         |
| $x_1$ = medium         | 0.10          | 0.20           | 0.15         |
| $x_1$ = thick          | 0.05          | 0.15           | 0.10         |
| **Sum**                | 0.20          | 0.45           | 0.35         |

*(Verify sums equal 1.0 for each table)*

Classify the measurement $(x_1=\text{medium}, x_2=\text{light})$ using the minimum error rate criterion.

---

## Problem 2.8

Consider two $d$-dimensional Gaussian distributions $p(\mathbf{x}|\omega_i) \sim N(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$ for $i=1, 2$. Assume they have arbitrary means $\boldsymbol{\mu}_i$ and covariance matrices $\boldsymbol{\Sigma}_i$. Show that the decision boundary defined by $P(\omega_1|\mathbf{x}) = P(\omega_2|\mathbf{x})$ (or equivalently $g_1(\mathbf{x}) = g_2(\mathbf{x})$ using $g_i(\mathbf{x}) = \ln p(\mathbf{x}|\omega_i) + \ln P(\omega_i)$) results in a hyperquadric decision surface of the form:
$$\mathbf{x}^T \mathbf{W} \mathbf{x} + \mathbf{w}^T \mathbf{x} + w_0 = 0$$
Express $\mathbf{W}$, $\mathbf{w}$, and $w_0$ in terms of $\boldsymbol{\mu}_i$, $\boldsymbol{\Sigma}_i$, and $P(\omega_i)$. This is the general case (Case 3).

---

## Problem 2.9

Specialize the result from Problem 2.8 to the case where the covariance matrices are equal, $\boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}_2 = \boldsymbol{\Sigma}$. Show that the decision boundary becomes linear:
$$\mathbf{w}^T \mathbf{x} + w_0 = 0$$
Express $\mathbf{w}$ and $w_0$ in terms of $\boldsymbol{\mu}_i$, $\boldsymbol{\Sigma}$, and $P(\omega_i)$. This is Case 2.

---

## Problem 2.11

Specialize the result from Problem 2.9 further to the case where $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$ (i.e., features are statistically independent and have the same variance $\sigma^2$). Show that the decision boundary remains linear and express $\mathbf{w}$ and $w_0$. Furthermore, show that this linear boundary corresponds to the minimum Euclidean distance classifier if the priors $P(\omega_i)$ are equal. This is Case 1.

---

## Problem 2.13

The Mahalanobis distance from a feature vector $\mathbf{x}$ to a mean vector $\boldsymbol{\mu}_i$, using the covariance matrix $\boldsymbol{\Sigma}_i$, is defined as:
$$r_i^2 = (\mathbf{x} - \boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{x} - \boldsymbol{\mu}_i)$$
Show that the discriminant function for Gaussian distributions (Case 3, from Problem 2.8) can be written in terms of Mahalanobis distances:
$$g_i(\mathbf{x}) = -\frac{1}{2} r_i^2 - \frac{d}{2} \ln(2\pi) - \frac{1}{2} \ln |\boldsymbol{\Sigma}_i| + \ln P(\omega_i)$$
Discuss how this simplifies for Case 2 ($\boldsymbol{\Sigma}_i = \boldsymbol{\Sigma}$) and Case 1 ($\boldsymbol{\Sigma}_i = \sigma^2 \mathbf{I}$).

---

## Problem 2.14

Consider a 1D, two-category classification problem with $p(x|\omega_1) \sim N(0, 1)$ and $p(x|\omega_2) \sim N(1, 1)$, and equal priors $P(\omega_1) = P(\omega_2) = 1/2$.
(a) Find the decision boundary $x^*$ that minimizes the probability of error.
(b) Calculate the minimum probability of error in terms of