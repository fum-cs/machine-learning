# Appendix: Bayes Decision Theory — Discrete Features 

## Introduction

In the general formulation of Bayes Decision Theory, we aim to minimize the probability of error (or more generally, the expected risk) by assigning a feature vector **x** to the class $\omega_i$ that maximizes the posterior probability $P(\omega_i | \mathbf{x})$. Using Bayes' theorem, this is often equivalent to maximizing the product of the likelihood $P(\mathbf{x} | \omega_i)$ and the prior probability $P(\omega_i)$:

$$
\text{Choose } \omega_i \text{ such that } P(\mathbf{x} | \omega_i) P(\omega_i) \ge P(\mathbf{x} | \omega_j) P(\omega_j) \text{ for all } j \neq i
$$

While the framework remains the same whether the features are continuous or discrete, the nature of the features significantly impacts how we model and estimate the class-conditional probability density (or probability mass function) $P(\mathbf{x} | \omega_i)$. This section focuses on the case where the feature vector $\mathbf{x} = [x_1, x_2, ..., x_d]^T$ consists of *discrete* features.

## Challenges with Discrete Features

When features $x_j$ are discrete, $\mathbf{x}$ can take on a finite number of possible values. Let $V_j$ be the number of possible values for feature $x_j$. Then the total number of possible feature vectors $\mathbf{x}$ is $V = \prod_{j=1}^{d} V_j$.

Estimating $P(\mathbf{x} | \omega_i)$ directly involves estimating the probability for each possible configuration of $\mathbf{x}$ for each class $\omega_i$.

*   **Curse of Dimensionality:** If the dimensionality $d$ is large, or if the features $x_j$ can take many values ($V_j$ is large), the total number of possible vectors $V$ becomes enormous.
*   **Data Sparsity:** To get reliable estimates for $P(\mathbf{x} | \omega_i)$ for every possible $\mathbf{x}$, we would need a vast amount of training data. Many possible feature vectors might not appear even once in the training set for a given class, leading to zero probability estimates, which is problematic.

## Modeling $P(\mathbf{x} | \omega_i)$ for Discrete Features: The Naive Bayes Assumption

Due to the challenges above, directly estimating the full joint probability $P(\mathbf{x} | \omega_i) = P(x_1, x_2, ..., x_d | \omega_i)$ is often infeasible. A common and simplifying approach is to assume **conditional independence** of the features given the class.

### The Naive Bayes Assumption

The *Naive Bayes* assumption states that the features $x_j$ are independent of each other, given the class $\omega_i$. Mathematically:

$$
P(\mathbf{x} | \omega_i) = P(x_1, x_2, ..., x_d | \omega_i) \approx \prod_{j=1}^{d} P(x_j | \omega_i)
$$

This assumption dramatically simplifies the problem: instead of estimating one large joint probability table, we only need to estimate $d$ smaller probability distributions $P(x_j | \omega_i)$ for each class $\omega_i$.

### Parameter Estimation and Smoothing

The parameters $P(x_j=v | \omega_i)$ and the prior probabilities $P(\omega_i)$ are typically estimated from the training data using Maximum Likelihood Estimation (MLE) or smoothed versions (like Laplace smoothing) to avoid zero probabilities:

*   **Prior Probability:** $\hat{P}(\omega_i) = N_i / N$
*   **Class-Conditional Feature Probability (MLE):** $\hat{P}(x_j=v | \omega_i) = N_{ijv} / N_i$
*   **Class-Conditional Feature Probability (Laplace Smoothing):** $\hat{P}(x_j=v | \omega_i) = (N_{ijv} + \alpha) / (N_i + \alpha V_j)$

## Case Study: Independent Binary Features (Section 2.9.1)

A particularly important special case arises when all $d$ features are **binary**, i.e., $x_j \in \{0, 1\}$. This is common in areas like document classification (presence/absence of words) or medical diagnosis (presence/absence of symptoms).

Let's define the probability that feature $j$ is present (takes value 1) for class $\omega_i$ as:
$$
p_{ji} = P(x_j=1 | \omega_i) \quad \quad (Eq. 84 \text{ concept})
$$
Consequently, the probability that the feature is absent (takes value 0) is:
$$
P(x_j=0 | \omega_i) = 1 - p_{ji}
$$
We can write the class-conditional probability for a single feature $x_j$ using the Bernoulli formula:
$$
P(x_j | \omega_i) = p_{ji}^{x_j} (1 - p_{ji})^{1-x_j} \quad \quad (Eq. 85)
$$
This formula works whether $x_j=1$ or $x_j=0$.

Under the Naive Bayes assumption (conditional independence), the full class-conditional probability for the feature vector $\mathbf{x} = [x_1, ..., x_d]^T$ is:
$$
P(\mathbf{x} | \omega_i) = \prod_{j=1}^{d} P(x_j | \omega_i) = \prod_{j=1}^{d} p_{ji}^{x_j} (1 - p_{ji})^{1-x_j} \quad \quad (Eq. 86)
$$

### Deriving the Discriminant Function

To classify a new sample $\mathbf{x}$, we use the decision rule based on the discriminant functions $g_i(\mathbf{x})$. For convenience, we use the log-posterior (ignoring the constant $P(\mathbf{x})$ term):
$$
g_i(\mathbf{x}) = \log P(\mathbf{x} | \omega_i) + \log P(\omega_i) \quad \quad (Eq. 87 \text{ base})
$$
Substituting the expression for $P(\mathbf{x} | \omega_i)$:
$$
g_i(\mathbf{x}) = \log \left( \prod_{j=1}^{d} p_{ji}^{x_j} (1 - p_{ji})^{1-x_j} \right) + \log P(\omega_i)
$$
Using the properties of logarithms ($\log(ab) = \log a + \log b$ and $\log(a^b) = b \log a$):
$$
g_i(\mathbf{x}) = \sum_{j=1}^{d} \log \left( p_{ji}^{x_j} (1 - p_{ji})^{1-x_j} \right) + \log P(\omega_i)
$$
$$
g_i(\mathbf{x}) =g_i(\mathbf{x}) = \sum_{j=1}^{d} \left[ \log(p_{ji}^{x_j}) + \log((1 - p_{ji})^{1-x_j}) \right] + \log P(\omega_i)
$$

$$
g_i(\mathbf{x}) = \sum_{j=1}^{d} \left[ x_j \log p_{ji} + (1-x_j) \log (1 - p_{ji}) \right] + \log P(\omega_i) \quad \quad (Eq. 88 \text{ structure})$$
We can rearrange this expression to highlight its linear form with respect to the features $x_j$. Let's expand the term inside the summation:
$$x_j \log p_{ji} + (1-x_j) \log (1 - p_{ji}) = x_j \log p_{ji} + \log (1 - p_{ji}) - x_j \log (1 - p_{ji})$$
$$= x_j \left( \log p_{ji} - \log (1 - p_{ji}) \right) + \log (1 - p_{ji})$$
$$= x_j \log \left( \frac{p_{ji}}{1 - p_{ji}} \right) + \log (1 - p_{ji})$$
The term $\log \left( \frac{p_{ji}}{1 - p_{ji}} \right)$ is the **log-odds** or **logit** of the probability $p_{ji}$.

Substituting this back into the equation for $g_i(\mathbf{x})$:

$$g_i(\mathbf{x}) = \sum_{j=1}^{d} \left[ x_j \log \left( \frac{p_{ji}}{1 - p_{ji}} \right) + \log (1 - p_{ji}) \right] + \log P(\omega_i)$$
Separating the terms that depend on $x_j$ from those that do not:
$$g_i(\mathbf{x}) = \sum_{j=1}^{d} \underbrace{\left( \log \frac{p_{ji}}{1 - p_{ji}} \right)}_{w_{ji}} x_j + \underbrace{\left[ \sum_{k=1}^{d} \log (1 - p_{ki}) + \log P(\omega_i) \right]}_{w_{i0}} \quad \quad (\text{Eq. 89 form})$$
This shows that the discriminant function $g_i(\mathbf{x})$ is **linear** in the features $x_j$:
$$g_i(\mathbf{x}) = \mathbf{w}_i^T \mathbf{x} + w_{i0}$$
where:
*   The weight vector $\mathbf{w}_i$ has components $w_{ji} = \log \frac{p_{ji}}{1 - p_{ji}}$ (the log-odds for feature $j$ being 1 in class $i$). ($Eq. 90a$)
*   The bias term (or threshold) $w_{i0} = \sum_{k=1}^{d} \log (1 - p_{ki}) + \log P(\omega_i)$. ($Eq. 90b$, index updated)

**Key Result:** For binary features under the Naive Bayes assumption, the optimal Bayes discriminant function $g_i(\mathbf{x})$ is a linear discriminant function. This provides an interesting link between generative models (like Naive Bayes, which models $P(\mathbf{x} | \omega_i)$ and $P(\omega_i)$) and discriminative models (which directly model the decision boundary or discriminant functions, often assuming linearity).

The decision boundary between two classes, say $\omega_i$ and $\omega_k$, is found by setting $g_i(\mathbf{x}) = g_k(\mathbf{x})$, which leads to a linear equation in $\mathbf{x}$:
$$(\mathbf{w}_i - \mathbf{w}_k)^T \mathbf{x} + (w_{i0} - w_{k0}) = 0$$

### Estimation

The parameters $p_{ji} = P(x_j=1 | \omega_i)$ and $P(\omega_i)$ are estimated from the training data, often using MLE or MAP (e.g., with Laplace smoothing).
*   MLE: $\hat{p}_{ji} = \frac{\text{Number of times } x_j=1 \text{ in class } \omega_i}{\text{Total number of samples in class } \omega_i}$
*   MLE: $\hat{P}(\omega_i) = \frac{\text{Total number of samples in class } \omega_i}{\text{Total number of samples}}$
Smoothing is typically applied, especially for $p_{ji}$, to avoid issues with zero counts leading to log probabilities of $-\infty$.

## Conclusion

Handling discrete features in Bayesian decision theory often requires simplifying assumptions due to the curse of dimensionality. The Naive Bayes assumption (conditional independence of features given the class) is a common and effective approach.

*   It simplifies the estimation of class-conditional probabilities from $O(V)$ parameters to $O(d \times \bar{V})$ parameters (where $\bar{V}$ is the average number of values per feature).
*   In the specific case of **independent binary features**, the Naive Bayes classifier results in a **linear discriminant function**. This highlights a connection between generative probability modeling and linear classifiers.

While the independence assumption might seem overly simplistic ("naive"), Naive Bayes classifiers often perform surprisingly well in practice, particularly in domains like text classification.

- For further information see section 2.9 of Duda et al.