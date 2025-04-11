---

## Covariance Matrix in Bayesian Decision Theory

In [Bayesian decision theory(05.05-Bayesian-Decision-Theory.ipynb), the covariance matrix plays a crucial role in defining the **class-conditional probability density functions** for multivariate normal distributions. The probability density function for a multivariate normal distribution is given by:

$$p(\mathbf{x}|\omega_i) = \frac{1}{(2\pi)^{d/2}|\mathbf{\Sigma}_i|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_i)^T \mathbf{\Sigma}_i^{-1} (\mathbf{x} - \boldsymbol{\mu}_i)\right)$$

where:
- $\mathbf{x}$ is the feature vector.
- $\boldsymbol{\mu}_i$ is the mean vector for class $\omega_i$.
- $\mathbf{\Sigma}_i$ is the covariance matrix for class $\omega_i$.
- $d$ is the number of dimensions (features).

The covariance matrix $\mathbf{\Sigma}_i$ determines the shape and orientation of the probability distribution for class $\omega_i$. It captures both the variance of each feature (along the diagonal) and the covariance between pairs of features (off-diagonal elements).

---

## Mahalanobis Distance and Its Role

The **Mahalanobis distance** is a measure of the distance between a point $\mathbf{x}$ and a distribution with mean $\boldsymbol{\mu}$ and covariance matrix $\mathbf{\Sigma}$. It is defined as:

$$D_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

This distance is central to many multivariate statistical methods, including Bayesian decision theory for classification, because it accounts for the underlying structure (variances and correlations) of the data.

<!-- ![Mahalanobis Distance(img/cov-figure-2.png)

*Figure 2: Mahalanobis distance accounts for the covariance structure of the data, unlike Euclidean distance.* -->

---

## Bridging from 1D Standardization to Multivariate Mahalanobis Distance

To understand the intuition behind Mahalanobis distance, let's first recall the familiar concept of **standardization** for a single normal random variable.

**The Univariate Case (1 Dimension):**

Suppose we have a normally distributed random variable $X$ with mean $\mu$ and variance $\sigma^2$. To measure how far an observed value $x$ is from the mean *relative to the spread of the distribution*, we calculate the **z-score**:

$$z = \frac{x - \mu}{\sigma}$$

The z-score tells us how many standard deviations $x$ is away from the mean $\mu$. It effectively removes the scale ($\sigma$) from the measurement, allowing comparison across different normal distributions.

Now, let's look at the squared z-score:

$$z^2 = \left(\frac{x - \mu}{\sigma}\right)^2 = \frac{(x - \mu)^2}{\sigma^2} = (x - \mu) (\sigma^2)^{-1} (x - \mu)$$

This $z^2$ represents a **squared, scale-normalized distance** from the mean. Notice its structure: `(difference) * (inverse variance) * (difference)`. Importantly, this squared z-score is exactly what appears (up to a factor of -1/2) in the exponent of the univariate normal probability density function:

$$p(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{1}{2} \frac{(x - \mu)^2}{\sigma^2}\right) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{1}{2} z^2\right)$$

So, the probability density decreases exponentially with this squared normalized distance $z^2$.

**Generalizing to Multiple Dimensions:**

When we move from one dimension ($\mathbb{R}^1$) to multiple dimensions ($\mathbb{R}^d$), we face two key challenges that simple Euclidean distance ($\sqrt{\sum (x_i - \mu_i)^2}$) doesn't handle well:

1.  **Different Scales:** Each dimension (feature) might have a different variance. Euclidean distance treats all dimensions equally, effectively giving more weight to dimensions with larger variances.
2.  **Correlation:** Features might be correlated. This means the data cloud isn't necessarily spherical or axis-aligned; it might be stretched and rotated into an ellipsoid shape. Euclidean distance ignores these correlations.

**The Mahalanobis Solution:**

The Mahalanobis distance is designed to address these challenges by incorporating the **covariance matrix** $\mathbf{\Sigma}$, which captures *both* the variances of individual features (on its diagonal) and the covariances between features (off-diagonal elements).

Let's look at the **squared Mahalanobis distance**:

$$D_M^2(\vec{x}, \vec{\mu}) = (\vec{x} - \vec{\mu})^\mathsf{T} \mathbf{\Sigma}^{-1} (\vec{x} - \vec{\mu})$$

Now, compare its structure directly to the squared z-score:

*   The vector difference $(\vec{x} - \vec{\mu})$ is the multivariate analog of the scalar difference $(x - \mu)$.
*   The **inverse covariance matrix** $\mathbf{\Sigma}^{-1}$ is the multivariate analog of the **inverse variance** $(\sigma^2)^{-1}$. It plays a crucial role:
    *   It accounts for the different variances along each dimension.
    *   It accounts for the correlations between dimensions, effectively "de-correlating" or "whitening" the space.

Just as $z^2$ appeared in the exponent of the univariate normal PDF, the **squared Mahalanobis distance** $D_M^2$ is precisely the term appearing in the exponent of the multivariate normal probability density function:

$$p(\mathbf{x}|\boldsymbol{\mu}, \mathbf{\Sigma}) \propto \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right) = \exp\left(-\frac{1}{2} D_M^2(\mathbf{x}, \boldsymbol{\mu})\right)$$

**Conclusion:**

The Mahalanobis distance ($D_M = \sqrt{D_M^2}$) is the natural generalization of the standardized distance (z-score concept) to multiple dimensions. It provides a statistically meaningful measure of distance from the mean (center) of a multivariate distribution by properly accounting for the variances and correlations inherent in the data, as captured by the covariance matrix $\mathbf{\Sigma}$. It essentially measures distance in a transformed space where the data cloud is spherical and has unit variance in all directions.