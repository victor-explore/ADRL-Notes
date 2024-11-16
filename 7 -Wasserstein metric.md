# Wasserstein metric

## Definition

We can use Wasserstein metric in place of F-divergence to define a distance between two distributions $P$ and $Q$ both defined over the same space $X$.

The Wasserstein distance $W_p(P,Q)$ of order $p$ is defined as:

$$W_p(P,Q) = \left( \inf_{\gamma \in \Gamma(P,Q)} \mathbb{E}_{(x,y) \sim \gamma} [d(x,y)^p] \right)^{1/p}$$

where:

- $\Gamma(P,Q)$ is the set of all possible joint distributions of $(x,y)$ where $x,y \in X$ with marginals $x \sim P$ and $y \sim Q$ (also called couplings)
- $\gamma(x,y)$ is a particular joint distribution chosen out of all possible joint distributions in $\Gamma(P,Q)$ that minimizes $\mathbb{E}_{(x,y) \sim \gamma} [d(x,y)^p]$
- $d(x,y)$ is the metric (distance) between points $x$ and $y$ in the space $X$
- $p \geq 1$ is the order of the Wasserstein distance. Common choices are $p=1$ (first-order Wasserstein distance) and $p=2$ (second-order Wasserstein distance).

## Properties of Wasserstein metric
1. When $p=1$, the Wasserstein distance is equivalent to the Earth Mover's Distance (EMD), because it can be interpreted as the minimum amount of work required to transform one distribution into another.
2. When $p=2$, the Wasserstein distance is equivalent to the squared Euclidean distance.
3. The Wasserstein distance is always finite, unlike the F-divergence, which can be infinite.
4. The Wasserstein distance is a true metric, meaning it satisfies the triangle inequality.
5. When KL divergence is $0$ then Wasserstein distance is $0$, but not vice versa.

## Proof for 5th property - KL divergence is 0 then Wasserstein distance is 0
To prove that when the KL divergence is 0, the Wasserstein distance is also 0, we start with the definitions of both metrics.

**KL Divergence**: The Kullback-Leibler (KL) divergence between two distributions $P$ and $Q$ is defined as:
$$D_{KL}(P \parallel Q) = \int_{X} p(x) \log \left( \frac{p(x)}{q(x)} \right) dx$$
where $p(x)$ and $q(x)$ are the probability density functions of $P$ and $Q$ respectively.

**Wasserstein Distance**: The Wasserstein distance of order 1 between two distributions $P$ and $Q$ is defined as:
$$W_1(P,Q) = \inf_{\gamma \in \Gamma(P,Q)} \mathbb{E}_{(x,y) \sim \gamma} [d(x,y)]$$
where $\Gamma(P,Q)$ is the set of all possible joint distributions with marginals $P$ and $Q$.

Now, if $D_{KL}(P \parallel Q) = 0$, it implies that $P$ and $Q$ are identical almost everywhere, i.e., $p(x) = q(x)$ for almost all $x \in X$. This is because the KL divergence is zero if and only if the two distributions are the same.

Since $P$ and $Q$ are identical, the optimal coupling $\gamma$ in the definition of the Wasserstein distance will be such that $x = y$ almost surely. Therefore, the expected distance $d(x,y)$ will be zero.

Hence, $W_1(P,Q) = 0$.

This completes the proof that when the KL divergence is 0, the Wasserstein distance is also 0.
To prove that the vice versa is not true, i.e., when the Wasserstein distance is 0, the KL divergence is not necessarily 0, we can consider the following example:

Consider two distributions $P$ and $Q$ defined over the real line $\mathbb{R}$.

Let $P$ be a Dirac delta distribution centered at 0:
$$P(x) = \delta(x)$$

Let $Q$ be a uniform distribution over the interval $[-\epsilon, \epsilon]$ for some small $\epsilon > 0$:
$$Q(x) = \begin{cases} 
\frac{1}{2\epsilon} & \text{if } x \in [-\epsilon, \epsilon] \\
0 & \text{otherwise}
\end{cases}$$

Now, let's compute the Wasserstein distance $W_1(P, Q)$ and the KL divergence $D_{KL}(P \parallel Q)$.

**Wasserstein Distance**: The Wasserstein distance of order 1 between $P$ and $Q$ is given by:
$$W_1(P, Q) = \inf_{\gamma \in \Gamma(P, Q)} \mathbb{E}_{(x,y) \sim \gamma} [|x - y|]$$

Since $P$ is a Dirac delta distribution centered at 0, the optimal coupling $\gamma$ will pair the point mass at 0 in $P$ with the uniform distribution over $[-\epsilon, \epsilon]$ in $Q$. The expected distance is:
$$W_1(P, Q) = \int_{-\epsilon}^{\epsilon} |0 - x| \cdot \frac{1}{2\epsilon} \, dx = \int_{-\epsilon}^{\epsilon} \frac{|x|}{2\epsilon} \, dx$$

Splitting the integral at 0, we get:
$$W_1(P, Q) = \int_{0}^{\epsilon} \frac{x}{2\epsilon} \, dx + \int_{-\epsilon}^{0} \frac{-x}{2\epsilon} \, dx = \frac{1}{2\epsilon} \left( \int_{0}^{\epsilon} x \, dx + \int_{0}^{\epsilon} x \, dx \right) = \frac{1}{2\epsilon} \left( \frac{\epsilon^2}{2} + \frac{\epsilon^2}{2} \right) = \frac{\epsilon}{2}$$

As $\epsilon \to 0$, $W_1(P, Q) \to 0$.

**KL Divergence**: The KL divergence between $P$ and $Q$ is given by:
$$D_{KL}(P \parallel Q) = \int_{X} p(x) \log \left( \frac{p(x)}{q(x)} \right) dx$$
cannot be directly applied in this case because the Dirac delta measure is singular with respect to the Lebesgue measure (the measure used for the uniform distribution $Q$).

When two probability measures are singular (i.e., they are concentrated on disjoint sets), the KL divergence between them is infinite by definition. In our case:
- $P$ (Dirac delta) is concentrated at a single point {0}
- $Q$ (uniform distribution) is absolutely continuous with respect to Lebesgue measure
- Therefore, $P$ and $Q$ are singular measures

Thus, $D_{KL}(P \parallel Q) = \infty$.

Therefore, even though the Wasserstein distance $W_1(P, Q)$ can be made arbitrarily small by choosing a small $\epsilon$, the KL divergence $D_{KL}(P \parallel Q)$ is infinite.

This example demonstrates that the vice versa is not true: when the Wasserstein distance is 0, the KL divergence is not necessarily 0.

**Key Takeaway**: The Wasserstein distance captures the geometric difference between distributions, while KL divergence requires absolute continuity between the measures. When measures are singular, as in this case with a Dirac delta and a continuous distribution, the KL divergence is infinite regardless of how geometrically close the distributions might be.




