# KL Divergence - Kullback-Leibler Divergence

KL divergence is a measure of how one probability distribution differes from another probability distribution.

Mathematically, for two probability distributions $P(x)$ and $Q(x)$, the KL divergence from $Q$ to $P$ is defined as:

$D_{KL}(p || q) = Σ p(x) * log\left(\frac{p(x)}{q(x)}\right)$ for discrete distributions 

$D_{KL}(p || q) = ∫ p(x) * log\left(\frac{p(x)}{q(x)}\right) dx$ for continuous distributions  

where the sum/integral is over all possible events $x$. And $p(x)$ and $q(x)$ are the probability density functions of distributions $P(x)$ and $Q(x)$ respectively.

## Intuition

KL divergence is a measure of how one probability distribution diverges from another. It is a measure of the information lost when $Q$ is used to approximate $P$.

## Properties

- KL divergence is not symmetric: $D_{KL}(P || Q) ≠ D_{KL}(Q || P)$
- KL divergence is always non-negative: $D_{KL}(P || Q) ≥ 0$
- KL divergence is 0 if and only if $P$ and $Q$ are the same distribution

## Usefulness in Machine Learning - Minimizing KL Divergence is Equivalent to Maxmimizing Likelihood

For a typical ML problem, all we have are samples from the true distribution $P(x)$ ie $data = \{(x_i)\}_{i=1}^N$ where $x_i \in \mathbb{R}^d$ are iid samples from the true distribution $P(x)$. 

We do not know the true distribution $P(x)$ explicitly.

We try our best to estimate the true distribution $P(x)$ by $Q(x; \theta)$ where $\theta$ are the parameters of the model.

We need to know how well our model $Q(x; \theta)$ is performing. We can do this by calculating the KL divergence between the true distribution $P(x)$ and the estimated distribution $Q(x; \theta)$.

$$D_{KL}(P || Q) = ∫ p(x) * log\left(\frac{p(x)}{q(x; \theta)}\right) dx$$

$$D_{KL}(P || Q) = E_{x \sim p(x)}[log\left(\frac{p(x)}{q(x; \theta)}\right)]$$

$$D_{KL}(P || Q) = E_{x \sim p(x)}[log\left(p(x)\right)] - E_{x \sim p(x)}[log\left(q(x; \theta)\right)]$$

We are trying to find the parameters $\theta^*$ that minimize the KL divergence between $p(x)$ and $q(x; \theta)$.

$$\theta^* = \underset{\theta}{argmin} \ D_{KL}(p || q(x; \theta))$$

$$\theta^* = \underset{\theta}{argmin} \ E_{x \sim p(x)}[log\left(p(x)\right)] - E_{x \sim p(x)}[log\left(q(x; \theta)\right)]$$

because $E_{x \sim p(x)}[log\left(p(x)\right)]$ does not depend on $\theta$, we can ignore it.


$$\theta^* = \underset{\theta}{argmin} \ -E_{x \sim p(x)}[log\left(q(x; \theta)\right)]$$


$$\theta^* = \underset{\theta}{argmax} \ E_{x \sim p(x)}[log\left(q(x; \theta)\right)]$$

$E_{x \sim p(x)}[log\left(q(x; \theta)\right)]$ is called the **Expected Log Likelihood**,

By the law of large numbers, we can approximate the expected log likelihood by the average log likelihood of the data:

$$E_{x \sim p(x)}[log(q(x; \theta))] \approx \frac{1}{N} \sum_{i=1}^N log(q(x_i; \theta))$$

Therefore, our optimization problem becomes:

$$\theta^* = \underset{\theta}{argmax} \ \frac{1}{N} \sum_{i=1}^N log(q(x_i; \theta))$$

This is equivalent to maximizing the log likelihood of the data under the model $q(x; \theta)$.

$$\theta^* = \underset{\theta}{argmax} \ \frac{1}{N} \sum_{i=1}^N log(q(x_i; \theta))$$

hence $\theta$ is also called the **maximum log likelihood estimate**. 



