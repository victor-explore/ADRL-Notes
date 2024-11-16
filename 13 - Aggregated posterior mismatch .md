# Aggregated posterior mismatch

## Why did we not have such a problem in EM algorithm?

The Variational Autoencoder (VAE) architecture can be visualized as follows:

<div style="text-align: center;"><img src="24.jpg" alt="Image Description" width="800" height="auto"/></div>

Before expanding the KL divergence term, recall that:

latent variable $q_{\phi}(z|x)$ is the approximate posterior (i.e., how likely the latent variable $z$ is given the input $x$).
$p_{\theta}(z|x)$ is the true posterior (i.e., the actual distribution of $z$ given $x$ under the model).
The KL divergence measures how much $q_{\phi}(z|x)$ deviates from $p_{\theta}(z|x)$:


We can expand this KL divergence as:
$$ D_{KL}(q_{\phi}(z|x) \| p_{\theta}(z|x)) = \int q_{\phi}(z|x) \log \frac{q_{\phi}(z|x)}{p_{\theta}(z|x)} dz $$

Using Bayes' rule, $p_{\theta}(z|x) = \frac{p_{\theta}(x,z)}{p_{\theta}(x)}$:
$$ D_{KL}(q_{\phi}(z|x) \| p_{\theta}(z|x)) = \int q_{\phi}(z|x) \log \frac{q_{\phi}(z|x)}{p_{\theta}(x,z)/p_{\theta}(x)} dz $$

Simplifying the fraction:
$$ D_{KL}(q_{\phi}(z|x) \| p_{\theta}(z|x))= \int q_{\phi}(z|x) \log \frac{q_{\phi}(z|x)}{p_{\theta}(x,z)} p_{\theta}(x) dz $$

Using the properties of logarithms:
$$ D_{KL}(q_{\phi}(z|x) \| p_{\theta}(z|x))= \int q_{\phi}(z|x) [\log q_{\phi}(z|x) - \log p_{\theta}(x,z) + \log p_{\theta}(x)] dz $$

Since $\log p_{\theta}(x)$ is constant with respect to $z$:
$$ D_{KL}(q_{\phi}(z|x) \| p_{\theta}(z|x))= \log p_{\theta}(x) + \mathbb{E}_{q_{\phi}(z|x)}\left[\log \frac{q_{\phi}(z|x)}{p_{\theta}(x,z)}\right] $$

Using the joint probability decomposition $p_{\theta}(x,z) = p_{\theta}(x)p_{\theta}(z)$:
$$ D_{KL}(q_{\phi}(z|x) \| p_{\theta}(z|x))= \log p_{\theta}(x) + \int q_{\phi}(z|x) \log \frac{q_{\phi}(z|x)}{p_{\theta}(x)p_{\theta}(z)} dz $$

Rearranging the fraction inside the expectation:
$$ D_{KL}(q_{\phi}(z|x) \| p_{\theta}(z|x))= \log p_{\theta}(x) - \mathbb{E}_{q_{\phi}(z|x)}\left[\log \frac{p_{\theta}(x,z)}{q_{\phi}(z|x)}\right] $$

By definition of the evidence lower bound (ELBO):
$$ D_{KL}(q_{\phi}(z|x) \| p_{\theta}(z|x))= \log p_{\theta}(x) - F_{\theta}(q_{\phi}) $$

Where $F_{\theta}(q_{\phi})$ is the evidence lower bound (ELBO) we derived earlier.


$$ F_{\theta}(q_{\phi}) = \log p_{\theta}(x) - D_{KL}(q_{\phi}(z|x) \| p_{\theta}(z|x)) $$

Recall that in the EM algorithm, we made $q_{\phi}(z|x) = p_{\theta}(z|x)$ in the E-step. And found $\theta$ by maximizing $F_{\theta}(q_{\phi})$.

However here we cannot get $q_{\phi}(z|x) = p_{\theta}(z|x)$. Hence we cannot set the KL divergence to zero, so we choose a family of distributions $q_{\phi}(z|x)$ and optimize $\phi$ to minimize the KL divergence between $q_{\phi}(z|x)$ and $p_{\theta}(z|x)$.

## What is the problem of Aggregated posterior mismatch in VAE?

Consider the distribution called aggregated posterior defined as:

$$ q_{\phi}(z) = \int q_{\phi}(z|x) p(x) dx $$

For every $x$, we get a conditional distribution. To get the aggregated posterior, we marginalize over all $x$.

The aggregated posterior $q_{\phi}(z)$ represents the overall distribution of the latent variable $z$ across all data points $x$. Ideally, this should match the model's prior distribution $p_{\theta}(z)$, which we assume before seeing any data.

However, when there is a mismatch between $q_{\phi}(z|x)$ and $p_{\theta}(z|x)$ for some data points, it leads to a discrepancy:

$$ q_{\phi}(z) = \int q_{\phi}(z|x) p(x) dx \neq \int p_{\theta}(z|x) p(x) dx = p_{\theta}(z) $$

This discrepancy between $q_{\phi}(z)$ and $p_{\theta}(z)$ is called the aggregated posterior mismatch. It indicates that the learned latent representations are not aligned with the true prior, which can result in poorer generalization and generation quality in VAEs.