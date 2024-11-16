# Wasserstein GANs (WGANs)

## Perfect Discriminator theorem

The Perfect Discriminator theorem states that if two distributions $p_x$ and $q_x$ have support on two disjoint subsets $M$ and $P$ respectively, there always exists a discriminator $D^*: x \rightarrow [0,1]$ that has accuracy 1:

$$\forall x \in M \cup P, D^*(x) = \begin{cases} 1 & \text{if } x \in M \\ 0 & \text{if } x \in P \end{cases}$$

This discriminator achieves perfect classification, as it correctly identifies the origin of every sample with 100% accuracy.

The Perfect Discriminator theorem has significant implications for GANs:

1. It can lead to training instability and convergence issues, as the generator receives no useful gradient information when the discriminator achieves perfect accuracy.
2. It may result in mode collapse, where the generator produces a limited set of samples rather than capturing the full diversity of the real data distribution.
3. This theorem motivates the use of regularization techniques and alternative GAN formulations to ensure meaningful learning can occur.

## Wasserstein GANs (WGANs)

Recall in the original formulation of GANs, we have:

$$\theta^* = \underset{\theta}{\argmin} \, D_f(P_{data} \parallel P_{generator})$$

In Wasserstein GANs, we replace the F-divergence with the Wasserstein distance of order 1:

$$\theta^* = \underset{\theta}{\argmin} \, W_1(P_{data} \parallel P_{generator})$$

$$\theta^* = \underset{\theta}{\argmin} \, \inf_{\gamma \in \Gamma(P_{data},P_{generator})} \mathbb{E}_{(x,y) \sim \gamma} [d(x,y)]$$

use Kantorovich-Rubinstein duality to get (*derivation skipped*):

$$\theta^* = \underset{\theta}{\argmin} \, \sup_{f \in \text{Lip}_1} \mathbb{E}_{x \sim P_{data}}[f(x)] - \mathbb{E}_{x \sim P_{generator}}[f(x)] \qquad \qquad (eqn - 1)$$ 

where $f$ is a 1-Lipschitz function.

## What is k - Lipschitz?

K lipschitz is defined as: 
$$|f(x) - f(y)| \leq K |x-y|$$

Hence 1-Lipschitz is defined as: 
$$|f(x) - f(y)| \leq 1 |x-y|$$

## How to enforce 1-Lipschitz constraint?
We use neural networks with parameter $\omega$ to model $f$, hence we can rewrite (1) as:

$$\theta^* = \underset{\theta}{\argmin} \, \max_{\omega} \mathbb{E}_{x \sim P_{data}}[f(x)] - \mathbb{E}_{x \sim P_{generator}}[f(x)]$$

<div style="text-align: center;"><img src="11.jpg" alt="Image Description" width="500" height="auto"/></div>

We enforce the 1-Lipschitz constraint on $f$ ie discriminator network by:
- Gradient clipping: $|| \nabla f || \leq 1$
- Weight normalization: $|| \omega || = 1$