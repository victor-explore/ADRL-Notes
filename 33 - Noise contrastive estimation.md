# Noise Contrastive Estimation

## Idea
The core idea behind Noise Contrastive Estimation (NCE) is:

1. Given samples from a distribution, if you know how to distinguish them from noise samples, then you implicitly know the underlying distribution.

2. If you know how to tell what is noise from what is real data, you must have learned something about the true data distribution.

This provides an alternative way to learn probability distributions by transforming the density estimation problem into a binary classification problem between real and noise samples.

## Formulation

- Let the dataset be denoted as 
$$D = \{x_1, x_2, \ldots, x_t\}$$
where each element $x_i$ represents a sample from the true data distribution.

- Let a noise sample be denoted as
$$D_N = \{y_1, y_2, \ldots, y_t\}$$

- Our objective is to estimate the true data distribution $p_D(x)$ given the dataset $D$ and the noise samples $D_N$ which we represent as $p_\theta(x)$ the model's estimate of the data distribution.

- We cast it as a binary classification problem between real and noise samples.
  $$ U = \{(u_1,c_1), (u_2,c_2), \ldots, (u_{2t},c_{2t})\} $$
  where 
  - $u_i \in \{x_1, x_2, \ldots, x_t, y_1, y_2, \ldots, y_t\}$ and 
  - $c_i \in \{0, 1\}$ is a binary label indicating whether $u_i$ is a real data sample ($c_i = 1$) or a noise sample ($c_i = 0$).

- Now the posterior is given by:
  - $p(u_i | c_i = 1) = p_D(u_i)$ ie the data distribution
  - $p(u_i | c_i = 0) = p_N(u_i)$ ie the noise distribution that is a known distribution

- Assume that the priors are equal, ie $p(c_i = 1) = p(c_i = 0) = 0.5$ ie the data and noise samples are equally likely because we there are as many data samples as noise samples. Then:

  - Using Bayes' rule, we can write:
  $$p(c=1|u) = \frac{p(u|c=1)p(c=1)}{p(u|c=0)p(c=0) + p(u|c=1)p(c=1)}$$

  - Substituting the distributions and priors:
  $$p(c=1|u) = \frac{p_D(u)}{p_N(u) + p_D(u)}$$

  - This gives us a way to estimate the probability that a sample u comes from the real data distribution versus the noise distribution.

- We define
  - $G(u;\theta) = \log p_\theta(u) - \log p_N(u)$ which is just the difference between the estimated log probabilities of the data distribution and the noise distribution.
  - $p(c=1|u) = h_\theta(u) = \sigma(G(u;\theta)) = \frac{1}{1 + \exp(-G(u;\theta))}$ where $\sigma(s) = \frac{1}{1 + \exp(-s)}$ is the sigmoid function representing the probability that a sample u comes from the real data distribution versus the noise distribution.

- The likelihood function for the binary classification problem is:
  $$L(\theta) = \sum_{t=1}^{2T} c_t \log p(c_t=1|u_t;\theta) + (1-c_t)\log p(c_t=0|u_t;\theta)$$
  
  Which can be rewritten as:
  $$L(\theta) = \sum_{t=1}^{T} \log h_\theta(x_t) + \sum_{t=T}^{2T} \log(1-h_\theta(y_t))$$

  where:
  - $h_\theta(u) = p(c=1|u)$ is the classifier's prediction
  - The first sum is over real data samples
  - The second sum is over noise samples
  - $\theta$ are the parameters we want to learn
  - $L(\theta)$ is also known as NCE estimator

- From law of large numbers, we can write the expected value of the likelihood:
  $$J(\theta) = \frac{1}{2} \mathbb{E}_{p_{DN}} [\log h_\theta(x) + \log(1-h_\theta(y))]$$

  where:
  - The expectation is taken over the joint distribution of data and noise samples
  - $x$ follows the data distribution $p_D$
  - $y$ follows the noise distribution $p_N$

- After some algebraic manipulation, we can write:
  $$J(\theta) = \frac{1}{2} \mathbb{E} [\log r(log(p_\theta(x)) - log(p_N(x))) + \log(1-r(log(p_\theta(y)) - log(p_N(y)))]$$

  where:
  - $r(\cdot)$ is the sigmoid function
  - The first two terms correspond to real data samples
  - The last two terms correspond to noise samples

- Theorem(without proof): $J(\theta)$ attains when $p_\theta(x) = p_D(x)$ if $p_N(x)$ is chosen such that $p_N(x)$ is nonzero for all $x$ in the support of $p_D(x)$.
  <div style="text-align: center;"><img src="4.png" alt="Image Description" width="800" height="auto"/></div>