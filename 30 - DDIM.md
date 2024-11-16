# DDIM (Denoising Diffusion Implicit Models)
## Motivation
During inference, we often had to sequentially traverse through 1000 time steps because of the Markov property assumption, which can be computationally intensive. 

Is it possible to give up the Markov property assumption so that we can train the same way but sample faster - not have to traverse through 1000 time steps?

## Recall the DDPM
Forward process:
$$q(x_t | x_0) = \int q(x_{t:1} | x_0) dx_{1:(t-1)}$$
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_0, (1 - \alpha_t) \mathbb{I})$$
$$\Rightarrow x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(.)$$
Note that here $\alpha_t$ is $\bar{\alpha}$ that we studied in DDPM.


Reverse process:
$$p_\theta(x_{0:T}) = p_\theta(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t)$$

The training loss function $L_T$ is given by:
$$L_T = \sum_{t=1}^T \mathbb{E}_{x_0,\epsilon} \left[\|\epsilon - \epsilon_\theta(\sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon, t)\|^2\right]$$

where:
- $\epsilon_\theta$ is the neural network that predicts the noise
- $\epsilon$ is the random noise added during training
- $x_0$ is the original data point
- $\alpha_t$ controls the noise schedule
- $\sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon = x_t$ is the forward process

This loss depends on the marginal distribution:
$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_0, (1-\alpha_t)\mathbb{I})$$

But importantly, it does not depend on the specific choice of intermediate distributions:
$$q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})$$

We can potentially modify the intermediate distributions while keeping the same marginal distribution to enable faster sampling

This observation opens up possibilities for accelerated sampling methods that don't require going through all timesteps sequentially.

## Formulation of DDIM
Let's formulate DDIM by deriving an alternative forward process that:
1. Maintains the same marginal distribution $q(x_t|x_0)$ as DDPM
2. Has different intermediate distributions $q(x_{1:T}|x_0)$

The key idea is to find a new forward process such that:

1. The marginal distribution matches DDPM:
   $$q_\sigma(x_t|x_0) = q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_0, (1-\alpha_t)\mathbb{I})$$

2. But the intermediate distributions $\prod_{t=1}^T q_\sigma(x_t|x_{t-1})$ can be different from DDPM

This is possible because the same marginal distribution can arise from different joint distributions, as long as they integrate to give the same $q(x_t|x_0)$.

## Deriving DDIM
We can write the joint distribution $q_\sigma(x_{t}, x_{t-1}|x_0)$ using Bayes rule as:
   $$q_\sigma(x_t|x_{t-1}, x_0) = \frac{q_\sigma(x_{t-1}|x_t,x_0)q_\sigma(x_t|x_0)}{q_\sigma(x_{t-1}|x_0)}$$

We assume $q_\sigma(x_{t-1}|x_t,x_0)$ follows a Gaussian distribution:

$$q_\sigma(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t,x_0), \sigma^2\mathbb{I})$$

where $\tilde{\mu}_t$ = $\sqrt{\alpha_{t-1}}x_0 + \sqrt{1-\alpha_{t-1}^2-\sigma^2}\left(\frac{x_t-\sqrt{\alpha_t}x_0}{\sqrt{1-\alpha_t}}\right)$

*derivation not done in class*

This formulation:
1. Maintains the same marginal distribution as DDPM
2. Introduces a new parameter $\sigma$ that controls the stochasticity
3. When $\sigma^2 = 1-\alpha_{t-1}/\alpha_t$, recovers the original DDPM
4. When $\sigma = 0$, gives a deterministic process (DDIM)
   
## How do we do inference with DDIM?
For inference with DDIM, we follow these steps:
1. First, we note that $x_0$ can be expressed in terms of $x_t$ and $\epsilon$ as:
   $$x_0 = \frac{x_t - \sqrt{1-\alpha_t}\epsilon}{\sqrt{\alpha_t}}$$

2. We can predict $x_0$ using a neural network $f_\theta^{(t)}(x_t) = \frac{x_t - \sqrt{1-\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\alpha_t}}$

3. Then, we can sample $x_{t-1}$ using:
$$p_\theta(x_{t-1}|x_t) = \begin{cases}
\mathcal{N}(f_\theta^{(t)}(x_t), \sigma_t^2\mathbb{I}) & \text{if } t=1 \\
q_\sigma(x_{t-1}|x_t, f_\theta^{(t)}(x_t)) & \text{otherwise}
\end{cases}$$

The iterations drop from 1000 to 20-30 steps, and the samples are of comparable quality to DDPM.