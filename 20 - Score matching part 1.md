# Score matching part 1
## Sampling as optimization

In probabilistic modeling, sampling can be viewed as an optimization problem where we aim to find the most probable sample from a probability distribution. This can be formulated mathematically as:

$$x^* = \mathop{\arg \max}_{x} p(x)$$  

Taking the log of the probability (which preserves the optimum due to monotonicity of log):

$$x^* = \mathop{\arg \max}_{x} \log p(x)$$

And converting to a minimization problem:

$$x^* = \mathop{\arg \min}_{x} - \log p(x)$$

This optimization problem can be solved using gradient descent, where we iteratively update our sample:

$$x_{t+1} = x_t + \alpha \cdot \nabla \log p(x_t)$$

where $\alpha$ is the learning rate that controls the size of each update step.

However, simple gradient descent can get stuck in local optima. A more robust approach is to use the Langevin dynamics equation:

$$x_{t+1} = x_t + \alpha \cdot \nabla_x \log p(x_t) + \sqrt{2\alpha} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$ 

The Langevin equation has three key components:
- The current state $x_t$
- A gradient term $\nabla_x \log p(x_t)$ that guides the sample toward higher probability regions
- A stochastic noise term $\sqrt{2\alpha} \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$ that helps explore the full distribution and escape local optima

This stochastic differential equation defines a Markov chain whose stationary distribution converges to our target distribution $p(x)$, allowing us to generate samples that accurately represent the underlying probability distribution.

## Score function

The score function is defined as the gradient of the log probability density with respect to the input data:

$$s(x) = \nabla_x \log p(x)$$

This function plays a crucial role in the Langevin dynamics equation that we use for sampling:

$$x_{t+1} = x_t + \alpha \cdot \nabla_x \log p(x_t) + \sqrt{2\alpha} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Intuitively, the score function acts as a vector field that points in the direction of increasing probability density. At each point $x$, $s(x)$ provides the direction and magnitude of the "force" that pushes samples towards regions of higher probability in the data distribution.

Since we typically don't have direct access to the true data distribution $p(x)$, we need to estimate the score function. This is done by training a neural network $s_\theta(x)$ to approximate $s(x)$. The neural network takes data points as input and outputs vectors that estimate the gradient of the log probability density at those points.

To train this score estimator network, we need an objective function that measures how well our approximation matches the true score function. This leads us to the concept of score matching, where we develop loss functions that allow us to train the network without requiring explicit knowledge of p(x).

## Naive score matching or explicit score matching
*Explicit means something that is clearly and directly stated, leaving no room for confusion or doubt.


$$J_{ESM}(\theta) = \mathbb{E}_{p(x)} \left[ \left\|  \hat{s}(x; \theta) - s(x) \right\|^2 \right]$$

where,

- $s(x)$ is the true score function.
- $\hat{s}(x)$ is the estimated score function.

This loss function is not tractable because we don't know the true score function.

## Implicit score matching
*Implicit means something that is implied or suggested but not directly expressed.

*We will not prove the following derivation, but it is there in class notes.

The implicit score matching loss function can be written as:

$$J_{ISM}(\theta) = \mathbb{E}_{p(x)} \left[ \frac{1}{2} \| \hat{s}(x; \theta) \|^2 + \text{tr}(\nabla_x \hat{s}(x; \theta)) \right] + C$$

As per the theorm this is equivalent to:

$$J_{ISM}(\theta) = J_{ESM}(\theta)  + C$$

Where:
- $\hat{s}(x; \theta)$ is the estimated score function
- $\text{tr}(\nabla_x \hat{s}(x; \theta))$ is the trace of the Jacobian of the estimated score function
- $C$ is a constant term independent of $\theta$

Also following assumptions were made for proving the theorm:
1. $p(x)$ is differentiable
2. $\mathbb{E}_{p(x)}[\|\nabla_x \log p(x)\|^2] < \infty$ for any $\theta$ & 
   $\mathbb{E}_{p(x)}[\|\hat{s}(x; \theta)\|^2] < \infty$ for any $\theta$
3. $p(x)\hat{s}(x; \theta) \to 0$ for any $\theta$ as $\|x\| \to \infty$



This formulation allows us to optimize the score function without explicitly knowing the true score function $\nabla_x \log p(x)$. The trace term $\text{tr}(\nabla_x \hat{s}(x; \theta))$ comes from the divergence of the score function, which is equal to $\nabla_x \cdot (\nabla_x \log p(x)) = \nabla_x^2 \log p(x)$.