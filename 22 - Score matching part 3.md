# Score matching part 3
## Recap
- Langevin equation: $x_{t+1} = x_t + \alpha \cdot \nabla_x \log p(x_t) + \sqrt{2\alpha} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$
- score function: $\nabla_x \log p(x) = s(x)$
- $J_{ESM}(\theta) = \mathbb{E}_{p(x)} \left[ \frac{1}{2} \| \hat{s}(x; \theta) - \nabla_x \log p(x) \|^2 \right]$
- $J_{ISM}(\theta) = \mathbb{E}_{p(x)} \left[ \frac{1}{2} \| \hat{s}(x; \theta) \|^2 + \text{tr}(\nabla_x \hat{s}(x; \theta)) \right] + C$
- Theorem: $J_{ISM}(\theta) = J_{ESM}(\theta) + C$
- $J_{PSM}(\theta) = \frac{1}{2} \mathbb{E}_v \mathbb{E}_{p(x)} \left[ \| v^\top \hat{s}_\theta(x) - v^\top s(x) \|^2 \right]$
- $J_{SSM}(\theta) = \mathbb{E}_v \mathbb{E}_{p(x)} \left[ \frac{1}{2}(v^\top \hat{s}_\theta(x))^2 + v^\top (\nabla_x \hat{s}_\theta(x)) v \right]$
- Theorem: $J_{PSM}(\theta) = J_{SSM}(\theta) + C$


## Big picture plan
  <div style="text-align: center;"><img src="40.jpg" alt="Image Description" width="500" height="auto"/></div> 


## Denoising score matching
Denoising score matching (DSM) is an alternative approach to score matching. It introduces an auxiliary variable and works with conditional scores, which offers several advantages.

Let's start with the DSM objective:

$$J_{DSM}(\theta) = \frac{1}{2} \mathbb{E}_{p(x,x')} \left[ \| \hat{s}_\theta(x) - \nabla_{x} \log p(x|x') \|^2 \right]$$

Where:
- $x$ is a sample from the true data distribution $p(x)$
- $x'$ is auxiliary variable obtained by adding Gaussian noise to $x$: $x' = x + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$
- $\hat{s}_\theta(x)$ is the estimated score function that we want to learn
- $\nabla_{x} \log p(x|x')$ is the conditional score, which has an analytical form for Gaussian noise

The DSM objective allows us to learn the score function $\nabla_x \log p(x)$ by:
1. Generating training pairs $(x,x')$ by adding noise to data samples
2. Leveraging the known form of $p(x|x')$ for Gaussian noise
3. Avoiding direct computation of the intractable normalization constant

Key aspects of Denoising Score Matching:

1. Auxiliary Variable: DSM introduces $x'$, which is a noisy version of the original data point $x$. This allows us to work with conditional distributions.

2. Conditional Score: Instead of estimating the score of the data distribution directly, DSM estimates the conditional score $\nabla_{x} \log p(x|x')$.

3. Gaussian Noise: Typically, Gaussian noise is added to create $x'$. This has two significant benefits:
   - We can add Gaussian noise to our data easily.
   - The conditional distribution $p(x|x')$ becomes Gaussian, which has a known analytical form for its score.

4. Theorem: There's an important relationship between DSM and the original score matching objective:

   $$J_{ESM}(\theta) = J_{DSM}(\theta) + C$$

   Where $C$ is a constant independent of $\theta$. This theorem shows that optimizing the DSM objective is equivalent to optimizing the original score matching objective.

The benefits of using conditional scores and Gaussian noise make DSM a powerful and practical approach to score matching, addressing many of the computational challenges faced by earlier methods.

## Continue DSM

Let's continue exploring Denoising Score Matching (DSM) in more detail. The original DSM objective was:

$$J_{DSM}(\theta) = \frac{1}{2} \mathbb{E}_{p(x,x')} \left[ \| \hat{s}_\theta(x) - \nabla_{x} \log p(x|x') \|^2 \right]$$

To align with the lecture notation make we'll make the following notation changes:
- Original datapoint $x$ becomes $\tilde{x}$
- Noisy version $x'$ becomes $x$

This gives us:

$$J_{DSM}(\theta) = \frac{1}{2} \mathbb{E}_{p(\tilde{x},x)} \left[ \| \hat{s}_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log p(\tilde{x}|x) \|^2 \right]$$

We can further expand this by explicitly defining the perturbation distribution:

$$J_{DSM}(\theta) = \frac{1}{2} \mathbb{E}_{p(x)} \mathbb{E}_{q_\sigma(\tilde{x}|x)} \left[ \| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x) \|^2 \right]$$

Key components:
1. Perturbation Distribution ($q_\sigma(\tilde{x}|x)$):
   - Defined as $q_\sigma(\tilde{x}|x) = \mathcal{N}(\tilde{x}|x, \sigma^2I)$
   - Generated via $\tilde{x} = x + \sigma\epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$
   - Adds controlled Gaussian noise to create noisy versions of data points

2. Score of Perturbation Kernel:
   The term $\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x)$ can be derived analytically:
   
   $$\begin{aligned}
   \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x) &= -\nabla_{\tilde{x}} \frac{(\tilde{x}-x)^2}{2\sigma^2} \\
   &= -\frac{\tilde{x}-x}{\sigma^2}
   \end{aligned}$$

Substituting this back into our objective:

$$J_{DSM}(\theta) = \frac{1}{2} \mathbb{E}_{p(x)} \mathbb{E}_{q_\sigma(\tilde{x}|x)} \left[ \| s_\theta(\tilde{x}) + \frac{\tilde{x}-x}{\sigma^2} \|^2 \right]$$

## Reparameterization for Efficient Training

We can make this objective more computationally tractable through reparameterization. Instead of sampling $\tilde{x}$ directly, we express it in terms of $x$ and $\epsilon$:

$$\tilde{x} = x + \sigma\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This transforms our objective into:

$$J_{DSM}(\theta) = \frac{1}{2} \mathbb{E}_{p(x)} \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} \left[ \| s_\theta(x + \sigma\epsilon) + \frac{\epsilon}{\sigma} \|^2 \right]$$

Further simplification yields:

$$J_{DSM}(\theta) = \frac{1}{2\sigma^2} \mathbb{E}_{p(x)} \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} \left[ \| \sigma s_\theta(x + \sigma\epsilon) + \epsilon \|^2 \right]$$

In the lecture, this was presented in a slightly different but equivalent form:

$$\mathbb{E}_{x,\epsilon} \left( \frac{1}{2\sigma^2} \| \epsilon_\theta(\tilde{x}) - \epsilon \|^2 \right)$$


Critical Properties of this Formulation:

1. Scale-Aware Training: The $\frac{1}{2\sigma^2}$ factor naturally weights the loss more heavily at smaller noise levels. This ensures accurate denoising across different scales.

2. Direct Noise Prediction: Instead of predicting scores, the model learns to predict the noise directly.

## Data Coverage Problem

A fundamental challenge in score matching is ensuring accurate score estimation across the entire data distribution $p(x)$, particularly in regions of low density. This is known as the data coverage problem.

Noise Conditional Score Networks (NCSNs) address this challenge through a multi-scale approach:

1. Noise Scale Hierarchy:
   $$\{\sigma_i\}_{i=1}^L, \quad \text{where } \sigma_1 < \sigma_2 < ... < \sigma_L$$
   - $L$ typically ranges from 10 to several hundred scales
   - $\sigma_1 = \sigma_{min}$ (minimal noise)
   - $\sigma_L = \sigma_{max}$ (maximal noise)

2. Distribution Bridging:
   Two key conditions are enforced:
   - $p_{\sigma_{min}}(x) \approx p(x)$ (preserves original distribution)
   - $p_{\sigma_{max}}(x) \approx \mathcal{N}(0, I)$ (approaches Gaussian)

3. Noise Application:
   For each scale $\sigma_i$:
   $$p_{\sigma_i}(\tilde{x}|x) = \mathcal{N}(\tilde{x}|x, \sigma_i^2I)$$

4. Score Estimation:
   The model learns scale-conditional scores:
   $$s_\theta(x, \sigma_i) \approx \nabla_x \log p_{\sigma_i}(x)$$

5. Scale-Specific Loss:
   $$\mathcal{L}(\theta, \sigma_i) = \frac{1}{2} \mathbb{E}_{p(x)} \mathbb{E}_{q_{\sigma_i}(\tilde{x}|x)} \left[ \left\| s_\theta(\tilde{x}, \sigma_i) + \frac{\tilde{x}-x}{\sigma_i^2} \right\|^2 \right]$$

6. Combined Training Objective:
   $$\mathcal{L}(\theta; \{\sigma_i\}_{i=1}^L) = \frac{1}{L} \sum_{i=1}^L \lambda(\sigma_i) \mathcal{L}(\theta, \sigma_i)$$
   where $\lambda(\sigma_i)$ weights different scales' contributions

Benefits of this Multi-Scale Approach:

1. Comprehensive Coverage:
   - High-density regions: Accurate modeling with small $\sigma_i$
   - Low-density regions: Stable estimation with large $\sigma_i$

2. Smooth Interpolation:
   - Gradual transition between noise levels
   - Continuous coverage of the data manifold

3. Training Stability:
   - Different scales provide complementary learning signals
   - Reduced sensitivity to individual scale choices

This approach enables NCSNs to effectively model complex data distributions while maintaining stability and accuracy across all regions of the data space.

![DDPM](6.jpg)










