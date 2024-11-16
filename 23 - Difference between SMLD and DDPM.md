# Difference between SMLD and DDPM
![DDPM](41.jpg)

Note that: NCSN: Noise Conditional Score Network is same as SMLD: Score matching Langevin Dynamics

### Inference
The inference processes also differ between SMLD and DDPM:

1. SMLD (Noise Conditional Score Networks):
   - We start with a sequence of noise scales arranged from largest to smallest:
     $$\{\sigma_i\}_{i=1}^L, \quad \sigma_1 > \sigma_2 > ... > \sigma_L \quad (L \approx 10)$$
   
   - Start with random Gaussian noise:
     $$x_0 \sim \mathcal{N}(0, \sigma_1^2I)$$
   
   - For each noise scale $\sigma_i$ from $i=1$ to $L$:
     - Perform $K$ Langevin steps ($K \approx 100$) at fixed $\sigma_i$:
       $$x_{k+1} = x_k + \alpha s_\theta(x_k, \sigma_i) + \sqrt{2\alpha}\epsilon_k, \quad \epsilon_k \sim \mathcal{N}(0,I)$$
       for $k = 0,1,...,K-1$
     - Use final sample $x_K$ as starting point for next scale $\sigma_{i+1}$
   
   - This gradually denoises the image by:
     1. Getting a good sample at high noise levels (large $\sigma$)
     2. Using that sample as initialization for next lower noise level
     3. Repeating until reaching smallest noise scale $\sigma_L$

2. DDPM (Denoising Diffusion Probabilistic Models):
   - Uses a deterministic reverse process
   - Starting from random Gaussian noise, applies:
     $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)\right) + \sigma_t z$$
   - Where:
     - $\alpha_t = 1 - \beta_t$
     - $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$
     - $\sigma_t^2 = \beta_t$ (simplified version)
     - $z \sim \mathcal{N}(0,I)$
   - One step per noise level
   - Follows the reverse Markov chain exactly

Key differences:
- SMLD requires multiple Langevin steps per noise level, while DDPM uses single deterministic steps
- SMLD's sampling is more computationally intensive due to multiple steps per level
- DDPM's reverse process is more structured and follows the exact reverse of the forward process
- SMLD can be less stable due to Langevin dynamics, while DDPM's deterministic steps are more stable

### Forward process
The forward processes of SMLD and DDPM differ in how they add noise to the data:

1. SMLD (Noise Conditional Score Networks):
   - Uses a predefined sequence of noise scales: $\{\sigma_i\}_{i=1}^L$
   - Forward process:
     $$x_{\sigma_{i+1}} = x_{\sigma_i} + \sigma_{i+1}\epsilon_{i+1}, \quad \epsilon_i,\epsilon_{i+1} \sim \mathcal{N}(0,I)$$
     $$x_{\sigma_i} = x + \sigma_i\epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0,I)$$
     $$x_{\sigma_{i+1}} = x_{\sigma_i} + k\epsilon \quad \text{(derivation skip)}$$

   - The value of k can be derived by considering the variance:
     $$\text{Var}(x_{\sigma_{i+1}}) = \text{Var}(x_{\sigma_i} + k\epsilon)$$
     $$\sigma_{i+1}^2 = \sigma_i^2 + k^2$$
     
   - Solving for k:
     $$k = \sqrt{\sigma_{i+1}^2 - \sigma_i^2}$$
     
   - Therefore the complete forward process step is:
     $$x_{\sigma_{i+1}} = x_{\sigma_i} + \sqrt{\sigma_{i+1}^2 - \sigma_i^2}\epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$
     
   - This ensures the variance increases smoothly between noise scales

2. DDPM (Denoising Diffusion Probabilistic Models):
   - Uses a Markov chain of diffusion steps: $\{x_t\}_{t=0}^T$
   - Forward process:
     $$x_{t+1} = \sqrt{\alpha_t}x_t + \sqrt{1-\alpha_t}\epsilon$$
     where $\epsilon \sim \mathcal{N}(0,I)$