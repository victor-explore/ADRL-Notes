# DDPMs as SDEs
## Step 1 of big picture  (Stochastic forward process to continuous SDE)
Now let's see how DDPMs can be viewed as discretizations of SDEs.

The forward process in DDPMs is given by:
$$x_{i+1} = \sqrt{1-\beta_{i+1}}x_i + \sqrt{\beta_{i+1}}\epsilon_i \quad \text{where} \quad \epsilon_i \sim \mathcal{N}(0,I)$$

This discrete process corresponds to a continuous-time SDE of the form(*derivation done in class*):
$$dx = -\frac{\beta(t)}{2}x dt + \sqrt{\beta(t)}dB_t$$

where:
- $\beta(t)$ is a continuous-time version of the discrete noise schedule $\beta_i$, with $\beta_i \in (0,1)$ and $\beta_T \approx 1$ because we want $x_T$ to be pure noise
- $\beta(t)$ must be bounded for the SDE to be well-defined
- The drift term $-\frac{\beta(t)}{2}x$ controls the gradual destruction of the data
- The diffusion term $\sqrt{\beta(t)}dB_t$ adds noise to the process
- This is also called a variance-preserving SDE since it maintains constant variance throughout the diffusion process
  
## Step 2 of big picture  (Continuous SDE to continuous reverse SDE)
Using Anderson's result, we can derive the reverse SDE for our DDPM forward process.

Given:
- Forward drift $f(x,t) = -\frac{\beta(t)}{2}x$
- Forward diffusion $g(t) = \sqrt{\beta(t)}$

Substituting into Anderson's reverse SDE formula we get:

$$dx = \left(-\frac{\beta(t)}{2}x - \beta(t)\nabla_x(\log p_t(x))\right)dt + \sqrt{\beta(t)}dB_t$$

## Step 3 of big picture  (Continuous reverse SDE to discrete reverse process)
*Derivation done in class - the final result is correct but the intermediate steps not checked*

1. Starting with the continuous reverse SDE:
   $$dx = \left(-\frac{\beta(t)}{2}x - \beta(t)\nabla_x(\log p_t(x))\right)dt + \sqrt{\beta(t)}dB_t$$

2. For discretization, we make the following substitutions:
   - $dx \approx x_t - x_{t+\Delta t}$ (finite difference)
   - $dt \rightarrow \Delta t$ 
   - $dB_t = \sqrt{\Delta t}\epsilon$ where $\epsilon \sim \mathcal{N}(0,I)$

3. Substituting these into the SDE:
   $$x_t - x_{t+\Delta t} = \left(-\frac{\beta(t)}{2}x_t - \beta(t)\nabla_x(\log p_t(x))\right)\Delta t + \sqrt{\beta(t)\Delta t}\epsilon$$

4. Rearranging to solve for $x_{t+\Delta t}$:
   $$x_{t+\Delta t} = x_t - \left(-\frac{\beta(t)}{2}x_t - \beta(t)\nabla_x(\log p_t(x))\right)\Delta t - \sqrt{\beta(t)\Delta t}\epsilon$$

5. Simplifying:
   $$x_{t+\Delta t} = x_t + \frac{\beta(t)}{2}x_t\Delta t + \beta(t)\nabla_x(\log p_t(x))\Delta t + \sqrt{\beta(t)\Delta t}\epsilon$$

6. Let $\Delta t = 1$ and $t \rightarrow i$ for discrete time steps:
   $$x_{i+1} = x_i + \frac{\beta_i}{2}x_i + \beta_i\nabla_x(\log p_i(x)) + \sqrt{\beta_i}\epsilon$$

7. Rearranging terms:
   $$x_{i+1} = \left(1 + \frac{\beta_i}{2}\right)x_i + \beta_i\nabla_x(\log p_i(x)) + \sqrt{\beta_i}\epsilon$$

8. Let $\epsilon_\theta(x_i,i) = -\nabla_x(\log p_i(x))$ be our neural network prediction:
   $$x_{i+1} = \left(1 + \frac{\beta_i}{2}\right)x_i - \beta_i\epsilon_\theta(x_i,i) + \sqrt{\beta_i}\epsilon$$

9. Finally, to match the DDPM reverse process shown in the image:
   $$x_{i+1} = \frac{1}{\sqrt{1-\beta_i}}\left(x_i + \beta_i\nabla_x(\log p_i(x))\right) + \sqrt{\beta_i}z_i$$
   where $z_i \sim \mathcal{N}(0,I)$
10. In practice, this is often rewritten using the neural network $\epsilon_\theta$ which predicts the noise:
   $$x_{i-1} = \frac{1}{\sqrt{1-\beta_i}}\left(x_i - \frac{\beta_i}{\sqrt{1-\beta_i}}\epsilon_\theta(x_i, i)\right) + \sigma_i z_i$$

This looks like the DDPM reverse process!

