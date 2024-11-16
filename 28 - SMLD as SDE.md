# SMLD as SDE
## Step 1 of big picture  (Stochastic forward process to continuous SDE)
We know SMLD (Score Matching with Langevin Dynamics) forward process is:
   $$x_i = x_{i-1} + \sqrt{\sigma_{t+1}^2 - \sigma_t^2}z_{i-1}, \quad z_{i-1} \sim \mathcal{N}(0,I)$$

Let's convert the discrete SMLD process to a continuous SDE:

1. First, let's write the discrete process in terms of continuous time:
   - Let $\Delta t = \frac{1}{N}$ be the time step
   - Map discrete index $i$ to continuous time $t$: $i = \frac{t}{N}$
   - Write $\sigma_i^2 = \sigma(t)^2$ for continuous noise schedule

2. The discrete process becomes:
   $$x(t + \Delta t) = x(t) + \sqrt{\sigma(t + \Delta t)^2 - \sigma(t)^2}z_t$$
   where $z_t \sim \mathcal{N}(0,I)$

3. Rearranging to get the change in x:
   $$x(t + \Delta t) - x(t) = \sqrt{\sigma(t + \Delta t)^2 - \sigma(t)^2}z_t$$

4. Note that for small $\Delta t$:
   $$\sigma(t + \Delta t)^2 - \sigma(t)^2 \approx \frac{d(\sigma(t)^2)}{dt}\Delta t$$

5. Also, $z_t\sqrt{\Delta t} = dB_t$ for Brownian motion increments

6. Substituting these we get:
   $$dx = \sqrt{\frac{d(\sigma(t)^2)}{dt}\Delta t}\frac{dB_t}{\sqrt{\Delta t}}$$

7. Simplifying:
   $$dx = \sqrt{\frac{d(\sigma(t)^2)}{dt}}dB_t$$

## Step 2 of big picture  (Continuous SDE to continuous reverse SDE)
Now we can write the reverse SDE:
1. Using Anderson's result, with:
   - Forward drift $f(x,t) = 0$ (no drift term)
   - Forward diffusion $g(t) = \sqrt{\frac{d(\sigma(t)^2)}{dt}}$

2. The reverse SDE is:
   $$dx = \left(-g(t)^2\nabla_x(\log p_t(x))\right)dt + g(t)dB_t$$

3. Substituting $g(t)$:
   $$dx = \left(-\frac{d(\sigma(t)^2)}{dt}\nabla_x(\log p_t(x))\right)dt + \sqrt{\frac{d(\sigma(t)^2)}{dt}}dB_t$$

## Step 3 of big picture  (Continuous reverse SDE to discrete reverse process)

1. To discretize this SDE:
   - Let $\alpha(t) = \frac{d(\sigma(t)^2)}{dt}$
   - Then $\alpha(t)\Delta t = \Delta(\sigma(t)^2)$

2. The discretized equation becomes:
   $$x(t + \Delta t) - x(t) = -\alpha(t)\nabla_x(\log p_t(x))\Delta t - \sqrt{\alpha(t)\Delta t}z(t)$$

3. Rearranging:
   $$x(t + \Delta t) = x(t) + \alpha(t)\nabla_x(\log p_t(x))\Delta t + \sqrt{\alpha(t)\Delta t}z(t)$$

4. Converting back to discrete indices:
   $$x_{i-1} = x_i + (\sigma_i^2 - \sigma_{i-1}^2)\nabla_x\log p_i(x) + \sqrt{\sigma_i^2 - \sigma_{i-1}^2}z_i$$

where $z_i \sim \mathcal{N}(0,I)$

This looks similar to the Langevin dynamics equation we saw earlier, with the gradient term $\nabla_x\log p_i(x)$ guiding the sampling process.
The key difference is that here the noise schedule $\sigma_i^2$ controls both the step size and noise magnitude, while in Langevin dynamics these were separate parameters.





















