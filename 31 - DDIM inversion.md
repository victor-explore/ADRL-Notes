
# DDIM inversion

## $X_T$ to $X_0$
For $\sigma = 0$, the DDIM process becomes deterministic:

$$x_{t-1} = \sqrt{\alpha_{t-1}}\left(\frac{x_t - \sqrt{1-\alpha_t}\epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}}\right) + \sqrt{1-\alpha_{t-1}}\epsilon_\theta^{(t)}(x_t)$$

where:
- $\epsilon_\theta^{(t)}(x_t)$ is the predicted noise from our trained model
- $\alpha_t$ is the cumulative product of $(1-\beta_t)$

This deterministic nature means that:
1. For a given $x_T$, we can find a unique $x_0$ by following the forward process
2. Given this $x_T$, we can recover the exact same $x_0$ through the reverse process

This property makes DDIM particularly useful for:
- Finding latent representations of real images
- Performing image editing in the latent space
- Interpolating between images in a semantically meaningful way

The key insight is that unlike DDPM which has stochastic transitions, DDIM with $\sigma=0$ gives us a one-to-one mapping between $x_0$ and $x_t$, making the inversion process well-defined and deterministic.

## $X_0$ to $X_T$
For the forward process from $x_0$ to $x_T$, we can write:

$$\frac{x_t}{\sqrt{\alpha_t}} = \frac{x_{t-1}}{\sqrt{\alpha_{t-1}}} + \epsilon_\theta^{(t)}(x_t)\left(\sqrt{\frac{1-\alpha_t}{\alpha_t}} - \sqrt{\frac{1-\alpha_{t-1}}{\alpha_{t-1}}}\right)$$

To convert this to continuous form, we can write:

$$\frac{dx_t}{\sqrt{\alpha_t}} = \frac{x_{t-1}dt}{\sqrt{\alpha_{t-1}}} + \epsilon_\theta^{(t)}(x_t)\left(\sqrt{\frac{1-\alpha_t}{\alpha_t}} - \sqrt{\frac{1-\alpha_{t-1}}{\alpha_{t-1}}}\right)dt$$

Let's define:
- $\bar{x}(t) = \frac{x_t}{\sqrt{\alpha_t}}$ 
- $\sigma(t) = \sqrt{\frac{1-\alpha_t}{\alpha_t}}$

Then we get (derivation not done in class):

$$d\bar{x}(t) = \epsilon_\theta^{(t)}\left(\frac{\bar{x}(t)}{\sqrt{1+\sigma^2(t)}}\right)d\sigma(t)$$

This continuous equation can be discretized back as:

$$\bar{x}(t) = \bar{x}(t-\Delta t) + \Delta\bar{x}(t-\Delta t)$$

Which gives us our original forward process equation:

$$\frac{x_t}{\sqrt{\alpha_t}} = \frac{x_{t-1}}{\sqrt{\alpha_{t-1}}} + \epsilon_\theta^{(t)}(x_{t-1})\left(\sqrt{\frac{1-\alpha_t}{\alpha_t}} - \sqrt{\frac{1-\alpha_{t-1}}{\alpha_{t-1}}}\right)$$

For DDIM inversion, we can rearrange this to solve for the previous timestep:

$$\frac{x_{t-1}}{\sqrt{\alpha_{t-1}}} = \frac{x_t}{\sqrt{\alpha_t}} - \epsilon_\theta^{(t)}(x_t)\left(\sqrt{\frac{1-\alpha_t}{\alpha_t}} - \sqrt{\frac{1-\alpha_{t-1}}{\alpha_{t-1}}}\right)$$

This is a deterministic ODE that we can solve using numerical methods to find a unique $x_T$ for a given $x_0$.