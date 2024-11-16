# Summarizing DDPM
- VAE:
  <div style="text-align: center;"><img src="38.jpg" alt="Image Description" width="200" height="auto"/></div> 
- Hierarchical VAE:
  <div style="text-align: center;"><img src="36.jpg" alt="Image Description" width="600" height="auto"/></div> 
- DDPM(Hierarchical VAE with deterministic forward process):
  <div style="text-align: center;"><img src="37.jpg" alt="Image Description" width="600" height="auto"/></div> 
Notice that $\phi$ has been removed from $q_\phi()$.

Also see that:
- The ELBO for a Variational Autoencoder (VAE) is given by:

$$
\text{ELBO (VAE)} = \mathbb{E}_{q_\phi(z|x)} \left[ \log \frac{p(x|z)}{q_\phi(z|x)} \right]
$$
- For Denoising Diffusion Models (DDPM), the ELBO is expressed as:

$$
\text{ELBO (DM)} = \mathbb{E}_{q(x_{1:T}|x_0)} \left[ \log \frac{p(x_{0:T})}{q(x_{1:T}|x_0)} \right]
$$

