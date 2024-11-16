# Vector Quantized Variational Autoencoders (VQ-VAEs)

Vector Quantized Variational Autoencoders (VQ-VAEs) were introduced to address several limitations of regular VAEs:

1. The "posterior collapse" problem where the latent code is ignored
2. The continuous latent space can make it difficult to model discrete structures
3. The aggregated posterior mismatch between $q_\phi(z)$ and the prior $p(z)$

VQ-VAEs solve these issues by using discrete latent variables through vector quantization. The architecture consists of:
<div style="text-align: center;"><img src="28.jpg" alt="Image Description" width="900" height="auto"/></div>

1. An encoder network $E_{\phi}$ that maps input $x \in \mathbb{R}^N$ to continuous latent vectors $z_e = E_{\phi}(x) \in \mathbb{R}^{D}$

2. A codebook $\mathcal{C} = \{e_k\}_{k=1}^K$ containing $K$ learnable embedding vectors $e_k \in \mathbb{R}^D$, where typically $K$ is large (e.g., 512 or 1024)

3. A vector quantization operation that maps $z_e$ to the nearest codebook vector:
   $z_q = e_k$ where $k = \argmin_j \|z_e - e_j\|_2$

4. A decoder network $D_{\theta}$ that reconstructs the input from the quantized vectors: $\hat{x} = D_{\theta}(z_q)$

The training objective consists of three terms:

$L = \underbrace{\|x - D_\theta(z_q)\|_2^2}_\text{reconstruction loss} + \underbrace{\|\text{sg}[z_e] - e_k\|_2^2}_\text{codebook loss} + \underbrace{\beta\|z_e - \text{sg}[e_k]\|_2^2}_\text{commitment loss}$

where:
- $\text{sg}[\cdot]$ is the stop-gradient operator
- The codebook loss updates the embeddings to match the encoder output
- The commitment loss ensures the encoder commits to codebook vectors
- $\beta$ is a hyperparameter controlling the commitment (typically 0.25)

Unlike regular VAEs that use a continuous latent space with sampling and KL divergence regularization, VQ-VAEs use a discrete latent space through the codebook. This allows them to better model discrete structures while avoiding posterior collapse and distribution matching issues.
