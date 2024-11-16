# Naive GANs
A Naive GAN (Generative Adversarial Network) refers to the basic or "vanilla" version of GANs, introduced by Ian Goodfellow in 2014.

## Derivation

Recall the genereral expression for the GANs that we got by minimizing F-divergence:

$$\theta^* = \underset{\theta}{\text{argmin}} \underset{\phi}{\sup} \left[\mathbb{E}_{x \sim p_{data}}[T_\phi(x)] - \mathbb{E}_{x \sim p_{generator}}[f^*(T_\phi(x))]\right]$$

where:

- $\theta$: Parameters of the generator network
- $\phi$: Parameters of the discriminator (critic) network
- $\theta^*$: Optimal parameters for the generator
- $\text{argmin}_\theta$: Argument that minimizes the expression with respect to $\theta$
- $\sup_\phi$: Supremum (least upper bound) with respect to $\phi$
- $\mathbb{E}_{x \sim p_{data}}$: Expectation over the real data distribution
- $\mathbb{E}_{x \sim p_{generator}}$: Expectation over the generator's distribution
- $T_\phi(x)$: The discriminator function, parameterized by $\phi$
- $f^*$: The convex conjugate of the function $f$ used in the F-divergence
- $p_{data}$: The true data distribution
- $p_{generator}$: The distribution of the generated data


Write $T_\phi(x)$ as composite function $T_\phi(x) = \sigma(V_\phi(x))$, where $\sigma$ is the sigmoid function Substitute this in the expression for GANs:

$$\theta^* = \underset{\theta}{\text{argmin}} \underset{\phi}{\sup} \left[\mathbb{E}_{x \sim p_{data}}[\sigma(V_\phi(x))] - \mathbb{E}_{x \sim p_{generator}}[f^*(\sigma(V_\phi(x)))]\right]$$

For the naive GAN, we use the Jensen-Shannon divergence, which corresponds to:

$$f(t) = t \log t - (t+1) \log(t+1)$$

The convex conjugate of this function is:

$$f^*(t) = -\log(1-e^t)$$


Substituting this into our expression:

$$\theta^* = \underset{\theta}{\text{argmin}} \underset{\phi}{\sup} \left[\mathbb{E}_{x \sim p_{data}}[\sigma(V_\phi(x))] - \mathbb{E}_{x \sim p_{generator}}[-\log(1-e^{\sigma(V_\phi(x))})]\right]$$

Substitute $\sigma(V_\phi(x)) = \frac{1}{1+e^{-V_\phi(x)}}$, we get:

$$\theta^* = \underset{\theta}{\text{argmin}} \underset{\phi}{\sup} \left[\mathbb{E}_{x \sim p_{data}}[\frac{1}{1+e^{-V_\phi(x)}}] - \mathbb{E}_{x \sim p_{generator}}[-\log(1-\frac{1}{1+e^{-V_\phi(x)}})]\right]$$

Simplify the expression using:

$$1 - \frac{1}{1+e^{-V_\phi(x)}} = \frac{e^{-V_\phi(x)}}{1+e^{-V_\phi(x)}}$$

This gives us:


$$\theta^* = \underset{\theta}{\text{argmin}} \underset{\phi}{\sup} \left[\mathbb{E}_{x \sim p_{data}}[\frac{1}{1+e^{-V_\phi(x)}}] - \mathbb{E}_{x \sim p_{generator}}[-\log(\frac{e^{-V_\phi(x)}}{1+e^{-V_\phi(x)}})]\right]$$



$$\theta^*= \mathbb{E}_{x \sim p_{data}}[\frac{1}{1+e^{-V_\phi(x)}}] - \mathbb{E}_{x \sim p_{generator}}[-(-V_\phi(x) - \log(1+e^{-V_\phi(x)}))]$$

For the second expectation term:
   $-(-V_\phi(x) - \log(1+e^{-V_\phi(x)}))$
   $= V_\phi(x) + \log(1+e^{-V_\phi(x)})$
   $= \log(e^{V_\phi(x)}) + \log(1+e^{-V_\phi(x)})$
   $= \log(e^{V_\phi(x)}(1+e^{-V_\phi(x)}))$
   $= \log(e^{V_\phi(x)} + 1)$
   $= \log(1+e^{V_\phi(x)})$



$$\theta^* = \mathbb{E}_{x \sim p_{data}}[\frac{1}{1+e^{-V_\phi(x)}}] - \mathbb{E}_{x \sim p_{generator}}[\log(1+e^{-V_\phi(x)})]$$
Using the properties of logarithms and the fact that $\frac{e^{-V_\phi(x)}}{1+e^{-V_\phi(x)}} = 1 - \frac{1}{1+e^{-V_\phi(x)}}$, we can rewrite:

$$\theta^* = \underset{\theta}{\text{argmin}} \underset{\phi}{\sup} \left[\mathbb{E}_{x \sim p_{data}}[-\log(1+e^{-V_\phi(x)})] - \mathbb{E}_{x \sim p_{generator}}[\log(1+e^{V_\phi(x)})]\right]$$

This is equivalent to:

$$\theta^* = \underset{\theta}{\text{argmin}} \underset{\phi}{\sup} \left[\mathbb{E}_{x \sim p_{data}}[\log(\frac{1}{1+e^{-V_\phi(x)}})] + \mathbb{E}_{x \sim p_{generator}}[\log(\frac{e^{-V_\phi(x)}}{1+e^{-V_\phi(x)}})]\right]$$

Let's define the discriminator function $D_\phi(x) = \frac{1}{1+e^{-V_\phi(x)}}$, which maps inputs to probabilities in [0,1]. Then our objective function takes the elegant form:

$$\theta^* = \underset{\theta}{\text{argmin}} \underset{\phi}{\sup} \left[\mathbb{E}_{x \sim p_{data}}[\log(D_\phi(x))] + \mathbb{E}_{x \sim p_{generator}}[\log(1-D_\phi(x))]\right]$$

This is the standard form of the GAN objective function as presented in the original paper by Goodfellow et al.

## Interpretation
<div style="text-align: center;">
    <img src="9.jpg" alt="Image Description" width="700" height="auto"/>
</div>


In the naive GAN framework, we can interpret the roles of the discriminator and generator as follows:

1. Discriminator ($D$):
   - The discriminator aims to maximize $\mathbb{E}_{x \sim p_{data}}[\log(D_\phi(x))] + \mathbb{E}_{x \sim p_{generator}}[\log(1-D_\phi(x))]$.
   - Because discriminator is of the form $D_\phi(x) = \frac{1}{1+e^{-V_\phi(x)}}$, it outputs a probability between 0 and 1, where:
     - $D_\phi(x) \approx 1$ indicates the discriminator believes $x$ is from the real data distribution
     - $D_\phi(x) \approx 0$ indicates the discriminator believes $x$ is from the generator (fake data)
   - This pushes the discriminator to correctly classify real and fake samples.

2. Generator ($G$):

   - The generator aims to minimize $\mathbb{E}_{x \sim p_{data}}[\log(D_\phi(x))] + \mathbb{E}_{x \sim p_{generator}}[\log(1-D_\phi(x))]$. The first term is the expected log-likelihood of the discriminator classifying a real sample as real, and the second term is the expected log-likelihood of the discriminator classifying a generated sample as fake. The first term is a constant with respect to the generator, so the generator aims to minimize the second term. Hence,generator tries minimise $\mathbb{E}_{x \sim p_{generator}}[\log(1-D_\phi(x))]$,
   - If the generator produces samples that the discriminator classifies as real, $D_\phi(x) \approx 1$. Then $\log(1-D_\phi(x)) \approx 0$.

Note that this kind of neat interpretation of the roles of the discriminator and generator is not possible for other choice of $f$ and $f^*$.

## Training

The training process for GANs involves alternating between training the discriminator and the generator. Here's a step-by-step explanation of the training process:

1. Initialize the Generator ($G$) and Discriminator ($D$) networks.

2. For each training iteration:

   a. Train the Discriminator:
      - Generate a batch of fake samples using the generator: $x_{fake} = G(z)$, where $z$ is random noise.
      - Sample a batch of real data: $x_{real}$.
      - Calculate the discriminator's loss:
        $L_D = -[\mathbb{E}_{x \sim p_{data}}[\log(D(x))] + \mathbb{E}_{x \sim p_g}[\log(1 - D(G(z)))]]$
      - Update the discriminator's parameters using gradient descent to minimize $L_D$.

   b. Train the Generator:
      - Generate a new batch of fake samples: $x_{fake} = G(z)$
      - Calculate the generator's loss:
        $L_G = -\mathbb{E}_{z \sim p_g}[\log(1 - D(G(z)))]$
      - In practice, to avoid vanishing gradients early in training, the generator's loss is often implemented as:
        $L_G = -\mathbb{E}_{z \sim p_g}[\log(D(G(z)))]$
      - Update the generator's parameters using gradient descent to minimize $L_G$.


The training process continues until the generator produces samples that are indistinguishable from real data, or until a predetermined number of iterations is reached.









