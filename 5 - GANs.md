# Generative Adversarial Networks (GANs)

Recall that generative models try to learn the data distribution $P_{data}$.

GANs try to learn $P_{data}$ by approximating it by $P_{generator}$ by minimizing F-divergence $D_f(P_{data} || P_{generator})$.

## Derivation of GANs

Let $\theta$ be the parameters of the generator such that it maps a sample $z \sim N(0, I)$ to a sample $x \sim p_{generator}$.
<div style="text-align: center;">
    <img src="5.jpg" alt="Image Description" width="500" height="auto"/>
</div>


Then, the optimal parameters for the generator can be expressed as:

$$\theta^* = \underset{\theta}{\text{argmin}} D_f(P_{data} || P_{generator})$$

$$\theta^* = \underset{\theta}{\text{argmin}} \int_{\mathclap{x}} p_{generator}(x) f\left(\frac{p_{data}(x)}{p_{generator}(x)}\right) dx$$

We know that $f^*(y) = \underset{x \in \text{dom} f}{\sup} (y^T x - f(x))$. Therefore, we can rewrite our optimization problem as:

$$\theta^* = \underset{\theta}{\text{argmin}} \int_{x} p_{generator}(x) \underset{t \in \text{dom} f^*}{\sup} \left[t \frac{p_{data}(x)}{p_{generator}(x)} - f^*(t)\right] dx$$

where $f\left(\frac{p_{data}(x)}{p_{generator}(x)}\right) = \underset{t \in \text{dom} f^*}{\sup} \left[t \frac{p_{data}(x)}{p_{generator}(x)} - f^*(t)\right]$ as $f$ is the convex conjugate of $f^*$.

Let's model $t \in \text{dom} f^*$ by a neural network with parameters $\phi$. We'll denote this neural network as $T_\phi(x)$.
<div style="text-align: center;">
    <img src="4.jpg" alt="Image Description" width="500" height="auto"/>
</div>




$$\theta^* = \underset{\theta}{\text{argmin}} \int_{x} p_{generator}(x) \underset{\phi}{\sup} \left[T_\phi(x) \frac{p_{data}(x)}{p_{generator}(x)} - f^*(T_\phi(x))\right] dx$$



Now see that:

$$\underset{\theta}{\text{argmin}} \int_{x} p_{generator}(x) \underset{\phi}{\sup} \left[T_\phi(x) \frac{p_{data}(x)}{p_{generator}(x)} - f^*(T_\phi(x))\right] dx \leq \underset{\theta}{\text{argmin}} \underset{\phi}{\sup} \int_{x} p_{generator}(x) \left[T_\phi(x) \frac{p_{data}(x)}{p_{generator}(x)} - f^*(T_\phi(x))\right] dx$$

We will replace $\underset{\theta}{\text{argmin}} \int_{x} p_{generator}(x) \underset{\phi}{\sup} \left[T_\phi(x) \frac{p_{data}(x)}{p_{generator}(x)} - f^*(T_\phi(x))\right] dx$ with $\underset{\theta}{\text{argmin}} \underset{\phi}{\sup} \int_{x} p_{generator}(x) \left[T_\phi(x) \frac{p_{data}(x)}{p_{generator}(x)} - f^*(T_\phi(x))\right] dx$ in the expression of $\theta^*$. Hence the $\theta^*$ we are optimising is minimum lower bound of original $\theta^*$ which is one of the draw backs of GANs. For not changing notations we will use the same notation $\theta^*$ for the new $\theta^*$ we are optimising.

Hence, by swapping the order of supremum and integral we get:

$$\theta^* = \underset{\theta}{\text{argmin}} \underset{\phi}{\sup} \int_{x} p_{generator}(x) \left[T_\phi(x) \frac{p_{data}(x)}{p_{generator}(x)} - f^*(T_\phi(x))\right] dx \qquad \qquad (eqn -1)$$

$$\theta^* = \underset{\theta}{\text{argmin}} \underset{\phi}{\sup} \int_{x} \left[T_\phi(x) p_{data}(x) - p_{generator}(x) f^*(T_\phi(x))\right] dx$$

$$\theta^* = \underset{\theta}{\text{argmin}} \underset{\phi}{\sup} \left[\int_{x} T_\phi(x) p_{data}(x) dx - \int_{x} p_{generator}(x) f^*(T_\phi(x)) dx\right]$$

The first integral is an expectation over the data distribution, and the second is an expectation over the generator distribution:

$$\theta^* = \underset{\theta}{\text{argmin}} \underset{\phi}{\sup} \left[\mathbb{E}_{x \sim p_{data}}[T_\phi(x)] - \mathbb{E}_{x \sim p_{generator}}[f^*(T_\phi(x))]\right]$$

This simplified form is the core of the GAN objective function. The generator (parameterized by $\theta$) tries to minimize this expression, while the discriminator also known as critic (parameterized by $\phi$) tries to maximize it.

## Why is it called adversarial network?

The term "adversarial" in GANs refers to the adversarial relationship between the generator and the discriminator. In the context of GANs, the generator and the discriminator play a two-player min-max game. The generator tries to minimize the objective function while the discriminator tries to maximize it. This adversarial relationship is what gives rise to the term "adversarial network."







