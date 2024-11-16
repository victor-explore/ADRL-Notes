# Conditional score matching
When we had unlabeled data, we estimated the score function:

$$s(x) = \nabla_x \log p(x)$$

Now, we have labeled data ie $D = \{(x_i, y_i)\}_{i=1}^N$ we instead estimate the conditional score function:

$$s(x|y) = \nabla_x \log p(x|y)$$

Where:
- $s(x)$ is the unconditional score function
- $s(x|y)$ is the conditional score function
- $p(x)$ is the data distribution
- $p(x|y)$ is the conditional data distribution given some condition $y$

This can be done in 2 ways:
- Classifier guided score matching
- Classifier free guidance

## Classifier guided score matching
   - Bayes rule
  $$ p(x|y) = \frac{p(y|x)p(x)}{p(y)} $$
   - Take log on both sides
   $$ \log p(x|y) = \log p(y|x) + \log p(x) - \log p(y) $$
   - Take gradient on both sides
   $$ \nabla_x \log p(x|y) = \nabla_x \log p(y|x) + \nabla_x \log p(x) $$
   - The gradient of the log likelihood is the score function, so we get:
   $$ s(x|y) = s(y|x) + s(x) $$

   Note that $\nabla_x \log p(y|x)$ is not a score because the gradient w.r.t $x$ and not $y$. However we write it like this for convenience.

   - Practically, we can use a classifier to estimate $s(y|x)$
     - Train a classifier $p_\phi(y|x)$ using cross entropy loss
     - The classifier is trained independently from the score model(after training the score model)
     - The gradient $\nabla_x \log p_\phi(y|x)$ gives us $s_\phi(y|x)$
     - The langevin dynamics equation becomes:
       $$x_{t+1} = x_t + \alpha(s_\phi(y|x) + s_\theta(x)) + \sqrt{2\alpha}\epsilon$$
       where:
       - $s_\phi(y|x)$ is the classifier guidance term
       - $s_\theta(x)$ is the score model
       - $\alpha$ is the step size
       - $\epsilon \sim \mathcal{N}(0,I)$ is random noise
     - This results in a stable diffusion process in latent space

## Classifier free guidance

Train a conditional score model $p(x|y)$ with conditioning dropout:
  - During training, for each batch:
    - With probability $p$ (e.g. $p=0.1$):
      - Replace the conditioning $y$ with a null token (e.g. zero vector or special embedding)
      - Train the model to predict $s_\theta(x|\text{null})$
    - With probability $(1-p)$:
      - Keep the original conditioning $y$
      - Train the model to predict $s_\theta(x|y)$
  - This training strategy results in a single model that can:
    - Act as a conditional score model $s_\theta(x|y)$ when given real conditioning
    - Act as an unconditional score model $s_\theta(x)$ when given the null token
    - The unconditional model emerges from training with dropped conditions
  - During sampling:
    - Can interpolate between conditional and unconditional by:
      $$s_\theta(x|y)_{CFG} = (1+w)s_\theta(x|y) - w s_\theta(x|\text{null})$$
      where $w$ controls the classifier-free guidance strength

