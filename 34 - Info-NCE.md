# Info-NCE
## Idea
Instead of using 1 negative sample, why not use more?

## Formulation
Let us formalize the Info-NCE framework:

1. Given:
   - A data point $x \in \mathcal{X}$ from input space $\mathcal{X}$
   - A context $c \in \mathcal{C}$ from context space $\mathcal{C}$
   - A positive sampling distribution $p_{pos}(x,c)$
   - A negative sampling distribution $p_{neg}(x,c)$

2. We sample:
   - One positive sample $x^+ \sim p_{pos}(x,c)$
   - $N-1$ negative samples $\{x^-_i\}_{i=1}^{N-1} \sim p_{neg}(x,c)$

3. The InfoNCE loss is defined as:

   $$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{\exp(f_\theta(x)^\top f_\theta(x^+))}{\exp(f_\theta(x)^\top f_\theta(x^+)) + \sum_{i=1}^{N-1} \exp(f_\theta(x)^\top f_\theta(x^-_i))}\right]$$

   where:
   - $f_\theta: \mathcal{X} \rightarrow \mathbb{R}^d$ is a neural network encoder that maps inputs to d-dimensional embeddings
   - $\theta$ represents the learnable parameters of the encoder
   - $x^+$ denotes the positive sample
   - $x^-_i$ denotes the i-th negative sample
   - The expectation is taken over the sampling distributions

4. This formulation can be interpreted as a softmax-based classifier that tries to identify the positive sample among N-1 negative samples.
5. We can use these embeddings to train other models downstream.
