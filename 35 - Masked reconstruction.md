# Masked reconstruction
## Idea
Contrastive learning is one way to learn representations from unlabeled data. Masked reconstruction is another way.

## Formulation
Let us formalize the masked reconstruction framework:

1. Given:
   - Input space $\mathcal{X}$ 
   - A data point $x \in \mathcal{X}$
   - A masking operator $\hat{x}$ that corrupts the input
   - An encoder-decoder model $T_\theta(x)$ parameterized by $\theta$

2. The process works as follows:
   - Apply masking operator: $\hat{x} = \text{mask}(x)$
   - Pass masked input through model: $\tilde{x} = T_\theta(\hat{x})$
   - Reconstruct original input: $\hat{x} \rightarrow \tilde{x} \approx x$

3. The objective is to minimize reconstruction loss:
   $$\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_\text{data}}\left[d(x, T_\theta(\text{mask}(x)))\right]$$
   
   where:
   - $d(\cdot,\cdot)$ is a distance metric (e.g. MSE, cross-entropy)
   - The expectation is taken over the data distribution
   - $\theta$ represents the learnable parameters

4. Common masking strategies include:
   - Random token masking (like in BERT)
   - Span masking (consecutive tokens)
   - Structured masking (task-specific patterns)

5. The learned representations can be used for:
   - Pre-training for downstream tasks
   - Feature extraction
   - Transfer learning

This approach forces the model to learn meaningful representations by reconstructing missing or corrupted parts of the input using available context.



