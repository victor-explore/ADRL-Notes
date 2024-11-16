
# F-Divergence

F-divergence is a generalized measure of difference between two probability distributions. For probability distributions $P$ and $Q$ over a event space $X$, the F-divergence is defined as:

$$D_f(P || Q) = ∫_X q(x) f(\frac{p(x)}{q(x)}) dx$$
Where:
- **Generator function**  $f: R^+ → R$ is a lower semi-continuous convex function with $f(1) = 0$. The F in F-divergence comes from generator "f"unction.
- $p(x)$ and $q(x)$ are the probability density functions of $P$ and $Q$ respectively

Some examples of F-divergences include:
- Kullback-Leibler divergence: 
  - $D_{KL}(P || Q) = ∫_X p(x) log(\frac{p(x)}{q(x)}) dx$ 
  - when $f(x) = x log(x)$
  - Note that KL divergence is just a special case of F-divergence
- Reverse Kullback-Leibler divergence: 
  - $D_{KL}(Q || P) = ∫_X q(x) log(\frac{q(x)}{p(x)}) dx$ 
  - when $f(x) = -log(x)$
- Jensen-Shannon divergence: 
  - $D_{JS}(P || Q) = \frac{1}{2}\int_X p(x) \log\left(\frac{2p(x)}{p(x)+q(x)}\right) + q(x) \log\left(\frac{2q(x)}{p(x)+q(x)}\right) dx$ 
  - when $f(x) = -(x+1)\log\left(\frac{1+x}{2}\right) + x \log(x)$

## Properties of F-divergences
### 1. F-divergence is always non-negative, i.e., $D_f(P || Q) ≥ 0$
   
Proof: 
By Jensen's inequality, since $f$ is convex and $p(x)$ and $q(x)$ are probability densities:

$$∫_X q(x)f(\frac{p(x)}{q(x)})dx ≥ f(∫_X q(x)(\frac{p(x)}{q(x)})dx)$$

$$∫_X q(x)f(\frac{p(x)}{q(x)})dx ≥ f(∫_X p(x)dx)$$

$$∫_X q(x)f(\frac{p(x)}{q(x)})dx ≥ f(1) = 0$$ 

$$D_f(P || Q) ≥ 0$$ 
Hence, F-divergence is always non-negative.

### 2. F-divergence is 0 if and only if P = Q

Proof:
We'll prove both directions:
- If $P = Q$, then $D_f(P || Q) = 0$, and
- If $D_f(P || Q) = 0$, then $P = Q$.

#### First, if $P = Q$, then $D_f(P || Q) = 0$:

   When $P = Q$ then $p(x) = q(x)$ for all $x$. Hence:
   
   $$D_f(P || Q) = \int_X q(x) f(\frac{p(x)}{q(x)}) dx$$
   $$D_f(P || Q) = \int_X q(x) f(\frac{p(x)}{p(x)}) dx$$
   $$D_f(P || Q) = \int_X q(x) f(1) dx$$
   $$D_f(P || Q) = f(1) \int_X q(x) dx$$
   $$D_f(P || Q) = f(1) * 1 = 0$$
   
   Since $f(1) = 0$ by definition of F-divergence.

#### Second, if $D_f(P || Q) = 0$, then $P = Q$:

   Assume $D_f(P || Q) = 0$. This means:   
   $$\int_X q(x) f(\frac{p(x)}{q(x)}) dx = 0$$

   Since $q(x) \geq 0$ for all $x$, and $f$ is convex, the integral can only be zero if:
   $$f(\frac{p(x)}{q(x)}) = 0 \text{ for all } x \in X$$

   Given that $f(1) = 0$ and $f$ is convex, the only way $f(\frac{p(x)}{q(x)}) = 0$ is if:
   $$\frac{p(x)}{q(x)} = 1 \text{ for all } x \in X$$

   This means that:
   $$p(x) = q(x) \text{ for all } x \in X$$

   Thus 
   $$P = Q$$

Therefore, $D_f(P || Q) = 0$ if and only if $P = Q$.

### 3. F-divergence is not symmetric, i.e., $D_f(P || Q) ≠ D_f(Q || P)$ in general

Proof:
Consider the following counterexample:

Let $X = \{0, 1\}$, and let $P$ and $Q$ be the following probability distributions:

$P(0) = 0.3, P(1) = 0.7$
$Q(0) = 0.7, Q(1) = 0.3$

Let $f(x) = x \log(x)$ (KL divergence).

Then:

$D_f(P || Q) = \int_X p(x) f(\frac{p(x)}{q(x)}) dx$
$= 0.3 f(\frac{0.3}{0.7}) + 0.7 f(\frac{0.7}{0.3})$
$= 0.3 (\frac{0.3}{0.7} \log(\frac{0.3}{0.7})) + 0.7 (\frac{0.7}{0.3} \log(\frac{0.7}{0.3}))$
$\approx -0.1209 + 0.9650 = 0.8441$

$D_f(Q || P) = \int_X q(x) f(\frac{q(x)}{p(x)}) dx$
$= 0.7 f(\frac{0.7}{0.3}) + 0.3 f(\frac{0.3}{0.7})$
$= 0.7 (\frac{0.7}{0.3} \log(\frac{0.7}{0.3})) + 0.3 (\frac{0.3}{0.7} \log(\frac{0.3}{0.7}))$
$\approx 1.3786 - 0.0518 = 1.3268$

In this case, $D_f(P || Q) \approx 0.8441 \neq D_f(Q || P) \approx 1.3268$.

Therefore, F-divergence is not symmetric in general.




