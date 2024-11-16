# Convex conjugate (also known as the Fenchel conjugate or the Legendre transform)

## Definition

Let $f: R^n → R$ be a convex function. The convex conjugate of $f$, denoted as $f^*$, is defined as:

$$f^*(y) = \underset{x \in \text{dom} f}{\sup} (y^T x - f(x))$$

Where:
- $y \in R^n$ is the variable
- $\text{dom} f$ is the domain of $f$
- $\sup$ denotes the supremum (least upper bound)

## Properties

### 1. **Convexity**: $f^{**}(x)$ is also convex
   
Proof: Let $y_1, y_2 \in \mathbb{R}^n$ and $\lambda \in [0,1]$. We need to show that: 

$$f^*(\lambda y_1 + (1-\lambda)y_2) \leq \lambda f^*(y_1) + (1-\lambda)f^*(y_2)$$

1. The definition states: $f^*(y) = \sup_{x \in \text{dom} f} (y^T x - f(x))$
2. Here, our $y$ is $\lambda y_1 + (1-\lambda)y_2$
3. Substituting this into the definition gives us:
   $$f^*(\lambda y_1 + (1-\lambda)y_2) = \sup_{x} \{(\lambda y_1 + (1-\lambda)y_2)^T x - f(x)\}$$

Next step will distribute the transpose operation and separate the terms:

$$f^*(\lambda y_1 + (1-\lambda)y_2) = \sup_{x} \{\lambda y_1^T x + (1-\lambda)y_2^T x - f(x)\}$$

$$f^*(\lambda y_1 + (1-\lambda)y_2) = \sup_{x} \{\lambda (y_1^T x - f(x)) + (1-\lambda)(y_2^T x - f(x))\}$$

The supremum of a sum is less than or equal to the sum of the suprema:

$$f^*(\lambda y_1 + (1-\lambda)y_2) \leq  \sup_{x} \lambda\{y_1^T x - f(x)\} + \sup_{x} (1-\lambda) \{y_2^T x - f(x)\}$$

$$f^*(\lambda y_1 + (1-\lambda)y_2) \leq  \lambda \sup_{x} \{y_1^T x - f(x)\} + (1-\lambda) \sup_{x} \{y_2^T x - f(x)\}$$

$$f^*(\lambda y_1 + (1-\lambda)y_2) \leq \lambda f^*(y_1) + (1-\lambda)f^*(y_2)$$

Thus, $f^*$ is convex.

### 2. **Conjugate of conjugate**: $f^{**}(y) = f(y)$

Proof: 

We know that:
$$f^*(y) = \underset{x \in \text{dom} f}{\sup} (y^T x - f(x))$$

Therefore:
$$f^{**}(y) = \underset{x}{\sup} \{y^T x - f^*(x)\}$$

$$f^{**}(y) = \underset{x}{\sup} \{y^T x - \underset{z \in \text{dom} f}{\sup} (x^T z - f(z))\}$$

$$f^{**}(y) = \underset{x}{\sup} \underset{z \in \text{dom} f}{\inf} \{y^T x - (x^T z - f(z))\}$$

$$f^{**}(y) = \underset{x}{\sup} \underset{z \in \text{dom} f}{\inf} \{y^T x - x^T z + f(z)\}$$

$$f^{**}(y) = \underset{x}{\sup} \underset{z \in \text{dom} f}{\inf} \{x^T(y - z) + f(z)\}$$

Now, we can apply the minimax theorem, which allows us to swap the order of sup and inf:

$$f^{**}(y) = \underset{z \in \text{dom} f}{\inf} \underset{x}{\sup} \{x^T(y - z) + f(z)\}$$

The inner supremum is unbounded unless $y = z$, in which case it equals $f(z)$. Therefore:

$$f^{**}(y) = \underset{z \in \text{dom} f}{\inf} \begin{cases} 
f(z) & \text{if } y = z \\
\infty & \text{otherwise}
\end{cases}$$

This simplifies to $f^{**}(y) = f(y)$ because:

1. When $y \neq z$, the infimum over $z$ where $f^{**}(y) = \infty$ cannot be the actual infimum, since we have at least one finite value when $y = z$

2. When $y = z$, we get $f^{**}(y) = f(z) = f(y)$

3. Therefore, the infimum must occur at $y = z$, giving us $f^{**}(y) = f(y)$

Thus, we have shown that $f^{**}(y) = f(y)$, completing the proof.
