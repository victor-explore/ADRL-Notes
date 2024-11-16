# Precursor: Stochastic Differential Equations (SDEs)
## Motivation
We explored different sampling methods for generative models:

1. Langevin dynamics (LE):
   $$x_{t+1} = x_t + \alpha \nabla_x \log p(x_t) + \sqrt{2\alpha}\epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

2. DDPM (Denoising Diffusion Probabilistic Models):
   $$x_{t+1} = \sqrt{1-\beta_t}x_t + \sqrt{\beta_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

3. SMLD (Score Matching with Langevin Dynamics):
   $$x_{t+1} = x_t + \sqrt{\sigma_{t+1}^2 - \sigma_t^2}\epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

A natural question arises: What is the unifying mathematical framework that connects all these sampling methods?

The answer lies in Stochastic Differential Equations (SDEs). We will show that all these methods are actually discrete approximations of continuous-time SDEs. Understanding this connection provides several benefits:

1. A unified theoretical framework for analyzing different sampling methods
2. Better understanding of the relationship between different approaches
3. Potential for developing new sampling methods by working with SDEs directly

## Discrete sampling steps as approximations of continuous processes

Let's start by understanding how discrete sampling steps can be viewed as approximations of continuous processes with help of examples.

### Example: From Discrete to Continuous Time

To understand how discrete sampling steps can be viewed as approximations of continuous processes, let's consider the following example where the discrete process is defined as:

$$x_i = (1-\frac{\beta}{2})x_{i-1}$$

Also, we have:
- $\Delta t = \frac{1}{N}$ : The time step size, where N is the total number of steps
- $i = \frac{t}{\Delta t}$ : The current step index, where t is the continuous time variable
- $\beta_t$ : The noise schedule at time t
- $x_i$ : The state at step i
- $x_{i-1}$ : The state at previous step
- $\epsilon \sim \mathcal{N}(0,I)$ : Random noise sampled from standard normal distribution
- $I$ : Identity matrix
  
Let's convert this discrete process to continuous time:
1. First, we can write $x_i$ in terms of continuous time $t$:
   $$x_i = x(\frac{i}{N}) = x(t)$$

2. Similarly for the next step:
   $$x_{i-1} = x(\frac{i-1}{N}) = x(t - \Delta t)$$

3. The discrete equation can be rewritten as:
   $$x(t + \Delta t) = (1-\frac{\beta\Delta t}{2})x(t)$$
   
   Note: We multiply $\beta$ by $\Delta t$ because $\beta$ represents a rate of change per unit time. 
   When we discretize time into small steps $\Delta t$, we need to scale $\beta$ accordingly to get 
   the correct amount of change for that time step. Without this scaling, the discrete steps would 
   not properly approximate the continuous process as $\Delta t \to 0$.

4. This is equivalent to:
   $$x(t + \Delta t) - x(t) = -\frac{\beta\Delta t}{2}x(t)$$

5. Dividing both sides by $\Delta t$:
   $$\frac{x(t + \Delta t) - x(t)}{\Delta t} = -\frac{\beta}{2}x(t)$$

6. Taking the limit as $\Delta t \to 0$:
   $$\lim_{\Delta t \to 0} \frac{x(t + \Delta t) - x(t)}{\Delta t} = -\frac{\beta}{2}x(t)$$

7. The left side is the definition of the derivative, so we get:
   $$\frac{dx}{dt} = -\frac{\beta}{2}x(t)$$

8. This ODE has the solution:
   $$x(t) = e^{-\frac{\beta}{2}t}$$

   This represents exponential decay of the state over time.

This is our continuous-time ordinary differential equation (ODE).

### Another example
Let's look at another example of converting a discrete equation to continuous form:

1. Starting with the discrete equation:
   $$x_i = x_{i-1} - \beta_{i-1}\nabla f(x_{i-1})$$
   where $\beta$ is the gradient descent step size

2. Make it continuous by substituting:
   $$x_i = x(\frac{i}{N}), \quad \Delta t = \frac{1}{N}, \quad \beta_{i-1} = \beta(t)\Delta t$$

3. This gives us:
   $$x(t + \Delta t) = x(t) - \beta(t)\Delta t\nabla f(x(t))$$

4. Rearranging to get the differential form:
   $$\frac{x(t + \Delta t) - x(t)}{\Delta t} = -\beta(t)\nabla f(x(t))$$

5. Taking the limit as $\Delta t \to 0$:
   $$\frac{dx(t)}{dt} = -\beta(t)\nabla f(x(t))$$

6. Finally, we can write this in the more compact differential form:
   $$dx = -\beta(t)\nabla f(x(t))dt$$

This demonstrates how a discrete gradient descent step can be viewed as a discretization of a continuous ordinary differential equation (ODE).