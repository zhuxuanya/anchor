# Optimizer

Optimizers are **algorithms** used to adjust the weights by the computed gradients of neural networks in order to minimize the loss function. The choice of optimizer can significantly affect the performance and convergence speed of the model.

## GD - (Batch) Gradient Descent

GD is a fundamental optimization algorithm used for minimizing a loss function, represented as $J(\theta)$, where $\theta$ denotes the model parameters. The core idea is to update the parameters in the direction that reduces the loss function, using the gradient of $J$ with respect to $\theta$. Normally, it uses the entire dataset to compute the gradient of the loss function. The parameters are updated as follows:

$$
\theta = \theta - \eta \nabla_\theta J(\theta)
$$

Here, $\eta$ is the learning rate, and $\nabla_\theta J(\theta)$ is the gradient of $J$ with respect to $\theta$. As it processes all data at once, it is time-consuming for computing the gradient across the large data sets.

## SGD - Stochastic Gradient Descent

### Advantages

SGD allows faster training. It updates the parameters using the gradient calculated from only one randomly selected sample $(x^{(i)}, y^{(i)})$ in each iteration, which is suitable for online learning and large-scale data sets. Additionally, the randomness helps the model escape local minima and potentially find global minima. The parameters are updated as follows:

$$
\theta = \theta - \eta \nabla_\theta J(\theta; x^{(i)}, y^{(i)})
$$

### Disadvantages

The use of a single sample update leads to high fluctuation in the learning path, making the loss unstable during convergence. It might require more steps to converge even if each step is faster. Despite its randomness, SGD can still get stuck in local optima.

## Momentum

Momentum is an optimization technique used to accelerate gradient descent algorithms, especially in the presence of high-dimensional spaces and non-convex surfaces. It introduces the concept of **inertia** to the updates, incorporating not only the current gradient but also considering the direction of the previous updates. This helps the optimizer accelerate in the correct direction and reduces oscillations.

### Update Rule

In traditional gradient descent, the parameter update rule is:

$$
\theta = \theta - \eta \nabla_\theta J(\theta)
$$

With momentum, the update rule becomes:

$$
v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)
$$

$$
\theta = \theta - v_t
$$

Here, $v_t$ is the current update vector, $\gamma$ (often referred as the momentum factor) is a parameter between zero and one that controls the influence of previous gradients.

### Advantages

Momentum accelerates the convergence of gradient descent by enhancing movement in relevant directions and reducing fluctuations in non-relevant ones, thus allowing for a faster approach to the optimal solution. Additionally, the accumulation of momentum helps the optimizer overcome local minima, which is especially helpful for loss function with irregular surfaces. By considering historical gradients, momentum also significantly reduces oscillations in narrow regions, enhancing the overall stability of the algorithm.

### Disadvantages

Momentum is highly sensitive to the choice of $\gamma$. Inappropriate values can lead to overshoot the optimum or even cause the algorithm to diverge. Additionally, while momentum can assist in escaping local minima, its inertia might also lead the algorithm to cross the global minima, thereby missing the optimal solution.

## Adagrad - Adaptive Gradient

Adagrad is an optimization algorithm designed for efficiently handling sparse data by adapting the learning rate for each parameter individually based on their accumulated historical gradients. 

### Update Rule

The core mechanism of Adagrad allows each parameter to have its own learning rate that adapts over time. This is achieved through the update formula:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot g_t
$$

Where $\theta_t$ is the parameter vector at time $t$. $\eta$ is the initial learning rate. $g_t$ is the gradient of the loss function with respect to $\theta_t$ at time $t$. $G_t$ is a diagonal matrix where each diagonal element is the gradient square sum with respect to $\theta_{t,i}$ up to time $t$. $\epsilon$ is a small smoothing term added to prevent division by zero, typically around $1e-8$.

### Advantages

The adaptive adjustment of learning rates in Adagrad means that if a parameter’s historical gradients are large, suggesting significant influence on the loss function, its accumulated gradient square sum will be large. Thus, dividing the learning rate by the square root of this sum results in a smaller learning rate for that parameter. This approach reduces the step size for parameters with large gradients to prevent overshooting. Conversely, parameters with smaller historical gradients have a larger learning rate, enabling more substantial updates where needed. 

### Disadvantages

The primary drawback is its tendency to lower the learning rates too aggressively due to the continuous accumulation of squared gradients in the denominator of the update formula. Over time, this can lead to an ever-decreasing learning rate, potentially becoming so minimal that the algorithm almost stops updates, a phenomenon often referred as premature convergence. Additionally, once the learning rate for a parameter has been reduced, it becomes challenging to increase it again. This inflexibility can be particularly problematic in dynamic environments or during the later stages of training.

## RMSprop - Root Mean Square Propagation

RMSprop is an optimization algorithm designed to address the rapid decline in learning rates found in Adagrad. RMSprop modifies the accumulation of the squared gradients by introducing a decay factor, thus overcoming the problem of diminishing learning rates in Adagrad.

### Update Rule

The key idea of RMSprop is the use of an exponential weighted moving average of the squared gradients, rather than a simple accumulation. This means that all historical gradients won't share equal weight. More recent gradients have a greater impact than older ones. This approach allows for quicker adjustments in the training and provides greater adaptability across different stages of the training process.

The parameter update formula for RMSprop is as follows:

$$
G_t = \beta G_{t-1} + (1 - \beta)g_t^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot g_t
$$

Where $\beta$ is the decay rate, typically set around 0.9.

### Advantages

The main advantage of RMSprop is that it addresses the rapid decline in learning rates seen in Adagrad, allowing for a more stable and effective learning rate across multiple training phases. Additionally, by adjusting the decay rate $\beta$, the model's sensitivity to changes in historical gradients can be controlled, making it more flexible and robust when facing various types of data and tasks.

## Adam - Adaptive Moment Estimation

Adam is a highly popular optimization algorithm. It combines the advantages of Momentum and RMSprop by utilizing estimates of the first moments (i.e., the gradients) and the second moments (i.e., the squared gradients) of the gradients to adjust the learning rates for each parameter.

### Update Rule

The Adam optimizer tracks the exponential moving averages of the gradients (similar to momentum) and the squared gradients (similar to RMSprop). The update rules for Adam are as follows:

1. Calculate the exponential moving average of the gradients, known as the first moment estimate:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

2. Calculate the exponential moving average of the squared gradients, known as the second moment estimate:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

3. Correct the estimates of the first and second moments for bias:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}\\
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

4. Use the bias-corrected first and second moment estimates to update the parameters:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t 
$$

Where $g_t$ is the gradient at timestep $t$, $\beta_1$ and $\beta_2$ are decay rate constants typically set to 0.9 and 0.999 respectively, $\eta$ is the learning rate, and $\epsilon$ is a very small number to prevent division by zero.

### Advantages

Adam automatically adjusts the learning rates for each parameter by calculating first and second moment estimates, making it suitable for different phases of training. Larger updates in the early stages help make rapid progress; smaller updates when approaching the optimum help convergence. Compared to other optimization algorithms, Adam requires less manual tuning of hyperparameters, and default parameters often yield good results.
