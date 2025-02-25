# Initialization

Initialization refers to how the weights and biases are set before training. Proper initialization methods can help increase training speed and avoid vanishing or exploding gradients problems, thereby improving model performance. [Documentation for torch initialization](https://pytorch.org/docs/stable/nn.init.html)

## Zero

Set all network parameters to zero, not recommended.

## Small Random Numbers

Draw weights from a normal distribution with a mean of zero and a small standard deviation (e.g., 0.01). The aim is to prevent activations and gradients from becoming too large at the start of training, thereby avoiding **exploding gradients**. This method is suitable for shallower networks but can lead to **vanishing gradients** in deeper networks.

## Uniform Distribution

The weights can be randomly selected from a specified uniform distribution, usually with a small range (such as -0.05 to 0.05). The aim of this method is similar to small random numbers initialization.

## Xavier

The aim of this method is to keep the variance of the input and output consistent in the early stages of training, thereby helping to avoid the vanishing or exploding gradients in deep networks. It is particularly suitable for networks using Sigmoid or Tanh. Specifically, the weights are randomly selected from a distribution with mean zero and variance $\frac{2}{fan_{in}+fan_{out}}$​​ . 

**Xavier Normal Initialization**: weights are initialized randomly from a normal distribution.

$$
\mathcal{N}(0,\sqrt\frac{2}{fan_{in}+fan_{out}})
$$

**Xavier Uniform Initialization**: weights are initialized randomly from a uniform distribution.

$$
\mathcal{U}(-\sqrt\frac{6}{fan_{in}+fan_{out}},\sqrt\frac{6}{fan_{in}+fan_{out}})
$$

$fan_{in}$ and $fan_{out}$ refer to the number of input and output units in the weight matrix.

## He

This method is suitable for ReLU and its variants. Using ReLU means that ideally, half of the activations in the network will be zero, effectively halving the number of active neurons in the network. Therefore, if Xavier initialization is used, the variance of the weights will gradually decrease with increasing layers, potentially leading to **vanishing gradients** in deep networks.

To solve this problem, He initialization doubles the variance of the weights to compensate for the signal attenuation caused by ReLU. He initialization typically uses a normal distribution.

$$
W\sim\mathcal{N}(0,\sqrt\frac{2}{fan_{in}})
$$
