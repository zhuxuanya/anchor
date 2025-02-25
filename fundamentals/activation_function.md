# Activation Function

The main purpose of the activation function is to introduce **non-linearity** into the neural network. Without an activation function, no matter how many layers the neural network has, the output is just a linear combination of the inputs, which will greatly reduce the expressive ability of the network. 

## Sigmoid

$$
\sigma(x)=\frac{1}{1+e^{-x}}\\
$$

$$
\sigma ^\prime(x)=\sigma(x)(1-\sigma(x))
$$

The output range is $(0,1)$, making it suitable to use as the activation function in the output layer of binary classification problems. However, there are two main issues.

When $|x|$ is large, the gradients are very small, almost zero, which can cause **vanishing gradients**. Since its outputs have an average value close to $0.5$ rather than zero, during gradients descent, all outputs tend to move in the same direction, which can cause unnecessary oscillations during the optimization.

## Tanh

$$
tanh(x)=\frac{1-e^{-2x}}{1+e^{-2x}}
$$

$$
tanh^\prime(x)=1-tanh^2(x)
$$

The output range is $(-1,1)$, making it zero-centered unlike the Sigmoid function. However, it may still cause the **vanishing gradients** when $|x|$ is very large.

## ReLU

$$
ReLU(x)=max(0,x)
$$

$$
\begin{equation*}
ReLU^\prime(x)=
\begin{cases}
0 &x\leq0\\
1 &x>0
\end{cases}
\end{equation*}
$$

The ReLU function involves only a comparison and the selection operation without any complex mathematical operations like exponentiation or division. This simplicity leads to sparsity in the activation outputs, with all negative values set to zero, which enhances the network's computational efficiency and reduces the computational resources needed.

It is especially suitable for large-scale networks and complex tasks such as image and video processing, natural language processing.

There is still a significant drawback, where neurons that receive negative input will output zero and thus have a zero gradient. During training, if these neurons consistently receive negative inputs, they stop updating permanently.

## Leaky ReLU

$$
Leaky\space ReLU(x)=max(0.01x,x)
$$

$$
\begin{equation*}
Leaky\space ReLU^\prime(x)=
\begin{cases}
0.01 &x\leq0\\
1 &x>0
\end{cases}
\end{equation*}
$$

Leaky ReLU keeps neurons active during training by providing a small positive slope for negative inputs.

## Softmax

$$
Softmax(x_i)=\frac{e^{x_i}}{\displaystyle\sum_{j} e^{x_j}}
$$

$$
Softmax^\prime(x_i)=Softmax(x_i)(\delta_{ik}-Softmax(x_k))
$$

$$
\begin{equation*}
\delta_{ik}=
\begin{cases}
1 &i=k\\
0 &i\neq k
\end{cases}
\end{equation*}
$$

Its main purpose is to convert a set of values, called logistic or log odds, into a true probability distribution, where the probabilities for each class sum to one. Specifically, using the Softmax function as the activation function in combination with cross-entropy as the loss function can greatly simplify the gradients computation during the backpropagation process.

$$
\frac{\partial L}{\partial z_i}=\hat{y_i}-y_i
$$
