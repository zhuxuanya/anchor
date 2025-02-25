# Layers

Layer is the basic unit of the network, and each layer is composed of multiple neurons. Generally, a network usually consists of **input layer**, **hidden layer** and **output layer**. The input layer is responsible for receiving input data. Each input feature usually corresponds to a neuron. The hidden layer is located between the input layer and the output layer and may contain multiple layers. It is responsible for extracting features from the data and providing a more abstract data expression for the output layer. The output layer is the last layer of the network, and its number of neurons usually corresponds to the output of the task, such as the number of categories in a classification problem. 

## Convolutional Layer

Convolutional layers extract features using matrices called convolution kernels. By computing element-wise multiplications and summing them up at each position, in this layer, each neuron is not connected to all neurons in the previous layer, but only to a local region in the input data, which is called **local connectivity**. Meanwhile, the weight of the kernel is not only applied locally, but also reused at different locations throughout the input data, which is called **weight-sharing** .

Since there is no need to set independent weights for each neuron, these two features greatly reduce the number of parameters in the model, thereby reducing the risk of overfitting. And this method can not only capture important local features such as edges and corners, but also ensure that the same convolution kernel can recognize the same pattern at any position in the input data, improving generalization capabilities. Both of these are huge advantages when dealing with spatially correlated data, such as images.

Convolutional layers are typically positioned at the beginning, directly processing raw pixel data. Stacking multiple convolutional layers allows the network to extract hierarchical features from low to high level. These layers are often followed by pooling layers to reduce dimensionality and enhance feature invariance and fully connected layers for decision-making tasks. The output can be expressed as

$$
Y=f(K∗X+b)
$$

$X$ is the input feature map, $f$ is the activation function, $K$ is the convolutional kernels, $*$ is the convolution operation, $b$​ is the bias.

## Pooling Layer

The pooling layer is usually located after one or more convolutional layers. It performs downsampling operations on the input feature map to extract more abstract feature information, thereby reducing the size of the feature map and increasing the model's robustness to small changes in object position in input data.

This is usually done by sliding a fixed-size window over the feature map and then applying a specific function such as maximum or average within the window.

## Fully Connected Layer

In a fully connected layer, each neuron is connected to all neurons in the previous layer. Its primary function is to integrate features from preceding layers, such as convolutional or pooling layers, for classification decisions at the end of the network. This integration is inherently linear, involving a weight matrix and biases. To introduce non-linearity, an activation function is typically used after this layer to facilitate transformations. The output can be expressed as

$$
Y=f(WX+b)
$$

$X$ is the input vector, $f$ is the activation function, $W$ is the weight matrix, $b$ is the bias.

## Activation Layer

Activation layers introduce non-linearity to neural networks through activation functions, enabling the network to learn complex patterns. Without activation functions, the network output would be merely a linear combination of inputs, limiting its expressive power. Additionally, activation functions control the information flow, such as ReLU preventing passing negative values to avoid vanishing gradients; Sigmoid and Tanh confine outputs within a fixed range, enhancing model stability.

For a detailed introduction, refer to [activation function](./activation_function.md).
