# Fundamentals

This directory focuses on the core concepts of machine learning, particularly neural networks.

## Supervised Learning

This section focuses on supervised learning, particularly the training of neural networks for tasks such as classification and regression. Below is a detailed pipeline for one cycle of parameter updates during training.

**1. Data Preprocessing**

Before training begins, data must be preprocessed, which includes normalization, standardization, and possibly data augmentation to improve model generalization and training speed.

**2. Weight Initialization**

Weights in the network are initialized before training begins. Choosing an appropriate weight initialization method is crucial for avoiding issues like vanishing or exploding gradients, and it significantly influences the model's convergence speed and final performance.

**3. Forward Propagation**

Data is fed into the network, passing through the input, hidden, and output layers. At each layer, the data is multiplied by weights, biases are added, and an activation function is applied. Batch normalization may also be used to stabilize training and speed up convergence.

**4. Loss Calculation**

The predictions from the output layer are compared with the actual labels, and the difference between them is quantified using a loss function. This function calculates the prediction error, which serves as a measure of the model's performance. Minimizing this loss is the primary goal during training

**5. Backpropagation**

During backpropagation, the gradient of the loss function with respect to each weight is calculated and propagated back through the network. This allows the weights to be updated in each layer based on how much they contributed to the error. The chain rule is used for efficient gradient computation, enabling this process to work layer by layer.

**6. Optimization**

After completing the forward and backward passes for the entire network, the optimizer adjusts all the model's parameters based on the gradients calculated during backpropagation. These updates aim to minimize the loss and improve the model's performance. The optimizer performs this update after each iteration until training is complete.
