# Fundamentals of Machine Learning

This directory focuses on the core concepts of machine learning, particularly neural networks. 

## Training Process

Training a neural network is a systematic process. Below is a detailed pipeline during one parameter update cycle in a neural network:

- **Data Preprocessing**  
  Before training begins, data must be preprocessed, which includes normalization, standardization, and possibly data augmentation to improve model generalization and training speed.

- **Weight Initialization**  
  Weights in the network are initialized. Choosing an appropriate initialization method is crucial for the model's convergence speed and final performance.

- **Forward Propagation**  
  Data is fed into the network, moving from the input layer through hidden layers to the output layer. At each layer, data is multiplied by weights, added to biases, and then passed through an activation function. Batch normalization may be used to help control gradients, improve stability, and speed up the process.

- **Loss Function Calculation**  
  Predictions at the output layer are compared with actual labels, and prediction errors are calculated using a loss function.

- **Backpropagation**  
  The gradient of the loss function is propagated back through the network to update weights in each layer. This crucial step relies on the chain rule for efficient gradient computation. Regularization techniques, such as Dropout or L2 regularization, might be employed during this process to adjust gradients and help prevent overfitting.

- **Optimizer Adjusts Weights**  
  The network's weights are adjusted based on the loss gradient and the chosen optimizor, aiming to minimize the loss.

## Additional Resources

- [**CNN Explainer**](https://github.com/poloclub/cnn-explainer)  
  An interactive visualization system that helps better understand how convolutional neural networks work through a dynamic visual representation of the training process.

- [**Netron**](https://github.com/lutzroeder/netron)  
  A viewer for neural network, deep learning, and machine learning models. It supports a variety of file formats and provides a graphical visualization of model architectures.

- [**PlotNeuralNet**](https://github.com/HarisIqbal88/PlotNeuralNet)  
  A package for drawing neural network structures in papers, especially supporting LaTeX code.
