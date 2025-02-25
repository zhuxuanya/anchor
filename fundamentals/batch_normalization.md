# Batch Normalization

Batch normalization is a technique used during training, commonly applied after convolutional or fully connected layers and before activation functions, aimed at accelerating the training process of neural networks and enhancing stability.

## Training Phase

Batch normalization involves several key steps that are applied to each mini-batch of data:

**1. Calculate Mean and Variance**

For the input mini-batch data, first compute the mean and variance for each feature. For example, for a specific pixel position across all images or an output value of a neuron, calculate the average and variance of this feature across the entire mini-batch.

**2. Normalization**

Normalize each feature using the mean and variance obtained in the previous step. The formula is:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

Where $x_i$ is the original data point, $\mu_B$ is the mean, $\sigma_B^2$ is the variance, and $\epsilon$ is a small number (e.g., $1e-8$) to prevent division by zero.

**3. Scaling and Shifting**

The normalized data is then scaled and shifted to potentially restore some of the original properties of the data, through parameters learned during training. The formula is:

$$
y_i = \gamma \hat{x}_i + \beta
$$

Here, $\gamma$ and $\beta$ are learnable parameters, $\gamma$ is a scaling factor, and $\beta$ is a shift amount. Their purpose is to enable the layer to restore important properties of the original data.

**4. Output**

The processed data $y_i$ is the output of the batch normalization layer, which is then passed to the next layer of the network, such as an activation function layer.

## Inference Phase

During the inference phase, batch normalization operates differently compared to the training phase. The key difference is that during inference, the mean and variance of each mini-batch cannot be used , as this would make the model's outputs dependent on the specific batch data, thereby affecting the model's generalization ability. Instead, global statistics computed during training are used.

### Using Global Statistics

**1. Global Statistics Computation**

During training, global estimates for each feature's mean and variance are maintained by computing a moving average of these statistics. This means that every time batch normalization is applied during training, not only are the current mini-batch's mean and variance calculated, but the global mean and variance are also updated.

**2. Normalization Using Global Statistics**

During inference, these global statistics are used to normalize the input data. The formula is:

$$
\hat{x} = \frac{x - \mu_{\text{global}}}{\sqrt{\sigma_{\text{global}}^2 + \epsilon}}
$$

Here, $\mu_{\text{global}}$ and $\sigma_{\text{global}}^2$ are the accumulated global mean and variance from training, and $\epsilon$ is a small constant to prevent division by zero.

**3. Scaling and Shifting Remains Applicable**

Like in training, the normalized data is scaled and shifted using the same parameters learned during training. The formula is:

$$
y = \gamma \hat{x} + \beta
$$

### Updating Global Statistics

**1. Initialize Global Statistics**

At the beginning of training, global mean and variance are initialized. These could be set to zero for the mean and one for the variance or appropriately initialized based on prior knowledge of the data.

**2. Update Global Statistics**

With each batch normalization operation during training, after computing the mean and variance for the current mini-batch, the global mean and variance are updated using a decay factor, often called momentum. The formula is:

$$
\mu_{\text{global}} = \mu_{\text{global}} \times \text{momentum} + \mu_{\text{batch}} \times (1 - \text{momentum})
$$

$$
\sigma_{\text{global}}^2 = \sigma_{\text{global}}^2 \times \text{momentum} + \sigma_{\text{batch}}^2 \times (1 - \text{momentum})
$$

Where $\mu_{\text{batch}}$ and $\sigma_{\text{batch}}^2$ are the current batch's mean and variance, and $\text{momentum}$ is a hyperparameter between 0 and 1, balancing the weight of historical and current batch information.

## Benefits

**1. Accelerates Model Convergence**

Batch normalization normalizes the inputs of each layer to have the same mean and variance, helping to **reduce internal covariate shift**. This shift refers to changes in the distribution of inputs to a layer due to changes in the parameters of previous layers. By stabilizing input distributions, batch normalization allows the use of higher learning rates, which speeds up the training process.

This popular belief that batch normalization accelerates training by reducing internal covariate shift comes from [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167). However, [How Does Batch Normalization Help Optimization?](https://arxiv.org/pdf/1805.11604) proves a more fundamental impact: batch normalization makes the optimization landscape significantly smoother. This smoothness induces a more predictive and stable behavior of the gradients, allowing for faster training.

**2. Improves Gradient Flow**

Batch normalization helps ensure that each layer's inputs are kept within the linear region of activation functions, promoting better gradient propagation.

**3. Acts as a Regularizer**

The process of normalizing inputs using the batch's mean and variance introduces noise, which acts as a form of regularization. This helps the model generalize better and reduces the risk of overfitting. Although this regularization effect is not as strong as **dropout**, it has proven beneficial in practice.

**4. Reduces Sensitivity to Initialization**

Without batch normalization, improper initialization can lead to gradient issues or non-convergence. Batch normalization reduces the model's dependency on initialization, making network training more robust.

**5. Facilitates the Use of ReLU**

Batch normalization enables the effective use of ReLU and its variants, as these activation functions do not saturate in the positive region. Normalization reduces the issues of input values becoming too large and saturating the activation functions.
