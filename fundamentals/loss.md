# Loss

Loss function is a non-negative function primarily used during the training phase of a model. In each training batch, the model processes inputs to produce predictions through forward propagation. The loss function then calculates the discrepancy, called **loss**, between these predictions and the actual values. The model uses this loss to update its parameters via backpropagation, aiming to reduce the discrepancy between predictions and actual values, thereby achieving its learning objectives.

## Distance-based

$n$ is the number of samples, $\hat{y_i}$ is the predictions, $y_i$ is the truth.

### MSE (same as L2 in PyTorch)

$$
MSE=\frac{1}{n}\sum^{n}_{i=1}(y_i-\hat{y_i})^2
$$

```python
def l2_loss(y_pred, y_true):
    differences = y_pred - y_true
    squared_differences = differences ** 2
    loss = np.mean(squared_differences)
    return loss
```

Mostly used in regression.

When the difference is greater than $1$, the error will be amplified. When the difference is less than $1$, the error will be reduced. It shows more sensitive to outlier points.

### L1 (same as MAE in PyTorch)

$$
L1=\frac{1}{n}\sum^{n}_{i=1}\left|y_i-\hat{y_i}\right|
$$

```python
def l1_loss(y_pred, y_true):
    differences = np.abs(y_pred - y_true)
    loss = np.mean(differences)
    return loss
```

Mostly used in regression.

Because of the discontinuity of the derivative of the L1 loss function at $0$, its derivative will be $-1$ (if the predicted value is less than the true value) or $+1$ (if the predicted value is greater than the true value) near $0$. This leads to a phenomenon: even if the predicted values of two samples are very close to the true values (resulting in a small loss), but if they cross $0$ (one is larger than $0$ while the other is less than $0$), the difference between them results in large gradients. This makes the optimization process with L1 loss more unstable as it can lead to sudden changes in gradients, affecting the performance of optimization algorithms.

### Smooth L1

[Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)

$$
\begin{equation*}
SmoothL1=
\begin{cases}
0.5x^2 &\text{if }\left|x\right|<1\\
\left|x\right|-0.5 &\text{otherwise}
\end{cases}
\end{equation*}
$$

```python
def smooth_l1_loss(y_pred, y_true):
    differences = np.abs(y_pred - y_true)
    loss = np.where(diff < 1, 0.5 * diff**2, diff - 0.5)
    loss = np.mean(loss)
    return loss
```

Mostly used in object detection.

It combines the smoothness of L2 loss with the robustness to outliers of L1 loss.

### IOU

$$
IoU=\frac{Area\space of\space Intersection}{Area\space of\space Union}
$$

The IoU is calculated as the ratio of the area of intersection to the area of union between two bounding boxes, which is a widely used metric for evaluating the accuracy of object detection. The loss is mostly $1-IoU$.

## Probability-based

$C$ is the number of categories, $\hat{y}$ is the output probability distribution, $i$ is the certain category, $y$ is the true label probability distribution, represented in **one-hot encoding**.

### Cross-Entropy

$$
CrossEntropy=-\sum^{C}_{i=1}y_ilog(\hat{y_i})
$$

```python
def cross_entropy_loss(y_pred, y_true):
    """
    Parameters:
    y_pred: numpy array, predictions after softmax transformation
    y_true: numpy array, truth in one-hot encoding
    
    Returns:
    loss: float, cross-entropy loss
    """
    # compute cross-entropy loss
    num_samples = y_pred.shape[0]
    logprobs = -np.log(y_pred[range(num_samples), y_true])
    loss = np.mean(logprobs)
    
    return loss
```

### Softmax

In most practical applications, the Softmax function is first used to convert the original output of the model into a probability distribution for subsequent Cross-Entropy loss. $z_i$ is the original output of the $i^{th}$ class predicted by the model.

$$
\hat{y_i} = \frac{e^{z_i}}{\displaystyle\sum_{j=1}^C e^{z_j}}
$$

```python
def softmax_loss(y_pred, y_true):
    """
    Parameters:
    y_pred: numpy array, predictions
    y_true: numpy array, truth in one-hot encoding
    
    Returns:
    loss: float, softmax loss
    """
    # compute softmax function
    exp_scores = np.exp(y_pred)
    y_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # compute cross-entropy loss
    loss = cross_entropy_loss(y_probs, y_true)
    
    return loss
```

Mostly used in classification problem, not suitable for regression problems.

If there is noise or outliers in the training data, the model will try to overfit these bad samples on the training set. In the case of class imbalance, the model may be biased toward classes with a larger number of samples and ignore classes with a smaller number of samples.

### Focal

[Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)

Because $y$ is represented in one-hot encoding, Cross-Entropy loss can be redefined when $\hat{y_i}=p$

$$
\begin{equation*}
CrossEntropy(p,y)=
\begin{cases}
-log(p) &\text{if }y=1\\
-log(1-p) &\text{otherwise}
\end{cases}
\end{equation*}
$$

$$
\begin{equation*}
p_t=
\begin{cases}
p &\text{if }y=1\\
1-p &\text{otherwise}
\end{cases}
\end{equation*}
$$

$$
Focal=-\alpha(1-p_t)^{\gamma}log(p_t)
$$

```python
def focal_loss(y_pred, y_true, alpha=0.25, gamma=2):
    """
    Parameters:
    y_pred: numpy array, predictions after softmax transformation
    y_true: numpy array, truth in one-hot encoding
    alpha: float, balancing factor between positive and negative samples
    gamma: float, focusing parameter to adjust the weight for easy samples
    
    Returns:
    loss: float, focal loss
    """
    # compute cross-entropy loss
    ce_loss = cross_entropy_loss(y_pred, y_true)
    
    # compute modulating factor
    modulating_factor = (1 - y_pred) ** gamma
    
    # compute balanced focal loss
    loss = alpha * modulating_factor * ce_loss
    
    return loss
```

The proposal of focal loss originated from the problem of imbalance in the number of samples in target detection tasks in the image field.

### KL

Assume that the true distribution is $P(x)$ and the approximate distribution is $Q(x)$​. 

$$
D_{KL}(P||Q)=\sum_{x}P(x)log(\frac{P(x)}{Q(x)})
$$

Normally, $P(x)$ is the normalized distribution of $y$ in one-hot encoding and $Q(x)$ is the distribution of $\hat{y_i}$ after softmax.

```python
def kl_divergence(y_pred, y_true):
    """
    Parameters:
    y_pred: numpy array, predictions after softmax transformation
    y_true: numpy array, truth in one-hot encoding
    
    Returns:
    kl_div: float, kullback-leibler divergence
    """
    # compute kl divergence
    kl_div = np.sum(y_true * np.log(y_true / y_pred))
    
    return kl_div
```

Easy to find $D_{KL}(P||Q) \neq D_{KL}(Q||P)$, which means KL divergence is without symmetry. So minimizing $D_{KL}(P||Q)$ and minimizing $D_{KL}(Q||P)$ will give different results. This means during the optimization, the model may tend to adjust certain parameters to minimize the divergence in one direction while ignoring the other direction. Adjustments in these two directions will offset or conflict with each other, making the optimization difficult.

Expanding $D_{KL}(P||Q)$ can get a formula with both relative-entropy and cross-entropy $D_{KL}(P||Q)=H_{PQ}(X)-H_{Q}(X)$. The first component $H_{PQ}(X)$ is the cross-entropy between the true distribution and the predicted distribution, which indicates the model's fitting performance. A smaller value suggests better performance. The second component $H_{Q}(X)$ is the entropy of the predicted distribution, which indicates how confident the model is about its predictions. A larger value indicates more uncertainty in the predictions. 

It can also be seen from here that the optimization goals of these two parts are not always consistent. When the predictions are very different from the true distribution, reducing cross-entropy means fitting more samples, resulting in a decrease in generalization ability, thereby increasing the uncertainty of the model when facing new data. Meanwhile, increase on the second component makes model less confident with its predictions, which will hinder the model attempting to reduce cross-entropy to improve prediction accuracy.

### JS

$$
M=\frac{1}{2}(P+Q)
$$

$$
D_{JS}(P||Q)=\frac{1}{2}(D_{KL}(P||M)+D_{KL}(Q||M))
$$

```python
def js_divergence(y_pred, y_true):
    """
    Parameters:
    y_pred: numpy array, predictions after softmax transformation
    y_true: numpy array, truth in one-hot encoding
    
    Returns:
    js_div: float, jensen-shannon divergence
    """
    # normalize true labels
    y_true_norm = y_true / np.sum(y_true, axis=1, keepdims=True)
    
    # compute average distributions
    m = 0.5 * (y_true_norm + y_pred)
    
    # compute kl divergences
    kl_pm = kl_divergence(y_true_norm, m)
    kl_qm = kl_divergence(y_pred, m)
    
    # compute js divergence
    js_div = 0.5 * (kl_pm + kl_qm)
    
    return js_div
```

Mostly used in GAN. 

It eliminates the asymmetry of KL divergence by first calculating the mean of two probability distributions and then calculating the KL divergence between the mean distribution and the two original distributions, and avoids the problem of the denominator or numerator being $0$ during calculations.
