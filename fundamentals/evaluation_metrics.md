# Evaluation Metrics

## Confusion Matrix
In the confusion matrix, **True** and **False** indicate whether the prediction is consistent with the actual situation. **Positive** and **Negative** indicate the predicted category. Refer to the table below.

| **Situation** | Predicted Positive | Predicted Negative |
| :--- | :--- | :--- |
| **Actual Positive** | TP (True Positive)  | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative)  |

## Common Metrics

**Precision**

$$
Precision = \frac{TP}{TP + FP}
$$

The proportion of **true positives** among **all positive predictions**.

**Recall**

$$
Recall = \frac{TP}{TP + FN}
$$

The proportion of **true positives** among **all actual positives**. 

**Accuracy**

$$
Accuracy = \frac{TP + TN}{All}
$$

The proportion of **correct predictions** among **all samples**.

**F1 Score**

$$
F1 \space Score = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

The harmonic mean of precision and recall.

**Sensitivity / Specificity**

$$
Sensitivity = Recall = \frac{TP}{TP + FN}
$$

Sensitivity measures the proportion of **actual positives** that are correctly identified by the model.

$$
Specificity = \frac{TN}{FP + TN}
$$

Specificity measures the proportion of **actual negatives** that are correctly identified by the model. 

**TPR / FPR**

$$
TPR = Sensitivity
$$

$$
FPR = 1 - Specificity = \frac{FP}{FP + TN}
$$

TPR measures the proportion of actual positives that are correctly identified by the model. FPR measures the proportion of actual negatives that are incorrectly identified as positives by the model. 

When deciding whether to prioritize TPR or FPR, consider the following: 

Scenarios where prioritizing TPR are those in which missing a positive case cause severe consequences. For example, in medical diagnosis, it is crucial to accurately identify every patient with a disease, as missing a single case could lead to serious consequences.

Scenarios where prioritizing FPR are those in which false alarms cause severe consequences. For example, in spam email detection, false positives mean that legitimate emails are incorrectly marked as spam, causing users to miss important messages.

**ROC**

The ROC (Receiver Operating Characteristic) curve is a plot of TPR (y-axis) vs. FPR (x-axis) across different thresholds.

**AUC**

AUC (Area Under the ROC Curve) quantifies the overall ability of the model to discriminate between positive and negative classes across all possible threshold values. A higher AUC indicates better performance.

| AUC | Model | ROC |
| :--- | :--- | :--- |
| 0.5       | a model with no discriminative power, equivalent to random guessing     | lies along the diagonal line, from (0,0) to (1,1)            |
| (0.5, 1)  | a model with some level of discriminative power                         |                                                              |
| 1         | a perfect model, which correctly classifies all positives and negatives | reaches the top left corner of the plot, TPR = 1 and FPR = 0 |
