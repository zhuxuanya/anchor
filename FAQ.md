# FAQ

## 基础知识

### 1.如何解决过拟合

过拟合是指模型在训练数据上表现良好，但在测试数据上表现不佳的现象。这通常是由于模型过于复杂，能够很好地记住训练数据中的噪声和细节，但未能捕捉到数据的普遍模式。解决的方法有：

- **增加数据量**：获取更多的训练数据以增强模型的泛化能力。
- **数据增强**：对现有数据进行随机变换，如旋转、缩放、裁剪等，增加数据多样性。
- **简化模型**：降低模型复杂度，如减少神经网络的层数或参数数量。
- **正则化***：使用L1/L2正则化或Dropout来约束模型。
- **早停法**：在验证集上误差增加时提前停止训练。
- **交叉验证**：使用交叉验证确保模型在不同数据子集上表现一致。

### 2.L1/L2正则化的区别

**L1正则化 (Lasso)**

- **惩罚项**：权重绝对值的和，即 $\lambda\sum|w_i| $。
- **特性**：会使一些权重变为零，导致模型的某些特征被完全忽略。
- **场景**：因此L1正则化具有特征选择的效果，能够产生稀疏模型。

**L2正则化 (Ridge)**

- **惩罚项**：权重平方和的和，即  $\lambda\sum|w_i|^2 $。
- **特性**：不会使权重变为零，而是使所有的权重趋向于较小的值。
- **场景**：这种正则化方法更适合在特征之间有高度相关性时使用，因为它可以均匀地缩小所有权重。

**正则化系数 $\lambda$**

- **$\lambda$ 较小**：正则化项的影响较弱，模型会更倾向于拟合训练数据。
- **$\lambda$ 较大**：正则化项的影响较强，模型的参数会被更大程度地压缩，可能导致欠拟合。

### 3.如何解决梯度问题*

梯度是损失函数对网络权重的偏导数向量，表示当权重发生微小调整时，损失函数的变化量。如果梯度过小，参数更新速度会非常缓慢，可能导致模型无法有效拟合数据，即梯度消失；相反，如果梯度过大，参数更新过快，可能导致模型训练过程中的震荡，即梯度爆炸。

- **选择合适的激活函数***：使用ReLU或其变体作为激活函数，ReLU在正区域的梯度恒定为1，避免了梯度消失。
- **选择合适的权重初始化***：Xavier初始化适用于sigmoid和tanh激活函数，通过根据输入和输出节点数来初始化权重，使得权重的方差保持适中，避免梯度爆炸和消失。He初始化适用于ReLU激活函数，通过放大权重的初始化范围，有助于保持梯度的稳定。
- **批归一化***：将每一层的输入数据转换为均值为0、方差为1的标准正态分布，使得每层输入的分布更加稳定。
- **梯度裁剪**：当梯度过大时，将其裁剪到一个预设的阈值范围内，从而防止梯度爆炸。
- **残差网络**：通过引入跳跃连接，残差网络可以缓解梯度消失问题，使得梯度可以更顺利地反向传播，适合训练非常深的网络。



## 卷积神经网络

### 1.ResNet主要解决了什么问题

ResNet主要解决了随着层数增加，网络出现退化的问题。具体来说，通过引入跳跃连接，它将输入直接传递到后面的层，并与该层的输出相加。这样，在反向传播时，这条路径上的梯度始终部分继承于上一层的梯度，从而避免了随着网络深度增加而导致的梯度消失。



## Transformer

### 1.解释位置编码

由于自注意力机制本身不处理输入数据的顺序，因此必须通过某种方式来引入序列的顺序信息。位置编码对每个词的位置生成一个独特的编码，使模型能够识别其在文本中的位置。在实现过程中，首先将输入文本分割为词并转换为向量，以捕捉语义和句法信息。接着，通过正弦和余弦函数生成与词向量维度相同的位置编码。最后，将位置编码与对应的词向量逐元素相加，使新的词向量同时包含词的语义和位置信息。随后，这一向量被输入到多头自注意力机制中。

### 2.Encoder中的Multi-head

多个注意力头能够并行地捕捉输入序列中不同部分的多种语义关系和特征。每个注意力头独立关注不同的子空间信息，如语言模型中的主谓结构、修饰关系等，从而使模型能够更全面地理解上下文。这些不同头计算出的向量会拼接在一起，并经过线性变换恢复到原始维度。这一过程确保了即使某个头的梯度出现问题，对最终输出的影响也会被其他头的贡献部分抵消，从而使得梯度更加稳定地传递，提升了模型的训练稳定性。同时，多头的并行处理还提高了计算效率。

### 3.为什么使用LN而不是BN

Transformer主要处理的是序列数据。在这种情况下，不同样本之间的某个特定维度通常没有直接的逻辑联系。每个样本都是独立的，而上下文和语义关系主要存在于同一序列的内部，而不是不同样本之间。因此，选择使用LN而不是BN。

- **LN**： 对每一个样本的所有维度进行归一化处理。计算的是每个样本内部的均值和方差，不依赖于批次的大小或分布。
- **BN**： 对一个批次的所有样本的同一个维度进行归一化处理。计算的是一个批次中每个特征的均值和方差，对该特征进行归一化。

### 4.Decoder为什么需要Mask

解码器使用Mask的核心原因是确保其自回归特性，即当前时间步的输出只能基于之前所有时间步的输出生成。为实现这一点，通常会在序列开头加入一个特殊标记并将整个序列右移，同时使用Mask屏蔽掉未来的词。这样，模型在每个时间步只能看到已经生成的部分，而无法访问未来的词。
