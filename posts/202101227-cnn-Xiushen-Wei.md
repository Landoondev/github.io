# 解析卷积神经网络——深度学习实践手册

《解析卷积神经网络——深度学习实践手册》是南京大学博士魏秀参写的一本小册子，给很多初学者提供了很多帮助。

- 魏秀参主页：[http://www.weixiushen.com/](<http://www.weixiushen.com/>)
- ✅ 手册下载地址：[http://www.weixiushen.com/book/CNN_book.pdf](<http://www.weixiushen.com/book/CNN_book.pdf>)

我是在阅读中科院计算所的王晋东《迁移学习简明手册》时，发现里面提到，并给出了不错的评价。因此，我就抽了一天时间看了一遍。

总体看一遍下来，我认为应该可以算是一份关于卷积神经网络非常好的综述性手册，涵盖了深度学习中卷积神经网络的方方面面。对我而言，既是复习，又获得了新的知识。

手册 ≈ 168 页，参考文献 101 篇（参考文献对现在的我而言，是一份优秀的论文阅读清单）。

接下来阅读第二遍，把第一遍标注的笔记进行整理成 Markdown 的形式。

笔记的形式大多是摘录的一句/段话，因此不会连贯。

偶尔也会有自己的注解。

# 笔记

## 前言

- Must Know Tips/Tricks in Deep Neural Networks by [Xiu-Shen Wei]: [http://www.weixiushen.com/project/CNNTricks/CNNTricks.html](<http://www.weixiushen.com/project/CNNTricks/CNNTricks.html>)

## 绪论

- 相比传统机器学习算法仅学得模型单一“任务模块”而言，深度学习除了模型学习，还有特征学习、特征抽象等 任务模块的参与，借助**多层**任务模块完成最终学习任务，故称其为**“深度”学习**。

- 神经网络的第二次高潮，即二十世纪八十至九十年代的连接主义（Connectionism）。但受限于当时数据获取的瓶颈，神经网络只能在中小规模数据上训练，因此**过拟合（Overfitting）**极大困扰着神经网络型算法。同时，神经网络 算法的**不可解释性**令它俨然成为一个“黑盒”，训练模型好比撞运气般，有人无奈的讽刺说它根本不是“科学”（Science）而是一种“艺术”（Art）。

- 2006 年，Hinton 等在 Science 上发表文章 Reducing the dimensionality of data with nerual network 提出：一种称为**“深度置信网络”**的神经网络模型可通过逐层预训练（greedy layer-wise pretraining）的方式有效完成模型训练过程。很快，更多的实验结果证实了这一发现，更重要的是除了证明神经网络训练的可行性外，实验结果还表明神经网络模型的**预测能力**相比其他传统机器学习算法可谓“鹤立鸡群”。

## 卷积神经网络基础知识

- 1980 年，日本科学家福岛邦彦（Kunihiko Fukushima）提出的一种层级化的多层人工神经网络 Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position，神经认知模型（Neocongnitron）在后来也被认为是现今卷积神经网络的前身。

  - 在福岛邦彦的神经认知模型中，两种最重要的组成单元是“S 型细胞”和“C 型细胞”，两类细胞交替堆叠在一起构成了神经认知网络。其中，S 型细胞用于抽取局部特征（local feature），C 型细胞则用于抽象和容错。

  - S-cells 和 C-cells 与现今卷积神经网络中的**卷积层（convolution layer）**和**池化层（pooling layer）**一一对应。

  - > CNN 的起源知识。

- 深度学习普及之后，人工特征已逐渐被表示学习根据任务需求**自动“学到”**的特征表示所取代。

- > 作者在 2014 年在北京与深度学习领域巨头 Yoshua Bengio 共进晚餐，Yoshua Bengio 回答：今后深度学习将取代与人工特征方面的研究。
  >
  > - 2014 年，那时我正在读高一。


## 卷积神经网络基本部件

- 卷积网络中的卷积核参数是通过网络训练**学习**出的。

  - 可以学到类似的横向、纵向边缘滤波器，还可以学到任意角度的边缘滤波器。

  - 检测颜色、形状、纹理等等众多基本模式（pattern）的滤波器（卷积核）都可以包含在一个足够复杂的深层卷积神经网络中。

  - 通过“组合” 这些滤波器（卷积核）以及随着网络后续操作的进行，基本而一般的模式会逐渐被抽象为具有**高层语义**的“概念”表示，并以此对应到具体的样本类别。**颇有“盲人摸象” 后，将各自结果集大成之意。**

  - > “盲人摸象”，“集之大成”，这个描述非常妙！

- 池化层实际上是一种降采样（down-sampling）操作。三种功效：

  - 特征不变性（feature invariant）
  - 特征降维
  - 一定程度上防止过拟合

- ✅ 德国 University of Freiburg 的研究者提出用一种特殊的卷积操作（stride convolutional layer）来代替池化层实现降采样，进而构建一个个含卷积操作的网络，其实验结果显示这种改造的网络可以达到、甚至超过传统卷积神经网络（卷积层汇合层交替）的分类精度。（Striving for Simplicity: The All Convolutional Net, arXiv: 1412.6806）

- 激活函数（Activation Function）的引入为的是增加整个网络的表达能力（即非线性）。否则， 若干线性操作层的堆叠仍然只能起到线性映射的作用，无法形成复杂的函数。

  - > 我看过有一篇论文“Activation Functions: Comparison of Trends in
    > Practice and Research for Deep Learning, arXiv: 1811.03378”，里面列出了 21 种激活函数。

  - 为了避免梯度饱和效应的发生，**Nair 和 Hinton 于 2010 年将修正线性单元（Rectified Linear Unit, ReLU）引入神经网络。** 论文：Rectified Linear Units Improve Restricted Boltzmann Machines

- 全连接层（fully connected layers）在整个卷积神经网络中起到“分类器”的作用。如果说卷积层、池化层和激活函数层等操作是**将原始数据映射到隐层特征空间**的话，全连接层则起到**将学到的特征表示映射到样本的标记空间**的作用。

## 卷积神经网络经典结构

- 小卷积核（如 3 × 3）通过多层叠加可取得与大卷积核（如 7 × 7）**同等规模的感受野**，此外采用小卷积核同时可带来其余两个优势：第一，由于小卷积核需多层叠加，加深了网络深度进而增强了网络容量（model capacity）和复杂度（model complexity）；第二，增强网络容量的同时减少了参数个数。

  - 目前已有不少研究工作，通过改造现有卷积操作试图**扩大原有卷积核在前层的感受野大小**，或使原始感受野不再是矩形区域而是更自由可变的形状，以提升模型预测能力。

  - > 扩张卷积：Multi-Scale Context Aggregation by Dilated Convolutions, arXiv: 1511.07122
    >
    > 可变卷积：Deformable Convolutional Networks, arXiv: 1703.06211

- 是同一卷积核在不同原图中响应的区域可谓大相径庭；

  - 对于某个模式，如鸟的躯干，会有不同卷积核（其实就是神经元）产生响应；
  - 同时对于某个卷积核（神经元），会在不同模式上产生响应；
  - 神经网络响应的区域多呈现“稀疏“特性，即**响应区域集中**且占原图比例较小。

- 深度特征的**层次性**已成为深度学习领域的一个共识。

- AlexNet 在整个卷积神经网络甚至连接主义机器学习发展进程中占据里程碑式的地位，一些训练的引入使得“不可为”变“可为”，甚至是“大有可为”。

  - ReLU
  - LRN
  - data augmentation
  - dropout
  - 此后的卷积神经网络大体都是遵循这一网络构建的基本思路。

- NIN 采用了复杂度更高的**多层感知机**作为层间映射形式，一方面提供了网络层间映射的一种新可能；另一方面增加了网络卷积层的非线性能力，使得上层特征可有更多复杂性与可能性的映射到下层，这样的想法也被后期出现的残差网络和 Inception 等网络模型所借鉴。

  - > Network In Network, arXiv: 1312.4400
    >
    > Deep Residual Learning for Image Recognition, arXiv: 1512.03385
    >
    > Going Deeper with convolution, arXiv: 14094842

  - NIN 网络模型的另一个重大突破是摒弃了全连接层作为分类层的传统，转而改用**全局平均池化**操作（global average pooling）。

- 如果一个浅层神经网络可以被训练优化求解到某一个很好的解，那么它对应的深层网络至少也可以，而不是更差。这一现象（*degradation problem*）在一段时间内困扰着更深层卷积神经网络的设计、训练和应用。

  - ResNet、Highway Network

## 卷积神经网络的压缩

- 许多研究表明，深度神经网络面临着严峻的过参数化（over-parameterization）——模型内部参数存在着巨大的冗余。

  - 参数冗余是有意义的。深度神经网络面临的是一个极其复杂的非凸优化问题，对于现有的基于梯度下降的优化算法而言，这种参数上的**冗余保证了网络能够收敛到一个比较好的最优值** 。

  - > 缓解神经网络遇到了局部最优问题。

  - 需要一定冗余度的参数数量来保证模型的可塑性与“容量”（capacity）。

  - Predicting Parameters in Deep Learning, arXiv:1306.0543 发现，只给定很小一部分的参数子集（约全部参数量的 5%），便能完整地重构出剩余的参数，这揭示了**模型压缩**的可行性。

  - 奇异值分解（Singular Value Decomposition）来重构全连接层的权重。

- 数据驱动的剪枝：根据输出中每一个通道（channel）的稀疏度来判断相应滤波器的重要程度。

  - > *Network Trimming: A Data*-*Driven Neuron Pruning Approach towards Efficient Deep Architectures*. arXiv: 1607.03250

- 知识蒸馏：其实是迁移学习（Transfer Learning）的一种，其最终目的是将一个庞大而复杂的模型所学到的知识，通过一定的技术手段迁移到精简的小模型上，使得小模型（学生）能够获得与大模型（老师）相近的性能。

  - SqueezeNet 在 ImageNet 上能够达到 AlexNet 的分类精度，而其模型大小仅仅为 4.8MB。

  - > *SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size*
    >
    > SqueezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters. Additionally, with model compression techniques we are able to compress SqueezeNet to less than 0.5MB (510x smaller than AlexNet).

## 数据扩增

- 注明的 AlexNet 提出的同时，Krizhevsky 等人还提出了一种名为”Fancy PCA“的数据扩增方法。

## 网络参数初始化

- Xavier 初始化方法仍有不甚完美之处，即该方法并未考虑非线性映射函数对输入 $s$ 的影响。因为使用如 ReLU 函数等非线性映射函数后，输出数据的期望往往不再为 0。

  - He 等人对此提出改进，将非线性映射造成的影响考虑进参数初始化中。

## 激活参数

- kaggle 上举办的 2015 年”国家数据科学大赛“（national data science bowl）——浮游动物的图像分类，随机化 ReLU 首次提出并一举夺冠。

## 网络正则化

- 正则化是机器学习中通过**显式的控制模型复杂度**来避免模型过拟合、确保泛化能力的一种有效方式。

  - $ \ell_1$ 正则化起到使参数更稀疏的作用。稀疏化的结果使优化后的参数一部分为 0 。另一部分非零值实值的参数可起到**选择重要参数或特征维度**的作用。

  - $\ell_1, \ell_2$ 也可以联合使用，这被称为”Elastic 网络正则化“。Regularization and variable selection via the
    elastic net.

  - > Elastic 初始化的论文在 arXiv 上没有。

## 超参数和网络训练

- Google 提出了批规范化操作（batch normalization, BN）。

  - *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*, arXiv:1502.03167
  - Google 的研究人员发现可通过 BN 来规范化某些层或所有层的输入，从而可以固定每层输入信号的均值与方差。这样一来，即使网络模型较深层的响应或梯度很小，也可**通过 BN 的规范化作用将其的尺度变大**，以此便可解决深层网络训练很可能带来的“梯度弥散” 问题。
  - 在卷积神经网络中，BN 一般应作用在非线性映射函数前。

- ✅ 基于动量的随机梯度下降：通过积累前几轮的“动量”信息辅助参数更新。基于动量的随机梯度下降法除了可以抑制振荡，还可在网络训练中后期趋于收 敛、网络参数在局部最小值附近来回震荡时帮助其**跳出局部限制**，找到更优的网络参数。

  - > 关于模型优化算法，可以看一篇综述。
    >
    > *An overview of gradient descent optimization algorithms*
    >
    > [https://arxiv.org/pdf/1609.04747.pdf](<https://arxiv.org/pdf/1609.04747.pdf>)

## 不平衡样本的处理

- 不平衡（imbalance）的训练样本会导致训练模型侧重样本数目较多的类别，而“轻视”样本数目较少类别，这样模型在测试数据上的泛化能力就会受到影响。
  - 数据层面处理方法多借助数据采样法()使整体训练集样本趋于平衡。
  - 算法层面的处理方法：注意力机制，通过优化目标函数就可以调整模型在小样本上的“注意力”。算法层面处理不平衡样本问题的方法也多从代价敏感（cost-sensitive）角度出发。

## 模型集成方法

- 如 ImageNet、KDD Cup、Kaggle 等竞赛的冠军做法，或简单或复杂其**最后一步必然是集成学习**。

  - >  ”众人拾柴火焰高“，”锦上添花“

- 多层特征融合操作时可直接将不同层网络特征级联（concatenate）。而对于特征融合应选取哪些网络层，一个实践经验是：最好使用**靠近目标函数的几层卷积特征**，因为愈深层特征包含的高层语义性愈强、分辨能力也愈强;相 反，网络较浅层的特征较普适，用于特征融合很可能起不到作用有时甚至会起 到相反作用。

- ✅ 网络“快照”集成法（snapshot ensemble）利用了网络解空间中的这些局部最优解来对单个网络做模型集成。通过循环调整网络学习率（cyclic learning rate schedul）可使网络依次收敛到不同的局部最优解。

  - 利用余弦函数 $cos(\cdot)$ 的循环特性来循环更新网络学习率，将学习率从 0.1 随 $t$ 的增长逐渐减缓到 0，之后将学习率重新放大从而跳出该局部最优解。

- > 关于模型集成也是可以看一篇综述。

## 深度学习开源工具简介

- > Torch 和 PyTorch 是同一个东西吗？我一直在 `import torch` ，所使用的是什么？

  - PyTorch 为 Torch 提供了更便利的接口。
