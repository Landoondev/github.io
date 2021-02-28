# 组会 2021.02.28

## Survey 综述

> yxy

医学图像自然语言处理报告生成任务的研究问题的综述。

![](./20210228/34.jpeg)

- 数据集：18 个数据集。
  - 胸片：MIMIC-CXR、PadChest
- 模型设计：视觉部分（CNN）+语言部分（LSTM、BiLSTM）
- 视觉部分：ResNet、DenseNet、VGG
- 语言部分：GRU、LSTM、Attention
- 辅助任务及优化策略：
  - 生成特征图的同时，输出相关标签分类，能够有效对下一步的报告生成有辅助效果提升作用。
- 可解释性：模型可解释性应当分为**局部和全局**，全局则是针对结果，局部是针对样本的说明。
  - 基于概念的 TCAV（概念激活向量）
- 未来的挑战：
  - Protocol for expert evaluation
  - Automatic metrics for medical correctness
  - 可解释性的改善
- 总结：忽略了疾病分类、图像分割任务中对的分析工作

当一个 AI 系统具有完美的报告生成精度时，如何信任它？——可解释性的重要性

## Dual-Teacher: Integration Intra-domain and Inter domain ...

> ylj

- 半监督域适应

- Inter-domain Teacher: UNet
- 一致性损失
- 少量标注的 CT 图像输入 Student 进行训练，在训练过程中**结合** Inter-domain Teacher 和 inter-domain Teacher 学习到的**特征知识**对 CT 图像进行特征学习，得到最终的分割模型。
- 实验的数据集：Multi-modality Whole Heart Segmentation(MM-WHS) 2017

问题：域内的教师网络是怎么得到的？

指数移动平均（EMA）。

## DARTS

> cxd

可复现

- 神经网络自动搜索

- 从离散、不可微分，变成了连续可搜索。
- Related Works: NAS-RL
  - 全局搜索
  - 800 块 GPU
- NASNet
  - CIFAR-10 Architecture：Normal Cell、Reduction Cell 堆叠
  - ImageNet Architecture
- DARTS
  - 跑了 4 天（华西的电脑）GPU Days
  - 松弛操作（Relaxation），可微分
  - 超网剪枝后得到子网，子网的性能依然不错。
- Bilevel Optimization Problem
  - $\alpha$：网络架构参数
  - ![](./20210228/35.jpeg)
  - 固定其中一个参数，优化另一个，交替进行。EM 算法也是如此。
- A Simple Approximation
- Secon-order Approximation
- Results
  - CIFAR-10：
  - PTB：自然语言处理中的一个小数据集
  - ImageNet：
  - WT2：

- 结论：
  - 练习空间上搜索算法，CNN、RNN。
  - SOTA
  - 搜索速度大大提升

GitHub：Awesome-NAS

问题：搜索时深度是固定吧？ 

搜索过程权重会变化？

改变了 NASNet 的搜索方式？

边之间的竞争，边之间的相关性。

## WRN 与 ResNeXt

> ww

- WideResNet
- ResNeXt
  - 思路：分组（增加分支/cardinality）
- ResNeXt 对比 InceptionV4
- CNN 基干网络总结
  - 基本组成：卷积、池化、全连接
  - 技巧：BN、GAP
  - 重要思路：分组、瓶颈、残差、密集连接、通道权重

