# 2021.02.01 课题组组会

## “Distilling the Knowledge in a Neural Network”

> md

知识蒸馏的开山之作。

- （1）到底什么是原来模型中所学到的知识？
- 神经网络中所蕴含的知识表达为：输入向量和输出向量之间的**映射关系**。
- （2）如何让小模型拥有和大模型一样的归纳（generalize）方式呢？
- 迁移集
- （3）怎么让模型学到这些有用的知识？
  - (i) logits
  - (ii) 蒸馏 Distillation
- 实验：集成大模型 > 蒸馏出来的模型 > Baseline
- Soft Target 知识蒸馏

## NAS

> cxd

- 神经网络架构搜索 NAS
- Auto-ML 子领域
  - 自动的设计神经网络的架构
- 通过 Performance **调整搜索策略**（在搜索空间内搜索）。

两篇都是基于 RL，Google 公司。

NAS-RL：the First Paper 开山之作

$\tau$

NAS-RL CNN ：深度需要人工定义吗？

- 给定一个特定深度的 CNN，就能够自动搜索到合适卷积核和是否有跳层连接吗？
- 不可复现吗？

---

非常耗时间，结果无法迁移。解决方案：Cell-based

NASNet

搜索出来的 cell 是通用的，只需要堆叠就行了。

Operation Set

