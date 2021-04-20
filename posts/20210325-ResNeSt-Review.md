![](./20210325/1.png)

- arXiv: <https://arxiv.org/abs/2004.08955>
- 时间：2020 年 04 月



SENet（17.09）、SKNet（19.03） 和 ResNeSt（20.04） 可以放在一起进行阅读，这三篇都是注意力机制的经典文献。其体现了一种层层递进、升级强化的过程。

最后一篇 ResNeSt 可以看做是 SENet 和 SKNet 的集大成之作。

ResNeSt 把 <font color='blue'>Group Convolution</font> 玩得算是炉火纯青，我认为这是它的一个最大的特点。它没有提出什么新的东西，只是把现有的技术（Attention Mechanism ）进行了完美的结合。



## Abstract

> ResNeSt integrates the channel wise attention with multi-path network representation.

本篇论文提出一个模块化的结构：将通道注意力应用到了多分支结构之上。

ResNe<font color='red'>S</font>t 中的 ”S“，表示的就是 <font color='red'>S</font>plit。ResNeSt 由 Split-Attention Block 堆叠而成。

ResNeSt 的取得的成绩如下，有 3 个数据集取得了 SOTA（State of the Art）。

![](./20210325/2.png)

ResNeSt 对比的是 EfficientNet。



## Relate Work

**(1) CNN Architectures:**

- *AlexNet*: shifted from engineering handcrafted features to engineering network architectures.
- *Network in Network*: first uses a **global average pooling layer** to replace the heavy fully connected layers, and adopts **1×1 convolutional layers** to learn non-linear combination of the featuremap channels, which is the **first kind of featuremap attention mechanism**. （第一个注意力机制）
- *VGG-Net*:  stacking the same type of network blocks repeatedly
- *Highway Network*: highway connection makes the information flow across several layers
- *ResNet*: one of the most successful CNN architectures 



**(2) Multi-path and featuremap Attention:**

- *GoogLeNet*: Multi-path representation
- *ResNeXt*: group convolution, converts the multi-path structure into a unified operation.
- *SE-Net*: channel-attention mechanism
- *SK-Net*: featuremap attention across two network branches.



总结 ResNeSt：<font color='blue'>integrates the channelwise attention with multi-path network representation.</font>



## Split-Attention  Network

ResNeSt（Split-Attention Network）的核心是 Split-Attention Block。

![](./20210325/3.png)

### 解析 Split-Attention Block

Split-Attention Block 和核心是 Featuremap Group 和 Split Attention。

**（1）Featuremap Group**

从 Figure 2(Right) 可以看到，ResNeSt Block 的分支结构中又包含着分支，有一种**套娃**的感觉。我把它想象成二叉树的结构，从根节点到叶子节点的路径称为一个分支。ResNeSt Block 中超参数 K（cardinality）、R（radix）控制分支的数量，总分支数为 G=KR。

有多少分支，就表示需要将输入的特征图通道数分成多少组。对于输入的特征图 64×16×16，K=2、R=2 表示将特征图分为 4（K*R）组，每组的特征图大小为 16×16×16。

即：Featuremap Group 总数 为 K×R，每条分支的特征图的通道数等于原始特征图的通道数除以Featuremap Group 总数。可以应用一系列的变换 {F1，F2，......FG} 到每一个单独的组。下图是 <font color=blue>K=2</font>、<font color=red>R=2</font> 情况，我一般称之为 2 条分支（K=2），每个分支内部有 2 个 Featuremap Group（R=2）。

![](./20210325/4.png)



**（2）Split Attention**

Input 表示的每个特征图组经过一系列变换后的输出，即 $Input = F(Featuremap\ Group)$。

![](./20210325/6.png)



Split Attention 是 ResNeSt 的核心，它将**通道注意力机制应用到了每个 Split 分支之上**。



## Split Attention 架构

![](./20210325/8.png)

<font color='red'>NOTES</font>：和 Figure 2(Right) 有所不同，但是其实是等价的，Figure 2 更符合人的直觉，对人来说友好。Figure 4 能够方便的使用计算机实现，对计算机友好。

这里我花了特别长的时间来了解这两种结构为什么是等价的。总结下来，**直接阅读源代码**是最好的方式。



### Split Attention Block 的实现

<https://github.com/zhanghang1989/ResNeSt>

假设输入的图片大小是 (3×64×64)。SplAtConv2d 表示的就是 Split Attention Block，输入的特征图大小为 (64×16×16)。通过 `summary` 可以知道，Split Attention 不改变特征图的 shape。 

![](./20210325/7.png)



以 K=2、R=2 为例，对于 (64×16×16) 的特征图，Split Attention Block 做了如下的事情：



（1）

源码实现中，不使用 1×1 的卷积。只有一个 3×3 的分组卷积。

```python
# In: 64×16×16, Out: 128×16×16
# kernel_size=3×3, same_conv, groups=2*2=4 (cadinality*radix)
self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
```



（2）

```python
# x=(128×16×16)
splited = torch.split(x, int(rchannel//self.radix), dim=1)

# if radix=2
# ==> splited[0] = (64×16×16)
# ==> splited[1] = (64×16×16)
```



（3）

```python
# In: splited[0], splited[1],   Out: 64×16×16
gap = sum(splited)
```



（4）全局平均池化

```python
# Global Pooling
# In: 64×16×16, Out: 64×1×1
gap = F.adaptive_avg_pool2d(gap, 1)
```



（5）

```python
# channels=64, inter_channels=32, groups=2(cardinality=2)
# In: 64×1×1, Out: 32×1×1
self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
```

NOTES：

```python
# in_channels=64, radix=2, reduction_factor=4 ==> inter_channels=32
inter_channels = max(in_channels*radix//reduction_factor, 32)
```



（6）

```python
# inter_channels=32, channels*radix=64*2=128
# In: 32×1×1, Out: 128×1×1
self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
```



（7）

```python
# rSoftmax
# 得到 128 维的注意力向量
atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
```



（8）

```python
# rchannel=128
attens = torch.split(atten, int(rchannel//self.radix), dim=1)
# attens[0] = 64
# attens[1] = 64
```



（9）

```python
# (64×16×16) 的特征图乘以其注意力权重 (64×1×1)
out = sum([att*split for (att, split) in zip(attens, splited)])
```

out 即为 Split Attention Block 的输出，其 shape = (64×16×16)。



## 实验结果

（1）SOTA Datasets

![](./20210325/9.png)

（2）ImageNet & COCO

![](./20210325/10.png)

