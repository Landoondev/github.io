# 视网膜血管分割 Retinal vessel segmentation

本周要汇报的论文：LadderNet



## 1. 数据集（Datasets）

视网膜血管分割的公开集最常用的有 DRIVE、STARE 和 CHASE_DB。



### 1.1 DRIVE

DRIVE: <font color='red'>D</font>igital <font color='red'>R</font>etinal <font color=red>I</font>mages for <font color='red'>V</font>essel <font color='red'>E</font>xtraction (<https://drive.grand-challenge.org/>)

DRIVE 数据库的图像来自荷兰的一个糖尿病视网膜病变筛查项目，筛查人群包括 400 名年龄在 25-90 岁的糖尿病患者。随机抽取了 40 张照片，其中 33 张没有显示任何糖尿病视网膜病变的迹象，7 张显示有轻度早期糖尿病视网膜病变的迹象（25_training、26_training、32_training、03_test、08_test、14_test、17_test）。

![](./20210516/2.png)

DRIVE 数据集基本信息如下：

- 每张图片的的分辨率为 584 × 565 pixels，3 通道的彩色图片。
- 训练集 20 张图像，测试集 20 张图像。对于测试案例，有两个人工分割：一个被用作  gold standard，另一个可用于比较计算机生成的分割与独立的人类观察者的分割。
- mask 表示 region of interest (RoI)。



DRIVE 数据库的建立是为了能够对视网膜图像中的血管进行分割的比较研究，被用于诊断、筛查、治疗和评估各种心血管和眼科疾病。

此外，每个人的视网膜血管树都是独一无二的，可用于生物识别。



### 1.2 STARE

STARE: <font color='red'>ST</font>ructured <font color='red'>A</font>nalysis of the <font color='red'>Re</font>tina (<https://cecas.clemson.edu/~ahoover/stare/probing/index.html>)

![](./20210516/3.png)

STARE 数据基本信息：

- 每张图片的分辨率为 700×605 pixels。
- 共 20 张图片。



### 1.3 CHASE_DB1

<https://blogs.kingston.ac.uk/retinal/chasedb1/>

![](./20210516/4.png)

- 每张图片的分辨率为 700×605 pixels。

- 14 个学生的左眼和右眼图像，共 28 张。



# LadderNet

![](./20210516/1.png)

LadderNet：一种基于 U-Net 的多路径医学图像分割网络。

作者：Juntang Zhuang（<https://juntang-zhuang.github.io/>），本科清华，目前在 Yale University Ph.D. in Biomedical Engineering。

- 论文时间：ArXiv 2018 年 10 月
- 论文地址：<https://arxiv.org/abs/1810.07810>



## Abstract

U-Net、Attention U-Net、R2-UNet 和 U-Net with residual blocks or blocks with dense connections 的信息流的路径数量是十分有限的。本篇论文提出的 LadderNet， 由于有 skip connections、residual blocks，所以有<font color='red'>更多的信息流路径</font>，可以被看作是全卷积网络（FCN）的集合。

在 DRIVE 和 CHASE_DB1 两个视网膜中的血管分割图像数据集上进行测试。

目前在视网膜分割任务上，基于 GAN 的方法（RV-GAN）取得了 SOTA，DRIVE（AUC = 0.989），CHASE_DB1 （AUC = 0.991）。 

LadderNet 在 DRIVE 和 CHASE_DB1 上都可以排在前十，相比于 U-Net，AUC 提升了 0.01。

![](./20210516/5.png)

## 1 Introduction

在各种分割网络的变体中，U-Net 是医学图像分析中使用最广泛的结构，主要是因为带有跳跃连接的 encoder-decoder 结构允许有效的信息流，并且在没有足够大的数据集的情况下表现良好。

各种 U-Net 的变体，仍属于 encoder-decoder 结构，其中信息流的路径数量是有限的，这是本篇论文的背景。

> However, all these U-Net variants still fall into the encoder decoder structure, where <font color='red'>the number of paths for information flow is limited.</font> 



本篇论文提出了 LadderNet，一种用于语义分割的卷积网络，**具有更多的信息流路径**。LadderNet 可以被视为 FCN 的集合（即 FCN 是其一种特殊形式），并实验验证了 LadderNet 在视网膜血管分割任务中的优异表现。在视网膜图像中的血管分割任务上验证了其优越性能。



## 2. Methons

LadderNet  有更多的信息流路径（more paths of information flow）。

![](./20210516/LadderNet.png)

A~E 表示不同的空间尺度的特征图；1、3 为 encoder 分支，2、4 为 decoder 分支。从一个级别到下一个级别，通道的数量增加一倍（例如，A 到 B）。



### LadderNet 和 U-Net 的联系

**（1）LadderNet 可以视为 U-Net 的链**，1 和 2 看做一个 U-Net，3 和 4 看做是另一个 U-Net。LadderNet  包含两个 U-Net，也可以连接更多的 U-Net 来形成复杂的网络结构。

LadderNet 的 skip connection 使用的 sum，而 U-Net 使用的是 Concate。



**（2）LadderNet 也可以被看作是多个 FCN 的集合体**，残差连接提供了多条信息流路径。

LadderNet 信息流路径总数随着 encoder-decoder 对的数量和空间层次的数量呈指数级增长。我简单数了一下 LadderNet 的从输入到输出，共有 75 条路径。

LadderNet 取得较高的精度的原因可以总结为：LadderNet has the potential to capture more complicated features and produce a higher accuracy.



### Shared-weights residual block

更多的 encoder-decoder 分支将增加参数的数量和训练的难度。为了解决这个问题，本篇论文提出共享权重的残差块（Fig 1）。受 RCNN 的启发，同一区块中的两个卷积层可以看做是一个递归层。除了两个批处理规范化层是不同的。

共享权重的残差块结合了 skip connection、recurrent convolution 和 dropout 正则化的力量，参数要比标准的残差块少得多。

参数量情况 LadderNet vs U-Net，前者的参数量减少了 97%（使用 torchsummary 中的 summary）。



```
Total params: 921,902
Trainable params: 921,902
Non-trainable params: 0
Total mult-adds (M): 41.66
```

```
Total params: 31,031,810
Trainable params: 31,031,810
Non-trainable params: 0
Total mult-adds (G): 16.45
```

对于输入的一张 1×48×48 的特征图，最后得到 2×48×48 的输出。通道数作者的是实现是 (1, 10, 20, 40, 80, 126)，有其他的实现使用的 (1, 16, 32, 64, 128, 128)，level E 不加倍。

![](./20210516/9.png)

## Experiment

DRIVE 数据集：40 张彩色图片，20 张训练集，20 张测试集，每张图片 565×584 pixels。为了增加训练的样本，随机采样  190,000 个 48×48 pixels 的 patch，10% 用作验证集。

CHASE_DB1 数据集：28 张彩色图片，20 张做训练集，8 张做测试集，每张图片 700×605 pixels。随机采样 760,000 个 48×48 pixels 的 patch，10% 用作验证集。

在数据预处理阶段，3 通道被转换为单通道。



### 数据预处理

<https://github.com/juntang-zhuang/LadderNet/blob/master/lib/pre_processing.py>

（1）提取彩色眼底图像血管与背景**对比度较高的绿色通道**；并利用双边滤波对其降噪。

- 在进行 RGB2gray 时，给予 g 通道更高的权重。

```python
#convert RGB image in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    # 通道顺序为：B G R.
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

```

（2）限制对比度直方图均衡化(CLAHE) 抑制噪声、提升对比度；全局锐化，抑制伪影、黄斑等噪声。

（3）局部自适应 Gamma 矫正，抑制光照不均匀因素与中心线反射现象。

（4）尺度形态学 Top-Hot 变换。

![](./20210516/10.png)

```python
#My pre processing (use for both training and testing!)
def my_PreProc(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs
```



### 评估指标

Accuracy (AC), Sensitivity (SE), Specificity (SP) and F1-score. 

True Positive (TP), True Negative (TN), False Positive (FP) and False Negative (FN).



### 实验结果

![](./20210516/11.png)

---



在实验时，对于 DRIVE 数据集，训练集 20 张 3×584×565 的图像，进行预处理，数据增强后，得到 100,000 个 1×64×64 patch。在华西的电脑上训练一个 Epoch 耗时：13 分钟。

![](./20210516/12.png)

























