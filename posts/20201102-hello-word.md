
# 关于我的 Blog

这个博客的建立时间是 2020 年 11 月 02 日。

## 博客主题

[https://github.com/pages-themes/cayman](https://github.com/pages-themes/cayman)

我进行了很多自定义配置，包括支持公式、访客统计、阴影框等等。我看到一些我喜欢的元素，就会添加到我的这里。

如果你也想使用和我一样的 Blog 主题，可以直接阅读我的配置文件源码，很简单的。

## 博客的内容

此时我正在读研一，这个博客将会伴随我之后的三年研究生生活。

博客的主要内容是一些关于技术笔记。可以直接使用 Markdown，这非常的方便。我日常都是使用 MD 来编写我的笔记。

在这个网站上传一篇自己的笔记的步骤很简单。

1. 将 MD 文件拷贝进入 `posts/` 文件夹
   - 这里我会选择压缩一下图片大小
2. 在` index.md` 文件中添加 MD 文件的索引连接
   - 如：本篇 Hello World 的文件名是 `20201102-hello-word.md`，已经拷贝到 `posts/`文件夹下；
   - 在 `index.md` 文件中添加: `[2020.11.02: Hello Word](posts/20201102-hello-word)`
   - 就会得到如下图的效果。![](./20201102/1.png)

3. 上传

```shell
git add .
git commit -m "Add new post: Hello World."
git push origin main
```

![](./20201102/2.png)

## 测试插入公式

能显示公式，是我的核心需求之一。

已知共有 $M$ 样本，各类别 $w_{i}, i = 1, 2, ..., M$ 的先验概率 $$P(w_{i})$$ 以及类条件概率密度函数

$$P(X|w_{i})$$

对于给定的待分类样本，贝叶斯公式可以计算出该样本分属各类别的概率。即将后验概率作为识别对象归属的依据。

$$P(w_{i}|X)=\frac{P(X|w_{i})P(w_{i})}{\sum^{M}_{j=1}P(X|w_{i})P(w_{i})}$$

类别的状态是一个随机变量，而某种状态出现的概率是可以估计的。贝叶斯公式体现了先验概率、类条件概率密度函数、后验概率三者的关系。

类条件密度可以采用多维变量的正态密度函数来模拟，此时正态分布的贝叶斯分类器判别函数为：

$$h_{i}(X) = P(X|w_{i})P(w_{i}) = \frac{1}{(2\pi)^{n/2}|S_{i}|^{1/2}}e^{[-\frac{1}{2}(X-\bar{X^{w_{i}}})S^{-1}_{i} (X-\bar{X^{w_{i}}})]}P(w_{i})$$

使用对数函数进行简化，得：

$$H_{i}(X) = -\frac{1}{2}(X - \bar{X^{w_i}})^TS^{-1}_{i}(X - \bar{X^{w_i}})-\frac{n}{2}ln2\pi-\frac{1}{2}|S_{i}|+lnP(w_i)$$

- 基于多类问题最小风险贝叶斯决策规则判别函数形式 

已知先验概率 $$P(w_{i})$$ 、类条件概率密度

$$P(X|w_{i}), i = 1, 2, ..., M​$$ 

对于待分类样本 $X$：

（1） 先根据贝叶斯公式计算后验概率

$$P(w_{i}|X)=\frac{P(X|w_{i})P(w_{i})}{\sum^{M}_{j=1}P(X|w_{i})P(w_{i})}, \qquad j = 1, 2, ..., M$$

（2）利用后验概率和损失函数 $$\lambda(\alpha_{i}, j)$$ ，按照下式计算出采取决策 $$\alpha_{i}, i = 1, 2, ..., M$$ 的条件风险

$$R(\alpha_{i}|X) = \sum^{M}_{j = 1}\lambda(\alpha_{i}, j)P(w_{j}|W), \qquad i =1, 2, ..., M$$

（3）对（2）中计算得到 M 和风险进行比较，选出使得风险最小的决策 $\alpha_{k}$ 。$\alpha_{k}$ 就是贝叶斯最小风险决策，$w_{k}$ 就是待分类样本的类别。

$$R(a_{k}|X) = \min_{i = 1, 2, ..., M}  R(\alpha_{i}|X)$$

**具体做法是：**

### 方法一

1、在文章头插入如下代码：

```html
<head>
  <script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});</script>
  <script type="text/javascript" async src="//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"></script>
</head>
```

2、将下面一行代码插入到配置文件`_config.yml`中

```shell
markdown: kramdown
```

### 方法二

新增 `_layouts/default.html` 文件，将显示公式需要的 `<head></head>` 添加即可。每次加载一篇博客时，会自动加载响应的脚本，使得 LaTeX 公式正常显示。

我现在使用的就是方法二。

本地的 Markdown 文件是怎么的，在博客中显示就是什么样的。 

⚠️注意：不要使用键盘上 return 上面的 $\mid$ 字符！！！。

只能以公式的形式使用，LeTeX 如下：

```latex
\mid
```

## 测试代码

能显示代码，是我的核心需求之一。

```c++
// hello.c
#include <stdio.h>

int main() {
  printf("hello, world\n");
  return 0;
}
```

## 添加阅读数和访客量

使用的是不蒜子。两行代码，搞定计数。

[https://busuanzi.ibruce.info/](https://busuanzi.ibruce.info/)

## 给 main 标签新增的阴影框

参考的是 [https://cjting.me/](https://cjting.me/) 博客主题，我特别喜欢。只需要配置 `style="box-shadow: 0 8px 10px #959da5;"`。

2020.12.19

## 又更新首页样式了

删除了 cayman 的 header 标签样式，显示首页一片白，非常整洁。