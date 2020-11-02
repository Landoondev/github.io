<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

# Hello World

这个博客的建立时间是 2020 年 11 月 02 日。

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



我的个人公众号「蓝本本」。

![](./20201102/2.png)



里面不谈技术，主要是我的一些碎碎念，以及读研的一些体会。陌生人可以在那里私信联系到我。



## 测试插入公式

已知共有 $M$ 样本，各类别 $w_{i}, i = 1, 2, ..., M$ 的先验概率 $P(w_{i})$ 以及类条件概率密度函数 $P(X|w_{i})$ ,对于给定的待分类样本，贝叶斯公式可以计算出该样本分属各类别的概率。即将后验概率作为识别对象归属的依据。

$P(w_{i}|X)=\frac{P(X|w_{i})P(w_{i})}{\sum^{M}_{j=1}P(X|w_{i})P(w_{i})}$

类别的状态是一个随机变量，而某种状态出现的概率是可以估计的。贝叶斯公式体现了先验概率、类条件概率密度函数、后验概率三者的关系。

类条件密度可以采用多维变量的正态密度函数来模拟，此时正态分布的贝叶斯分类器判别函数为：

$h_{i}(X) = P(X|w_{i})P(w_{i}) = \frac{1}{(2\pi)^{n/2}|S_{i}|^{1/2}}e^{[-\frac{1}{2}(X-\bar{X^{w_{i}}})S^{-1}_{i} (X-\bar{X^{w_{i}}})]}P(w_{i})$

使用对数函数进行简化，得：

$H_{i}(X) = -\frac{1}{2}(X - \bar{X^{w_i}})^TS^{-1}_{i}(X - \bar{X^{w_i}})-\frac{n}{2}ln2\pi-\frac{1}{2}|S_{i}|+lnP(w_i)$



- 基于多类问题最小风险贝叶斯决策规则判别函数形式 

已知先验概率 $P(w_{i})$ 、类条件概率密度 $P(X|w_{i}), i = 1, 2, ..., M$ 。对于待分类样本 $X$：

（1） 先根据贝叶斯公式计算后验概率

$P(w_{i}|X)=\frac{P(X|w_{i})P(w_{i})}{\sum^{M}_{j=1}P(X|w_{i})P(w_{i})}, \qquad j = 1, 2, ..., M$

（2）利用后验概率和损失函数 $\lambda(\alpha_{i}, j)$ ，按照下式计算出采取决策 $\alpha_{i}, i = 1, 2, ..., M$ 的条件风险

$R(\alpha_{i}|X) = \sum^{M}_{j = 1}\lambda(\alpha_{i}, j)P(w_{j}|W), \qquad i =1, 2, ..., M$

（3）对（2）中计算得到 M 和风险进行比较，选出使得风险最小的决策 $\alpha_{k}$ 。$\alpha_{k}$ 就是贝叶斯最小风险决策，$w_{k}$ 就是待分类样本的类别。

$R(a_{k}|X) = \min_{i = 1, 2, ..., M}  R(\alpha_{i}|X)$



## 测试代码

```c++
// Quick Sort
#include <iostream>

using namespace std;

int partition(int A[], int l, int r) {
  int v = A[l];  // 为了方便分析取第一个数作为基准值
                  // 实践中可以随机，避免快排退化： 
                  // int r = l + rand() % (r - l + 1);
                  // swap(A[r], A[l]);
                  // int v = A[l];
  int i = l + 1;
  int j = r;
  while (true) {
    while (i <= r && A[i] <= v)
      i++;
    while (j > l && A[j] > v)
      j--;
    if (i >= j) // 结束循环
      break;
    swap(A[i++], A[j--]);
  }
  swap(A[l], A[j]);
  return j;
}

void print_res(int A[], int n) {
  for (int i = 0; i < n; ++i) {
    cout << A[i] << " ";
  }
  cout << endl;
}

void quick_sort(int A[], int l, int r) {
  if (l >= r)
    return;
  
  int p = partition(A, l, r);
  quick_sort(A, l, p - 1);
  quick_sort(A, p+1, r);
}

void quick_sort(int A[], int n) {
  quick_sort(A, 0, n - 1);
}

```

