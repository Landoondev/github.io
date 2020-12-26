# 国科大高级人工智能（2020 秋季学期）（1）

面向题目复习法

## 1. 2018 年卷

## 1.1 图灵测试

A. M. Turing, Computing Machinery and Intelligence, Mind, 59:433-460, 1950.

- 图灵问题：Can Machine Think?
- 图灵测试（Turing Test）：一个人（C）在完全不接触对方（A 和 B）的情况下，和对方进行一系列的问答，如果在相当长时间内，C 无法根据这些问题判断对方是人（B）还是计算机（A）， 那么，就认为该计算机具有同人相当的智能，(即计算机是能思维的)
- 质疑
  - 图灵测试是不可构造的：“完全不接触“的环境难以构造
  - 图灵测试不是可重现的：答案正确性的判断是主观的
  - 图灵测试无法进行数学分析：缺少形式化描述

## 1.2 人工智能的 3 大分支

- 符号主义学派：
  - 逻辑学派
  - ”人的认知基元是符号，认知过程即符号操作过程“
  - 人工智能对的核心是知识表示、知识推理和知识运用
  - 西蒙（1975 年获图灵奖，1978 年获诺贝尔经济学奖）、妞厄尔
  - 衍生出：**逻辑、专家系统、知识库**
- 联结主义学派
  - 仿生学派或生理学派
  - 人的思维基元是神经元
  - 神经网络及神经网络间的连接机制和学习算法
  - 麦卡洛可（McCulloch）、皮茨（Pitts）
  - 衍生出：**人工神经网络、认知科学、脑类计算**
- 行为主义学派
  - 进化主义或控制论学派
  - 智能取决于感知和行动
  - 主张利用机器对环境作用后的响应或反馈为原型来实现智能化
  - 人工智能可以像人类智能一样通过进化、学习来逐渐提高和增强
  - 衍生出：**控制论、多智能体、强化学习等**

## 1.3 搜索算法

> Search—第二讲 2020-09-22
>
> 自动化所的吴老师

**1、无信息搜索（Uninformed Search）**

- 深度优先搜索（Depth-First Search）：LIFO stack
- 广度优先搜索（Breadth-First Search）：FIFO queue
- 迭代深入搜索（Iterative Deepening）：结合 DFS 的空间优势与 BFS 的时间优势
- 代价敏感搜索（Cost-Sensitive Search）：find the least-cost path.
  - 代价一致搜索（Uniform Cost Search）：priority queue

![](./20201226/1.jpeg)

**2、启发式搜索（Informed Search）**

启发策略：

- 估计一个状态到目标距离的函数
- 问题给予算法的额外信息，为特定搜索问题而设计

**（1）贪婪搜索（Greedy Search）**

- 启发函数：$f(n) = h(n)$
- 最坏情况：类似 DFS

**（2）A* 搜索**

- UCS 和 Greedy 的结合
- $f(n)=g(n)+h(n)$
- 启发函数 h 是可采纳的：$0 \leqslant h(n) \leqslant h^*(n)$
  - $h^*(n)$ ：到最近目标的真实耗散
- UCS vs A*
  - 代价一致搜索在所有“方向”上等可能的扩展
  - A* 搜索主要朝着目标扩展，而且能够保证最优性

对于解决难的搜索问题，大部分工作就是想出可采纳的启发函数。

**（3）图搜索**

- A* 图搜索

**A* 算法的总结：**

- A* 使用后向耗散和前向耗散（估计）
- A* 是完备的、最优的，也是效率最优的（可采纳的/一致的启发函数）
- 启发式函数设计是关键：常使用松弛问题
- A* 往往在计算完之前就耗尽了空间

**3、局部搜索**

- 爬山法搜索
- 模拟退火搜索：避免局部极大（允许向山下移动）
- 遗传算法：基于适应度函数、配对杂交、产生可选的变异

### 1.4 A* 图搜索的最优性条件

A* 树搜索的最优性条件是：启发函数是一致的。

**一致性：**

- 沿路径的节点估计耗散 f 值单调递增 $h(A) ≤ cost(A\  to\  C) + h(C)$
- A* 图搜索具备最优性

A* 图搜索的最优性条件是：启发函数是一致的。

### 1.5 Deep Belief Networks 网络结构

> DNN 神经网络基础 Deep Neural Network 2020 年 10 月 06 日
>
> 吴老师 

Deep Learning 的常用模型：

- Deep Belief Networks(DBN)
- Deep Boltzmann Machine(DBM)

发展历程

- Hopfield network：单层全互连、对称权值的反馈网络。
- Boltzman machine：结构类似于 Hopfield 网络，但它是具有隐单元的反馈互联网络。
- Restricted Boltzman machine ：通过输入数据集学习概率分布的随机生成神经网络。一个可见层、一个隐层、层内无连接。
- Deep Belief Networks(DBN) ：概率生成模型、深层结构-多层、非监督的预学习提供了网络好的初始化、监督微调（fine-tuning）

![](./20201226/2.jpeg)

- Deep Boltzmann Machine(DBM)

### 1.6 卷积神经网络（CNN）的特点

> DL for image
>
> 图像数据的深度学习模型 2020-10-13
>
> 自动化所吴老师

卷积神经网络是一种特殊的深层神经网络模型。

- 它的神经元间的连接是非全连接的
- 同一层中某些神经元之间的连接的权重是共享的（即相同的）。

局部连接

- 局部感受野
- 参数共享：平移不变性

卷积（Convolution）

- 稀疏连接
- 参数共享

填充（Padding）

步长（Stride）

输入与输出的尺寸关系：

- $n \times n \ images, \ f \times f \ fileter, padding \ p, stride \ s $
- $\left\lfloor \frac{n+2p-f}{s} + 1 \right\rfloor \times \left\lfloor \frac{n+2p-f}{s} + 1 \right\rfloor $

池化 Pooling

- 子采样。没有需要学习参数，所有不把它看做是一层。
- Average pool
- Max pool
- L2 pool

**卷积神经网络：**（总结得非常好）

- 卷积网络的核心思想:
  - 将局部感受野、权值共享以及时间或空间亚采样这三种结构思想结合起来获得了某种程度的位移、尺度、形变不变性。
- 层间联系和空域信息的紧密关系，使其适于图像处理和理解。
  - 图像和网络的拓扑结构能很好的吻合
- **避免了显式的特征抽取，而隐式地从训练数据中进行学习**。
    - 特征提取和模式分类同时进行，并同时在训练中产生;
    - 权重共享可以减少网络的训练参数，使神经网络结构变得更简单， 适应性更强。

### 1.7 感知器（Perceptron）模型

> PRML 的一个重点内容。
>
> 人工神经网络 2020-09-29
>
> 自动化所吴老师

感知器实质上是一种神经元模型。

感知器特性：

- 可分性：true if some parameters get the training set perfectly correct Can represent AND, OR, NOT, etc., but not XOR.
- 感知器收敛定理：若训练数据集是线性可分的，则感知机模型收敛。

缺点：

- 噪声（不可分情况）
- 泛化性

### 1.8 LSTM

> 序列数据的深度学习模型 2020-10-20
>
> 自动化所的吴老师

RNN：

- BPTT, Back Propagation Through Time

长序列循环神经网络：BP 困难，梯度得膨胀和消散。

- RNN 单元（Gated Recurrent Unit(GRU)）

  - $a^{<t>} = tanh(W_{ax}x^{t} + W_{aa}a^{t-1} + b_a)$
  - $\check{y}^{<t>} = softmax(W_{ya}a^{<t>} + b_y)$
- GRU
  - 有 2 个门
- Long Short Term Memory, LSTM
  - 解决了 RNN 长期（like hundreds of time steps）记忆的问题
  - LSTM 是一个存储单元，使用 logistic 和 linear 单元执行乘法运算
  - 3 个门和一个 cell

### 1.9-1.13 数理逻辑



### 1.14 多臂赌博机

> 2020.12.06
>
> 计算所的沈老师

问题形式化：$q_{*}(a) \dot{=} E[R_t \mid A_t = a]$

玩家在第 $t$ 轮时只能依赖于当时对 $q_*(a)$ 的估值 $Q_t(a)$ 进行选择，此时，贪心策略在第 $t$ 轮选择 $Q_t(a)$ 最大的 $a$ 。

![](./20201226/3.jpg)

贪心策略的形式化表示：$A_t \dot{=} arg \max_{a} Q_t(a)$

$\epsilon $ 贪心策略：

- 以概率 $1-\epsilon$ 按照贪心策略进行行为选择——Exploitation
- 以概率 $\epsilon$ 在所有行为中随机选择一个——Exploration
- $\epsilon$ 的取值取决于 $q_*(a)$ 的方差，方差越大 $\epsilon$ 取值应越大。

行为估值方法：根据历史观测样本的均值对 $q_*(a)$ 进行估计。

- 一般性的行为估值：$NewEstimate \leftarrow OldEstimate + StepSize[Target - OldEstimate] $
- 非平稳问题下的行为估计：$Q_{n+1} \dot{=} Q_n + \alpha[R_n-Q_n]$
- UCB（Upper-Confidence-Bound）置信上界 

UCB vs  𝜺 

- UCB 策略一般会优于 𝜺 贪心策略，不过最初几轮相对较差。

- UCB 策略实现起来比 𝜺 贪心策略要复杂，在多臂赌博机之外 的强化学习场景中使用较少。

梯度赌博机算法：是一种随机策略。

三种方法的比较：

![](./20201226/4.jpeg)



多臂赌博机总结：

- 多臂赌博机是强化学习的一个简化场景：行为和状态之间没有关联关系
- 扩展情形：有上下文的多臂赌博机（Contextual bandit）
- 更一般的情形：马尔科夫决策过程























