# PRML 计算散度矩阵和 K-L 变换（5）

## 1. 类间散度矩阵 $S_b$ 和类内散度矩阵 $S_w$ 的计算

> $$w_1 : \{(1,0)^T, (2,0)^T, (1,1)^T \}$$
>
> $$w_2: \{(-1,0)^T, (0, 1)^T, (-1, 1)^T\}$$
>
> $$w_3 : \{ (-1,-1)^T, (0,-1)^T, (0,-2)^T \}$$
>
> 先验概率相等。
>
> 问：计算 $S_w$ 和 $S_b$ ❓

- 类内散度矩阵 $S_w$：
  - $$S_w = \sum_{i=1}^{3} P(w_i) E \{ (x_i - m_1)(x_i - m_i)^T \mid w_i \}$$
    - $E \{ (x_i - m_1)(x_i - m_i)^T \mid w_i \}$ ：以 $w_1$ 为例，有
    - (i) $$X_1 = \left[ \begin{matrix}1&2&1\\ 0&0&1\end{matrix} \right] $$ 
    - (ii) $$m_1 = \left[ \begin{matrix} \frac{4}{3} \\ \frac{1}{3} \end{matrix} \right]  $$
    - $$\Rightarrow (X_1 - m_1)(X_1 - m_1)^T = \left[ \begin{matrix}&\\ &\end{matrix} \right]_{2\times 2}  $$
- 类间散度矩阵 $S_b$：
  - $$S_b = \sum_{i=1}^{3} P(w_i)(m_i - m_0)(m_i - m_0)^T$$

（0）先验概率 $P(w_1) = P(w_2) = P(w_3) = \frac{1}{3}$

（1）类均值向量

$$m_1 = [\frac{4}{3}, \frac{1}{3}]^T$$

$$m_2 = [-\frac{2}{3},\frac{2}{3}]^T$$

$$m_3 = [-\frac{1}{3},-\frac{4}{3}]^T$$

（2）总体均值向量

$$m_0 = [\frac{1}{9}, -\frac{1}{9}]^T$$

（3）类间散布矩阵

$$S_b = \sum_{i=1}^{3} P(w_i) (m_i - m_0)(m_i - m_0)^T$$

$$S_b = \frac{1}{3} \left\{ \left[ \begin{matrix}&\\ &\end{matrix} \right]_{2\times 2}  + \left[ \begin{matrix}&\\ &\end{matrix} \right]_{2\times 2}  +\left[ \begin{matrix}&\\ &\end{matrix} \right]_{2 \times 2}   \right\}  = \left[ \begin{matrix}\frac{62}{81} & \frac{13}{81} \\ \frac{13}{81} & \frac{62}{81} \end{matrix} \right]_{2 \times 2}  $$

```matlab
% octave
((m1 - m0)*(m1 - m0)' + (m2 - m0)*(m2 - m0)' + (m3 - m0)*(m3 - m0)') ./ 3
```

（4）类间散度矩阵

$$S_w = \sum_{i=1}^{3} P(w_i) E \{ (x_i - m_1)(x_i - m_i)^T \mid w_i \}$$

$$S_w = \frac{1}{3} \left\{ \left[ \begin{matrix}&\\ &\end{matrix} \right]_{2\times 2}  + \left[ \begin{matrix}&\\ &\end{matrix} \right]_{2\times 2}  +\left[ \begin{matrix}&\\ &\end{matrix} \right]_{2 \times 2}   \right\}  = \left[ \begin{matrix}\frac{2}{9} & -\frac{1}{27} \\ -\frac{1}{27} & \frac{2}{9} \end{matrix} \right]_{2 \times 2}$$

---

## K-L 降维

计算出 $S_w$ 和 $S_b$ 之后，Fisher 的准则函数为：

$$J_F (W) = \frac{W^T S_b W}{W^T S_w W}$$

（5）使得 $J_F (W)$ 取得最大的 $W^*$ 为：

- 二类：$$W^* = S_{w}^{-1} (m1-m2)$$

（6）对训练集内的所有样本进行投影：

$$y = (W^*)^T X$$

（7）计算在投影空间上的分割阈值 $y_0$ 有两种方式：

在一维 $Y$ 空间，各样本均值 

$$\tilde{m}_i = \frac{1}{N_i} \sum_{y \in w_i} y$$

$$i = 1, 2$$ 

- 方式 1：
  - $$y_0 = \frac{N_1 \tilde{m}_1 + N_2 \tilde{m}_2}{N_1 + N_2}$$

- 方式 2：
  - $$y_0 = \frac{\tilde{m}_1 + \tilde{m}_2}{2} + \frac{ln(P(w_1)/P(w_2))}{N_1 + N_2 - 2}$$

（8）根据决策规则进行分类：

$$\begin{cases}y > y_0 & \Rightarrow X \in w_1 \\ y < y_0 & \Rightarrow X \in w_2 \end{cases} $$