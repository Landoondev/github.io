# 高级人工智能-命题逻辑与一阶谓词逻辑（3）

## 1. 基础知识（选择题）

符号

- $\Rightarrow, \rightarrow $：表示 “蕴涵”（Implication）.

  - $P \rightarrow Q$：读作 “P 推出 Q” 或 “P 蕴涵 Q”

  - “仅当 P 真 Q 假时，P 蕴涵 Q 为假”
- $\Leftrightarrow, \leftrightarrow$ ：表示“当且仅当”

  - $P \leftrightarrow Q$：读作 “P 等价 Q” 或者 “P 当且仅当 Q”
- $\vDash $：$KB \vDash \alpha$，Entailment 是指一件事从另一件事中产生。知识库 $KB$ 包含句子 $\alpha$，如果且仅当 $\alpha$ 在 $KB$ 为真的所有世界中为真。

**真值表（Truth tables for connectives）**

![](./20201227/1.jpeg)

**用于判断永真式（valid）**

![](./20201227/2.jpeg)

**一些定理**

![](./20201227/3.jpeg)

## 2. “永真”（valid） ？

- $(Smoke \Rightarrow Fire) \Rightarrow ((Smoke \wedge Heat) \Rightarrow Fire)$

## 3. “不可满足”（un-satisfiable）？

不可满足公式：谓词公式在任一解释下都为假。例如：

- $$\forall x (P(x) \wedge \neg P(x))$$
- $$\forall x P(X) \wedge \exists x (\neg P(x))$$

如下是一些题目：

- $\beta \vDash \alpha$ 为真，当且仅当 $(\beta \wedge \neg \alpha)$ 是不可满足的。
- 既不是 “永真” 的，又不是 “不可满足的”：$(Smoke \vee Heat) \Rightarrow (Smoke \wedge Heat)$

## 4. 一阶谓词逻辑

使用一阶谓词逻辑表达 “胜者为王，败者为寇”？

$$\forall x (Person(x) \wedge Winner(x)) \Rightarrow king(x)$$

$$\forall x (Person(x) \wedge Loser(x)) \Rightarrow kou(x)$$

## 5. 模糊逻辑



