# 3-6 numpy 数组的合并与分割


```python
import numpy as np
```


```python
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
```


```python
x
```




    array([1, 2, 3])




```python
y
```




    array([3, 2, 1])




```python
# concatenate((a1, a2, ...), axis=0, out=None) 拼接
np.concatenate([x, y])
```




    array([1, 2, 3, 3, 2, 1])




```python
z = np.array([666, 666, 666])
```


```python
np.concatenate([x, y, z])
```




    array([  1,   2,   3,   3,   2,   1, 666, 666, 666])




```python
# A有2个样本，3个特征
A = np.array([[1, 2, 3],
              [4, 5, 6]])
```


```python
np.concatenate([A, A])
```




    array([[1, 2, 3],
           [4, 5, 6],
           [1, 2, 3],
           [4, 5, 6]])




```python
# 沿着另一个维度拼接
np.concatenate([A, A], axis=1)
```




    array([[1, 2, 3, 1, 2, 3],
           [4, 5, 6, 4, 5, 6]])




```python
# 将样本z拼接到A：维度不同的拼接
A2 = np.concatenate([A, z.reshape(1, -1)])
A2
```




    array([[  1,   2,   3],
           [  4,   5,   6],
           [666, 666, 666]])




```python
A
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
# 垂直方向堆叠 2x3 - 1x3
np.vstack([A, z])
```




    array([[  1,   2,   3],
           [  4,   5,   6],
           [666, 666, 666]])




```python
B = np.full((2, 2), 100)
B
```




    array([[100, 100],
           [100, 100]])




```python
# 2x3-2x2
np.hstack((A, B))
```




    array([[  1,   2,   3, 100, 100],
           [  4,   5,   6, 100, 100]])



## 分割操作

# x = np.arange(10)
x


```python
# 分成三段
np.split(x, [3, 7])
```




    [array([0, 1, 2]), array([3, 4, 5, 6]), array([7, 8, 9])]




```python
x1, x2, x3 = np.split(x, [3, 7])
```


```python
# 分成两段
np.split(x, [5])
```




    [array([0, 1, 2, 3, 4]), array([5, 6, 7, 8, 9])]




```python
A = np.arange(16).reshape((4, 4))
A
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])




```python
np.split(A, [2])
```




    [array([[0, 1, 2, 3],
            [4, 5, 6, 7]]),
     array([[ 8,  9, 10, 11],
            [12, 13, 14, 15]])]




```python
A1, A2 = np.split(A, [2])
```


```python
# 另一个维度进行分割
np.split(A, [2], axis=1)
```




    [array([[ 0,  1],
            [ 4,  5],
            [ 8,  9],
            [12, 13]]),
     array([[ 2,  3],
            [ 6,  7],
            [10, 11],
            [14, 15]])]




```python
# 垂直方向分割
np.vsplit(A, [2])
```




    [array([[0, 1, 2, 3],
            [4, 5, 6, 7]]),
     array([[ 8,  9, 10, 11],
            [12, 13, 14, 15]])]




```python
# 水平方向
np.hsplit(x, [2])
```




    [array([0, 1]), array([2, 3, 4, 5, 6, 7, 8, 9])]




```python
data = np.arange(16).reshape((4, 4))
data
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])



data 表示有 4 个样本，每个样本包含 3 个特征和 1 个标签。

在运行机器学习算法时，需要将样本的特征和标签分割。


```python
X, y = np.hsplit(data, [-1])
```


```python
X
```




    array([[ 0,  1,  2],
           [ 4,  5,  6],
           [ 8,  9, 10],
           [12, 13, 14]])




```python
y
```




    array([[ 3],
           [ 7],
           [11],
           [15]])




```python
# 除去为1的维度，还有一个很方便的函数 squeeze()
y[:, 0]
```




    array([ 3,  7, 11, 15])




```python
np.squeeze(y)
```




    array([ 3,  7, 11, 15])


