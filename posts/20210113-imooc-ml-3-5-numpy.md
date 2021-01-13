# 3-5 Numpy.array 的基本操作


```python
import numpy as np
```


```python
x = np.arange(10)
x
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
X = np.arange(15).reshape(3, 5)
X
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])



## 基本属性


```python
x.ndim
```




    1




```python
# 二维数组
X.ndim
```




    2




```python
# 返回一个元组
x.shape
```




    (10,)




```python
# 两个维度
X.shape
```




    (3, 5)




```python
# 元素个数
x.size
```




    10




```python
X.size
```




    15



## numpy.array 数据访问


```python
x
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
# 索引访问
x[0]
```




    0




```python
x[-1]
```




    9




```python
X
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])




```python
# 不建议使用这种方式
X[0][0]
```




    0




```python
# 建议使用 X[0, 0]
X[(0, 0)]
```




    0




```python
# 切片
x[0:5]
```




    array([0, 1, 2, 3, 4])




```python
x[:5]
```




    array([0, 1, 2, 3, 4])




```python
x[5:]
```




    array([5, 6, 7, 8, 9])




```python
# 从头到尾，步长为2
x[::2]
```




    array([0, 2, 4, 6, 8])




```python
x[::-1]
```




    array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])




```python
# 前两行前三列
X[:2, :3]
```




    array([[0, 1, 2],
           [5, 6, 7]])




```python
# 使用两个方括号的缺点
X[:2][:3]
```




    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])




```python
X[:2]
```




    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])




```python
X[:2][:3]
```




    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])




```python
X[:2, ::2]
```




    array([[0, 2, 4],
           [5, 7, 9]])




```python
X[::-1, ::-1]
```




    array([[14, 13, 12, 11, 10],
           [ 9,  8,  7,  6,  5],
           [ 4,  3,  2,  1,  0]])




```python
X[0, :]
```




    array([0, 1, 2, 3, 4])




```python
X[0, :].ndim
```




    1




```python
# 列
X[:, 0]
```




    array([ 0,  5, 10])




```python
subX = X[:2, :3]
subX
```




    array([[0, 1, 2],
           [5, 6, 7]])




```python
subX[0, 0] = 100
```


```python
subX
```




    array([[100,   1,   2],
           [  5,   6,   7]])




```python
# 切片会改变原矩阵
X
```




    array([[100,   1,   2,   3,   4],
           [  5,   6,   7,   8,   9],
           [ 10,  11,  12,  13,  14]])




```python
X[0, 0]=0
X
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])




```python
subX
```




    array([[0, 1, 2],
           [5, 6, 7]])




```python
# copy
subX = X[:2, :3].copy()
subX
```




    array([[0, 1, 2],
           [5, 6, 7]])




```python
subX[0, 0] = 100
subX
```




    array([[100,   1,   2],
           [  5,   6,   7]])




```python
# X 就不会改变
X
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])



### Reshape


```python
x.shape
```




    (10,)




```python
x.reshape(2, 5)
```




    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])




```python
x
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
A = x.reshape(2, 5)
A
```




    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])




```python
x
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
B = x.reshape(1, 10)
B
```




    array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])




```python
B.ndim
```




    2




```python
B.shape
```




    (1, 10)




```python
# x 和 B 在维度上是不同的
x.ndim
```




    1




```python
x.shape
```




    (10,)




```python
# 自动推导列维度
x.reshape(10, -1)
```




    array([[0],
           [1],
           [2],
           [3],
           [4],
           [5],
           [6],
           [7],
           [8],
           [9]])




```python
x.reshape(-1, 10)
```




    array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])




```python
x.reshape(2, -1)
```




    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])




```python
# 10 不能被3整除，报错
x.reshape(3, -1)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-84-9f73776a2fdc> in <module>
          1 # 10 不能被3整除，报错
    ----> 2 x.reshape(3, -1)
    

    ValueError: cannot reshape array of size 10 into shape (3,newaxis)



```python

```
