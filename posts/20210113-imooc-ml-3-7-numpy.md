# 3-4 其他创建  numpy.array 的方法


```python
import numpy as np
```


```python
np.zeros(10)
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
# 默认为浮点型
np.zeros(10).dtype
```




    dtype('float64')




```python
np.zeros(10, dtype=int)
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])




```python
np.zeros((3, 5))
```




    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])




```python
# 良好的代码习惯，注明参数
np.zeros(shape=(3, 5), dtype=int)
```




    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])




```python
np.ones(10)
```




    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])




```python
np.ones((3, 5))
```




    array([[1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.]])




```python
# 整型
np.full(shape=(3, 5), fill_value=666)
```




    array([[666, 666, 666, 666, 666],
           [666, 666, 666, 666, 666],
           [666, 666, 666, 666, 666]])




```python
# 浮点型
np.full(shape=(3, 5), fill_value=666.0)
```




    array([[666., 666., 666., 666., 666.],
           [666., 666., 666., 666., 666.],
           [666., 666., 666., 666., 666.]])




```python
# 顺序可以变
np.full(fill_value=666, shape=(3, 5))
```




    array([[666, 666, 666, 666, 666],
           [666, 666, 666, 666, 666],
           [666, 666, 666, 666, 666]])



### arange


```python
[i for i in range(0, 20, 2)]
```




    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]




```python
[i for i in range(start=0, stop=20, step=2)]
```


    ---------------------------------------------------------------------------
    
    TypeError                                 Traceback (most recent call last)
    
    <ipython-input-17-4ae1aca1c77c> in <module>
    ----> 1 [i for i in range(start=0, stop=20, step=2)]


    TypeError: range() does not take keyword arguments


`range([start,] stop[, step]) -> list of integers`

⚠️ 这个 range() 的参数非常奇怪。不能显示的使用。


```python
np.arange(0, 20, 2)
```




    array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])




```python
# range 步长不能为浮点数
np.range(0, 20, 0.2)
```


    ---------------------------------------------------------------------------
    
    AttributeError                            Traceback (most recent call last)
    
    <ipython-input-25-2c2f91dac55a> in <module>
    ----> 1 np.range(0, 20, 0.2)


    AttributeError: module 'numpy' has no attribute 'range'



```python
# arange 步长可以为浮点数
np.arange(0, 20, 0.2)
```




    array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ,  1.2,  1.4,  1.6,  1.8,  2. ,
            2.2,  2.4,  2.6,  2.8,  3. ,  3.2,  3.4,  3.6,  3.8,  4. ,  4.2,
            4.4,  4.6,  4.8,  5. ,  5.2,  5.4,  5.6,  5.8,  6. ,  6.2,  6.4,
            6.6,  6.8,  7. ,  7.2,  7.4,  7.6,  7.8,  8. ,  8.2,  8.4,  8.6,
            8.8,  9. ,  9.2,  9.4,  9.6,  9.8, 10. , 10.2, 10.4, 10.6, 10.8,
           11. , 11.2, 11.4, 11.6, 11.8, 12. , 12.2, 12.4, 12.6, 12.8, 13. ,
           13.2, 13.4, 13.6, 13.8, 14. , 14.2, 14.4, 14.6, 14.8, 15. , 15.2,
           15.4, 15.6, 15.8, 16. , 16.2, 16.4, 16.6, 16.8, 17. , 17.2, 17.4,
           17.6, 17.8, 18. , 18.2, 18.4, 18.6, 18.8, 19. , 19.2, 19.4, 19.6,
           19.8])



### linspace


```python
# 在 [0, 20] 区间中分出 10 个数
np.linspace(0, 20, 10)
```




    array([ 0.        ,  2.22222222,  4.44444444,  6.66666667,  8.88888889,
           11.11111111, 13.33333333, 15.55555556, 17.77777778, 20.        ])




```python
np.linspace(0, 20, 11)
```




    array([ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18., 20.])



### 随机数 random


```python
np.random.randint(0, 10)
```




    2




```python
np.random.randint(0, 10, 10)
```




    array([0, 8, 5, 8, 2, 1, 4, 2, 9, 5])




```python
# [0, 10) 取不到 10
np.random.randint(low=0, high=10, size=10, dtype=int)
```




    array([5, 5, 2, 1, 9, 5, 0, 5, 2, 4])




```python
# [0, 1) 取不到 1
np.random.randint(low=0, high=1, size=10)
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])




```python
np.random.randint(low=0, high=10, size=(3, 5))
```




    array([[6, 1, 4, 2, 3],
           [8, 4, 8, 7, 2],
           [5, 8, 5, 2, 3]])




```python
# 随机种子
np.random.seed(666)
```


```python
np.random.randint(low=0, high=10, size=(3, 5))
```




    array([[2, 6, 9, 4, 3],
           [1, 0, 8, 7, 5],
           [2, 5, 5, 4, 8]])




```python
# 没有设置随机种子
np.random.randint(low=0, high=10, size=(3, 5))
```




    array([[4, 4, 0, 0, 4],
           [0, 4, 5, 7, 1],
           [0, 0, 6, 6, 0]])




```python
# 设置了随机种子
np.random.seed(666)
np.random.randint(low=0, high=10, size=(3, 5))
```




    array([[2, 6, 9, 4, 3],
           [1, 0, 8, 7, 5],
           [2, 5, 5, 4, 8]])




```python
# [0.0, 1.0) 浮点数
np.random.random(size=10)
```




    array([0.8578588 , 0.76741234, 0.95323137, 0.29097383, 0.84778197,
           0.3497619 , 0.92389692, 0.29489453, 0.52438061, 0.94253896])




```python
# normal(loc=0.0, scale=1.0, size=None) 正态分布 loc 均值 scale 方差
np.random.normal(size=1)
```




    array([-0.35371521])




```python
np.random.normal(0, 1, (3, 5))
```




    array([[-1.95332994, -0.34376486, -1.47693162, -0.70022971,  0.77605168],
           [ 1.18063598,  0.06102404,  1.07856138, -0.79783572,  1.1701326 ],
           [ 0.1121217 ,  0.03185388, -0.19206285,  0.78611284, -1.69046314]])




```python
# 函数文档查询
np.random.normal?


# Docstring:
# normal(loc=0.0, scale=1.0, size=None)

# Draw random samples from a normal (Gaussian) distribution.
```


```python
# 查询文档
np.random?
```


```python
# 文档嵌入 notebook
help(np.random.normal)
```

    Help on built-in function normal:
    
    normal(...) method of numpy.random.mtrand.RandomState instance
        normal(loc=0.0, scale=1.0, size=None)
        
        Draw random samples from a normal (Gaussian) distribution.
        ...
        
        >>> np.random.normal(3, 2.5, size=(2, 4))
        array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
               [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random


