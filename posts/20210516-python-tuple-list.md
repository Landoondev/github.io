# Python 中 tuple 和 list 的区别

见到很多次了。列表  List，元组 tuple。

相同点：

- 都是 **一个可以放置任何数据类型的有序集合**。
- 都支持 负数索引。
- 都支持 切片操作。
- 都可以 随意嵌套。
- 可以通过 `list()` 和 `tuple()` 函数相互转换。

```python
tup = ('jason', 22, 1) # 元组中同时含有 int 和 string 类型的元素
tup
```

```
('jason', 22, 1)
```



```python
l = [1, 2, 3, 4]
l[3] = 40
l
```

```
[1, 2, 3, 40]
```



不同点：

- list 是动态的，长度大小不固定，可以随意地增加、删除或者改变元素（mutable）。
- tuple 是静态的，长度大小固定，无法增加删除或者改变（immutable）。
- `list.reverse()` 、`list.sort()` ，tuple 没有这两个内置函数。
- 

```python
l = [1, 2, 'hello', 'world'] # 列表中同时含有 int 和 string 类型的元素
l
```


    [1, 2, 'hello', 'world']




```python
tup = (1, 2, 3, 4)
tup[3] = 40
```


    ---------------------------------------------------------------------------
    
    TypeError                                 Traceback (most recent call last)
    
    <ipython-input-7-c2b03e8e4127> in <module>
          1 tup = (1, 2, 3, 4)
    ----> 2 tup[3] = 40


    TypeError: 'tuple' object does not support item assignment

如果想对已有的元组做任何“改变”，那就只能重新开辟一块内存，创建新的元组了。



## 列表 和元组存储方式的差异

相同的元素，元组的存储空间比列表要少 16 字节。

```
l = [1, 2, 3]
l.__sizeof__()
64

tup = (1, 2, 3)
tup.__sizeof__()
48
```

因为列表是动态的，需要存储指针，来指向对应到的元素（int 型，8 字节）。

由于列表可变，需要额外存储已经分配的长度大小（8 字节），这样才能实时跟踪列表空间的使用情况，当空间不足时，及时分配额外空间。

```
tup = ()
tup.__sizeof__() # 空的元组存储空间为 24 字节
24

l = []
l.__sizeof__() # 空的列表存储空间为 40 字节
40

l.append(1)
l.__sizeof__() # 加入元素 1，列表分配了可以存储 4 个元素的空间 (70 - 40)/8 = 4
72

l.append(2)
l.__sizeof__() # 由于之前分配了空间，加入元素 2，列表空间不变
72

l.append(3)
l.__sizeof__() # 同上
72

l.append(4)
l.__sizeof__() # 同上
72

l.append(5)
l.__sizeof__() # 加入元素 5，列表空间不足，所有又额外分配了可以存储 4 个元素的空间
104
```

为了减小每次增加/删减操作时空间分配的开销，Python 每次分配空间时都会**额外多分配一些**，这样的机制 (over-allocating) 保证了其操作的高效性：增加/删除的时间复杂度均为 O(1)。



## 元组和列表的性能

元组的性能速度要略优于列表。元组的初始化速度，比列表快 5 倍。

```
%timeit x = (1, 2, 3, 4, 5)
12.3 ns ± 0.0635 ns per loop (mean ± std. dev. of 7 runs, 100000000 loops each)

%timeit x = [1, 2, 3, 4, 5]
54.1 ns ± 0.848 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
```


索引操作两者差不多。

```
x = (1, 2, 3, 4, 5)
%timeit y = x[3]
35.6 ns ± 1.32 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)

x = [1, 2, 3, 4, 5]
%timeit y = x[3]
35.9 ns ± 1.41 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
```



## 应用场景

如果存储的数据和数量不变：选用 tuple 更合适。

如果存储的数据或数量是可变的：选用 list 更合适。



## 列表和元组的底层实现机制

list 本质上是一个 over-allocate 的 array。`ob_item` 是一个指针列表，里面的每一个指针都指向列表的元素。而 allocated 则存储了这个列表已经被分配的空间大小。

```
allocated >= len(list) == ob_size
```

当类别分配的空间已满时，`allocated == len(list)`，则会向系统请求更大的内存空间，并把原来的元素全部拷贝过去。

```
# 列表每次分配的空间大小
0, 4, 8, 16, 25, 35, 46, 58, 72, 88 ...
```



tuple 本质也是一个  array，但是空间大小固定。
















