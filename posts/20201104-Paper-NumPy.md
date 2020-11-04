# #论文阅读#

论文名称：Array Programming with NumPy

这是一篇关于 NumPy 的论文，最近也在学习 Python，NumPy 也是常用的一个包，可以花点时间读一读这篇论文。

## 摘要

NumPy 是 Python 的一个主要的数组编程库（array programming library）。

数组编程（Array Programming）为访问和操作向量、矩阵或高维数组中的数据提供了一种强大、紧凑的语法，在不同的领域的研究分析中发挥着重要的作用。

本篇论文会介绍几个基本的数组概念，从而探索如何形成一个简单而强大的编程范式来组织、探索和分析科学数据。

NumPy 是构建 Python 科学计算的基础。有一些特殊的项目，会针对性的开发出类似 NumPy 的接口和数组对象。

## 历史

NumPy 之前 Python 有两个数组包，分别是 Numeric 和 Numarray。

Numeric 始于 1995 年左右，它使用 C 语言结合线性代数的标准编写实现。最早是为了处理来自哈勃空间遥感仪的大型天文图像。

Numarray 是对 Mumeric 的重写，增加了对结构化数组的支持、灵活的索引、内存映射、字节顺序变体（byte-order variants）、更有效的内存使用、灵活的 IEEE 错误处理能力以及更好的类型定义规则。

这两个包相互兼容，但是在很多方面存在差异和分歧。

2005 年，NumPy 作为二者的完美统一体诞生，将 Numarray 的丰富功能与 Numeric 在数组上的高性能及其丰富的 C 应用编程接口结合起来。

15 年后的今天，NumPy 已经成为几乎所有科学计算的 Python 包的基础，包括 SciPy、Matplotlib、pandas、scikit-learn 和 scikit-image。

NumPy 是一个社区开发的开源库，它提供了一个多维的 Python 数组对象以及对其进行操作的函数（array-aware functions）。 

NumPy 使用了 CPU 对内存中的数组进行操作。目前计算机的存储和硬件不断的升级，NumPy 也在不断的更新。

## NumPy 数组

NumPy 数组是一个高效存储和访问多维数组的数据结构，这也被称为张量（Tensors），可以进行各种科学计算。

它由一个指向内存的指针，以及用于解释存储在那里的数据的元数据组成，特别是数据类型（data type）、形状（shape）和步长（strides）。

![](./20201104/1.png)



如下图，显示了 NumPy 数组包含的几个基本的数组概念。

- a：NumPy 数组数据结构及其相关的元数据字段（metadata fields）。
- b：用 `slices` 和 `steps` 对数组进行索引，这些操作返回原始数据的视图。
- c：用掩码（）、标量坐标或其他数组对数组进行索引，使其返回原始数据的副本。
- d：向量化高效地操作数组元素。 
- e：在二维数组的乘法中广播（broadcasts）。在这个例子中，一个数组沿选择 `axes` 求和产生一个向量，或者沿两个 `axes` 连续求和产生一个标量。 
- f：还原操作沿一个或多个 `axes` 进行。在这个例子中，一个数组沿着选定的 `axes` 求和产生一个向量，或者沿着两个 `axes` 连续求和产生一个标量。
- g：NumPy 代码示例。

![](./20201104/2.png)

数组的形状决定了每个轴上元素的数量，轴的数量就是数组的维数。例如，一个矢量数字可以存储为一个 `(N, 1)` 的向量，彩色视频存储为 `(T, M, N, 3)`。

`Strides`（步幅）是将线性存储元素的计算机内存解释为多维数组的必要条件。它描述了在内存中从一行跳到另一行，从一列跳到另一列等要向前移动的字节数。例如，考虑一个形状为 `(4, 3)` 的二维浮点数组，每个元素在内存中占据 8 个字节。要在相邻的列之间移动，则需要在内存中向前或向后移动 8 个字节，访问下一行需要移动 3×8=24 个字节（a）。

![](./20201104/3.png)

NumPy 可以按照 C 一样的内存顺序进行存储数组，先迭代行或先迭代列，这使得用 C 语言编写的外部库可以直接访问内存中的 NumPy 数组数据。

用户使用索引（indexing）访问子数组或单个元素、运算符（向量化）以及 array-aware functions 与 NumPy 数组进行交互。

对数组进行索引，可以返回单个元素、子数组或满足特定条件的元素（b）。

数组甚至可以使用其他数组进行索引（c）。

只要有可能，检索子数组的索引就会返回原始数组的视图，这样两个数组之间就可以共享数据。这提供了一种强大的方法来操作数组数据的子集，同时限制了内存的使用。

> "Wherever possible, indexing that retrieves a subarray returns a view on the original array, such that data is shared between the two arrays. This provides a powerful way to operate on subsets of array data while limiting memory usage."

NumPy 包含了对数组进行矢量化（对整个数组进行操作）计算的函数，包括算术、统计和三角函数（d）。

在 C 语言中需要几十行代码才能完成的操作，Python 只需要简洁清晰的一行代码。在内部的细节中，NumPy 以近乎最佳的方式处理数组元素的循环，考虑到例如步长（strides），以最好地利用计算机的快速缓存内存。

当两个形状相同的数组执行向量化操作时，其结果显而易懂。当形状不同时，广播（broadcasting）机制可以实现相同的效果。

举一个例子，将一个标量值加到一个数组上，广播机制可以将这个标量分别加到数组的每一个元素上。

其他的 array-aware 函数，如：`sum/mean/max` ，创建（creating）、重塑（reshaping）、连接（concatenating）和填充（padding）数组，搜索、排序和计数数据，读写文件，随机数、各种概率分布等。

> Altogether, the combination of a simple in-memory array representation, a syntax that closely mimics mathematics, and a variety of array-aware utility functions forms a productive and powerfully expressive array programming language. 

## Python 生态

Python 是一门开源的、通用的、解释型的编程语言，非常适合于标准的编程任务，如清理数据、与网络资源交互和解析文本。

NumPy 为 Python 增加了快速数组运算和线性代数，使得科学家们可以在 Python 中完成所有的工作，而且 Python 具有的易学易教的优势，许多大学都将其作为主要学习语言。

NumPy 不是 Python 标准库的一部分，因此 NumPy 能够决定自己的发布策略和开发模式。

SciPy、Matplotlib 与 NumPy 有紧密的联系。SciPy 提供了科学计算的基本算法，包括数学、科学和工程例程。Matplotlib 用于数据的可视化。

NumPy、SciPy 和 Matplotlib 的结合，再加上 IPython 或 Jupyter 这样的高级交互环境，为Python 的数组编程提供了坚实的基础。

![](./20201104/4.png)

- eht-imaging：用于射电干涉测量成像、分析和模拟。这个库依赖于 Python 生态系统的许多低级组件。如 NumPy 数组用于存储和处理每一步的数值数据：从原始数据到校准和图像重建。
- scikit-image：SciPy 的拓展图像处理库，提供了更高层次的功能，如边缘过滤器和Hough变换。
- NetworkX：一个用于复杂网络分析的软件包，用于验证图像对比的一致性。
- Astropy：处理标准的天文文件格式，并计算时间/坐标变换

交互式编程环境非常适合探索性数据分析，用户可以流畅地检查、操作和可视化他们的数据，并快速迭代以完善编程语句。

探索性之外的科学计算工作通常是在集成开发环境（IDE）中完成的。

这一段我不太理解：（Keywords：分布式、自动化测试）

> To complement this facility for exploratory work and rapid prototyping, NumPy has developed a culture of employing time-tested software engineering practices to improve collaboration and reduce error [31]. This culture is not only adopted by leaders in the project but also enthusiastically taught to new- comers. The NumPy team was early in adopting distributed revision control and code review to improve collaboration on code, and continuous testing that runs an extensive battery of automated tests for every proposed change to NumPy. The project also has comprehensive, high-quality documentation, integrated with the source code [32, 33, 34].

大概的意思就是这一种新的机制非常的好，非常 nice。已经被建立在 NumPy 生态基础上的库所采用。例如，在英国皇家天文学会对 Astropy 库就给出了如下夸奖。

> “The Astropy Project has provided hundreds of junior scientists with experience in professional-standard software development practices including use of version control, unit testing, code review and issue tracking procedures. This is a vital skill set for modern researchers that is often missing from formal university education in physics or astronomy.”

目前数据科学、机器学习和人工智能发展迅猛，Python 被大规模的推广使用。现在在自然科学和社会科学领域中，几乎每一个学科都有一些库，这些库已经成为许多领域的主要软件环境。

NumPy 及其生态是全球社区会议和研讨会的焦点，NumPy 和它的 API 已经变得真正的无处不在。

## Array proliferation and interoperability 

NumPy 既可以在嵌入式设备上运行，也可以在世界上最大的超级计算机上运行，其性能接近于编译语言。NumPy 从其诞生到现在，负责了绝大多数的数组科学计算。

目前大部分的科学数据集通常会超过单台计算机的内存容量，这些数据集存储在多台机器上或云端。

此外，最近深度学习和人工智能应用的需求导致了专门的加速器硬件的出现，例如，图形处理单元（GPU）、张量处理单元（TPU）和现场可编程门阵列（FPGA）。

NumPy 目前还不能直接利用这种专用的硬件。

> However, both distributed data and the parallel execution of GPUs, TPUs, and FPGAs map well to the paradigm of array programming: a gap, therefore, existed between available modern hardware architectures and the tools necessary to leverage their computational power.

为了解决这个问题，每个深度学习框架都有针对性的创建了自己的数组（array）：PyTorch Tensorflow、Apache MXNet、JAX 等，这些数组实现都有能力以分布式方式在 CPU 和 GPU 上运行。

此外，还有一些项目是建立在 NumPy 数组作为数据容器的基础上，并扩展其功能。这样的库通常会模仿 NumPy API，提供了稳定的数组编程接口，这能大大的吸引新人，或者降低学习的成本。

这防止了 NumPy 重蹈 Numeric 和 Numarray 的破坏性分裂的覆辙。

探索新的数组工作方式本质上是实验性的，目前几个比较有前途的库 Theano、Caffe 已经停止开发了。每次用户决定尝试一种新的库（框架）时，必须改变 import 语句，并确保新的库实现了他们目前使用的 NumPy API 的所有部分。

在理想情况下，用户使用  NumPy function 或 semantics 编写一次代码，然后根据实际情况在NumPy 数组、GPU 数组、分布式数组等之间进行切换。NumPy 提供了一个规范完善的 API。

![](./20201104/5.png)

为了促进这种互操作性，NumPy 提供了协议（protocols），允许将专门的数组传递给NumPy 函数（Fig. 3）。而 NumPy 则根据需要将操作分配给如 Dask、CuPy、xarray 和 PyData/Sparse。

例如，用户现在可以使用 Dask 将他们的计算从单机扩展到分布式系统。

关于大规模部署，我不太明白：

> The protocols also compose well, allowing users to redeploy NumPy code at scale on distributed, multi- GPU systems via, for instance, CuPy arrays embedded in Dask arrays. Using NumPy’s high-level API, users can leverage highly parallel code execution on multiple systems with millions of cores, all with minimal code changes [42].

这些数组协议现在是 NumPy 的一个关键功能，具有很高的重要性。与 NumPy 的其他部分一样，协议是在不断完善和增加的，以提高实用性和简化采用。

## 总结

NumPy 将 array programming 的表达能力、C 语言的性能、Python 的易读性、可用性和通用性等优点结合在一起，形成了一个成熟的、经过良好测试的、有良好文档的、由社区开发的库。

Python 生态中的库提供了大多数重要算法的快速实现。

在需要极端的要求高性能的情况下，可以使用如 Cython、Numba 和 Pythran 等编译型语言，这些语言扩展了 Python 并透明地加速了瓶颈。

由于 NumPy 的简单内存模型，可以使用低级编程语言如 C 来操作 NumPy 数组，然后将其传回给 Python。此外，使用数组协议，可以在对现有代码进行最小改动的情况下，利用全部的专用硬件加速代码运行。

NumPy 最初是由学生、教师和研究人员开发的，目的是为 Python 提供一个先进的、开源的数组编程库，它可以免费使用。

可以想象这样的场景：一群志同道合的人，为了共同的利益，一起建立了一些有意义的东西。

这些最初的开发者使用 C 等低级的编程语言，参考了其他强大的科学计算交互式编程语言如 MATLIB 来编写代码。

最初的版本是在 Python 中添加一个数组对象，到后来通过不断升级迭代，成为一个充满活力的工具生态的基础。现在，大量的科学工作都依赖于 NumPy。NumPy 已经成为了一个核心的科学基础设施。

现在的 NumPy 项目开发流程已经成熟，这个项目有正式的管理结构，并由 NumFOCUS 提供财政赞助，NumFOCUS 是一个非营利性组织，旨在促进研究、数据和科学计算方面的开放实践。

在赞助资金的支持下，该项目能够（也是）在多个月内持续专注于实现实质性的新功能和改进。尽管如此，它仍然在很大程度上依赖于研究生和研究人员在业余时间做出的贡献。

NumPy 不再只是科学 Python 生态的基础数组库，而且已经成为张量（tensor）计算的标准 API，也是 Python 中数组类型和技术之间的协调机制。目前仍然在继续努力扩展和改进相关互操作性功能。

在未来的十年，NumPy 将面临着不少的挑战。

- 设备仪器的升级，导致科学数据收集的规模继续扩大
- 专有硬件的提升不协调
- 新一代编程语言、解释器和编译器可能出现

NumPy 已经准备好迎接挑战，并继续在交互式科学计算中发挥领导作用。要做到这一点，需要政府、学术界和工业界的持续资助。但更重要的是，它还需要新一代的研究生和其他开发者的参与贡献，以建立一个满足下一个十年数据科学需求的 NumPy。

## 方法

使用 Git 进行版本控制，项目托管在 GitHub，利用 GitHub 的 pull request（PR）机制，在合并代码之前进行审核，还使用 GitHub 的 issue 来收集社区用户的改进建议。

## 库组织

NumPy 库由以下几个部分组成：

- 数据结构 `ndarray`
- 通用函数（universal functions）
- 一组用于操作数组和进行科学计算的库函数
- 单元测试和 Python 包构建的基础库
- 用于在 Python 中包装 Fortran 代码的程序 `f2py`

`ndarray` 和通用函数一般被认为是 NumPy 的核心。

`ndarray` 是 NumPy 的核心数据结构。该数据结构在一个连续的块状内存中存储有规律的同质数据类型，允许有效地表示 n 维数据。

通用函数 `ufuncs` 是使用 C 语言编写的函数，它实现了对 NumPy 数组的高效循环。`ufuncs` 的一个重要特点是内置了广播的实现。例如，函数 `arctan2(x, y)` 是一个接受两个值并计算 $$tan^{-1}(y/x)$$ 的 `ufunc`。

todo：计算库 P14.































