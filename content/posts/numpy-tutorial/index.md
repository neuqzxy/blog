---
title: NumPy 完全指南
date: 2026-02-11 21:38:03
tags:
  - Python
  - NumPy
  - ML
  - AI
categories:
  - Python
---

## 一、数组创建

### 1.1 基于Python的List/Tuple

```python
# 默认按传入数字的格式确定类型
a1 = np.array([1, 2, 3])
a2 = np.array([[1, 2, 3], [4, 5, 6]])
a3 = np.array([1., 2, 3])
# 也可以传入dtype关键字参数定义类型
a4 = np.array([1, 2, 3], dtype=np.float16)
a5 = np.array([1.2, 2.6, 3.7], dtype=np.int32) # [1 2 3]
```

### 1.2 基于arange

```python
a1 = np.arange(12)
a2 = np.arange(12.)
a3 = np.arange(10, 20, 2) # start, end, step
```

### 1.3 基于linspace / logspace

`linspace` 和 `logspace` 用于创建等差和等比数列，需要 3 个参数：开头、结尾、数量。

```python
# 在 0 到 9 之间生成 10 个等间距的数
a1 = np.linspace(0, 9, 10)  # [0., 1., 2., ..., 9.]

# 在 0 到 9 之间生成 3 个等间距的数
a2 = np.linspace(0, 9, 3)   # [0., 4.5, 9.]

# logspace：生成等比数列，指数 base 默认为 10
a3 = np.logspace(0, 9, 10, base=np.e)  # [e^0, e^1, ..., e^9]

# 验证 linspace 和 logspace 的关系
print(np.all(np.e ** np.linspace(0, 9, 10) == np.logspace(0, 9, 10, base=np.e)))
# True
```

### 1.4 np.ones / np.zeros / np.zeros_like

```python
np.ones((2, 3))
np.zeros((2, 3))
np.zeros_like([[1, 2, 3], [4, 5, 6]]) # shape一样的零矩阵
```

### 1.5 np.random

```python
# 生成单个数
np.random.rand()

# 0-1 均匀分布
np.random.random((2, 3))
# 随机整数（离散均匀分布），指定上下界和 shape
np.random.randint(0, 10, (2, 3))
# 指定上下界的连续均匀分布
np.random.uniform(-1, 1, (2, 3))
# 标准正态分布
np.random.randn(2, 4)
# 高斯分布
np.random.normal(0, 1, (3, 5))

# 新版API
rng = np.random.default_rng(42)
# 0-1 均匀分布
rng.random((2, 3))
# 随机整数（离散均匀分布），指定上下界和 shape
rng.integers(0, 10, (2, 3))
# 指定上下界的连续均匀分布
rng.uniform(-1, 1, (2, 3))
# 标准正态分布
rng.standard_normal((2, 4))
# 高斯分布
rng.normal(0, 1, (3, 5))
# 关键字参数写法的高斯分布
rng.normal(loc=0.0, scale=1.0, size=(2, 3))
```

### 1.6 其他创建方法

```python
# 创建单位矩阵
np.eye(3)  # 3x3 单位矩阵

# 创建对角矩阵
np.diag([1, 2, 3, 4])

# 创建空数组（未初始化，速度快）
np.empty((2, 3))

# 创建填充指定值的数组
np.full((2, 3), 7)  # 用 7 填充

# 创建网格坐标
x = np.arange(0, 5)
y = np.arange(0, 3)
xx, yy = np.meshgrid(x, y)
```

## 二、数组运算

### 2.1 基本运算

NumPy 支持向量化运算，比 Python 循环快得多：

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# 加减乘除
print(a + b)   # [11 22 33 44]
print(a - b)   # [-9 -18 -27 -36]
print(a * b)   # [10 40 90 160]
print(a / b)   # [0.1 0.1 0.1 0.1]

# 幂运算
print(a ** 2)  # [1 4 9 16]

# 数学函数
print(np.sin(a))
print(np.exp(a))
print(np.log(a))
print(np.sqrt(a))
```

### 2.2 矩阵运算

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 逐元素乘法
print(A * B)
# [[ 5 12]
#  [21 32]]

# 矩阵乘法
print(A @ B)        # Python 3.5+ 推荐写法
print(np.dot(A, B)) # 传统写法
# [[19 22]
#  [43 50]]

# 矩阵转置
print(A.T)

# 矩阵求逆
print(np.linalg.inv(A))

# 矩阵行列式
print(np.linalg.det(A))

# 特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
```

### 2.3 广播机制（Broadcasting）

广播是 NumPy 的强大特性，允许不同形状的数组进行运算：

```python
# 标量广播
a = np.array([1, 2, 3])
print(a + 10)  # [11 12 13]

# 一维与二维广播
a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])
print(a + b)
# [[11 22 33]
#  [14 25 36]]

# 不同维度广播
a = np.array([[1], [2], [3]])  # (3, 1)
b = np.array([10, 20, 30])      # (3,)
print(a + b)
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]
```

**广播规则**：
1. 如果数组维度不同，用 1 填充较小数组的形状
2. 如果两个数组在某个维度上大小相同，或其中一个为 1，则兼容
3. 如果两个数组在所有维度上都兼容，则可以广播

## 三、逻辑运算与判断

### 3.1 np.all / np.any

```python
# 所有项均为 True
arr = np.array([True, True, True])
np.all(arr)  # True

# 验证两个数组相等
np.all(np.e ** np.linspace(0, 9, 30) == np.logspace(0, 9, 30, base=np.e))

# 存在一个 True 即可
arr = np.array([-1, 2, -3])
np.any(arr > 0)  # True

# 按轴操作
arr = np.array([[True, False], [True, True]])
np.all(arr, axis=0)  # [True False]
np.any(arr, axis=1)  # [True True]
```

### 3.2 条件判断与过滤

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# 条件判断
mask = arr > 3
print(mask)  # [False False False True True True]

# 条件过滤
print(arr[mask])  # [4 5 6]
print(arr[arr > 3])  # 一步到位

# 多条件
print(arr[(arr > 2) & (arr < 5)])  # [3 4]
print(arr[(arr < 2) | (arr > 5)])  # [1 6]

# np.where：三元运算符
result = np.where(arr > 3, arr, 0)
print(result)  # [0 0 0 4 5 6]

# 条件赋值
arr[arr > 3] = 0
print(arr)  # [1 2 3 0 0 0]
```

## 四、统计度量（Statistical Measure）

```python
arr = np.array([[0.77395605, 0.43887844, 0.85859792, 0.69736803],
       [0.09417735, 0.97562235, 0.7611397 , 0.78606431],
       [0.12811363, 0.45038594, 0.37079802, 0.92676499]])
```

### max/min

```python
m1 = arr.max() # 0.97562235
m2 = arr.max(axis=0) # [0.77395605 0.97562235 0.85859792 0.92676499]
m2 = arr.max(axis=0, keepdims=True) # [[0.77395605 0.97562235 0.85859792 0.92676499]]
m3 = arr.max(axis=1) # [0.85859792 0.97562235 0.92676499]
m3 = arr.max(axis=1, keepdims=True) # out:
# [[0.85859792]
#  [0.97562235]
#  [0.92676499]]
```

### median / quantile

```python
# 中位数
np.median(arr)  # 0.729253865

# 分位数，取第 0.5 分位的数（等价于中位数）
np.quantile(arr, q=0.5)  # 0.729253865

# 分位数，按列取 0.5 分位
np.quantile(arr, q=0.5, axis=0)  # [0.12811363 0.45038594 0.7611397 0.78606431]

# 百分位数（percentile）
np.percentile(arr, 50)  # 等价于 median
np.percentile(arr, [25, 50, 75])  # 四分位数
```

### mean / average / sum / std / var

均值/加权平均/求和/标准差/方差

```python
# 均值（mean）
np.mean(arr)  # 所有元素的平均值
np.mean(arr, axis=0)  # 每列的平均值

# 加权平均（average）
weights = np.array([1, 2, 3, 4])
np.average(arr[0], weights=weights)

# 求和
np.sum(arr, axis=0)  # 按列求和
np.sum(arr, axis=1)  # 按行求和 [2.76880044 2.61700371 1.87606258]
np.cumsum(arr)  # 累计求和

# 标准差
np.std(arr)  # 0.28783096560658955

# 方差
np.var(arr)  # 0.08284666476202175

# 其他统计函数
np.prod(arr)  # 所有元素的乘积
np.cumprod(arr)  # 累计乘积
```

### argmax / argmin

返回最大/最小值的索引

```python
arr = np.array([[9, 2, 3],
                [4, 8, 6]])

# 全局最大值索引（展平后）
np.argmax(arr)  # 0

# 按轴最大值索引
np.argmax(arr, axis=0)  # [0 1 1]
np.argmax(arr, axis=1)  # [0 1]

# 最小值索引
np.argmin(arr, axis=0)  # [1 0 0]
```

## 五、数组形状操作

### 5.1 查看形状属性

```python
arr = np.random.randint(0, 10, (2, 3))

print(arr.shape)  # (2, 3) - 形状
print(arr.size)   # 6 - 元素总数
print(arr.ndim)   # 2 - 维度数
print(arr.dtype)  # dtype('int64') - 数据类型
print(arr.itemsize)  # 8 - 每个元素字节数
print(arr.nbytes)    # 48 - 总字节数
```

### 5.2 改变形状

```python
arr = np.arange(12)  # [0, 1, 2, ..., 11]

# flatten：展平为一维（返回副本）
arr_flat = arr.reshape(3, 4).flatten()

# ravel：展平为一维（返回视图，更快）
arr_ravel = arr.reshape(3, 4).ravel()

# reshape：改变形状（返回视图）
arr.reshape(3, 4)    # 3行4列
arr.reshape(2, -1)   # 2行，列数自动计算
arr.reshape(-1, 4)   # 4列，行数自动计算
arr.reshape(2, 2, 3) # 三维数组

# resize：就地修改形状（会改变原数组）
arr2 = arr.copy()
arr2.resize((2, 6))  # 注意：会修改原数组

# newaxis：增加维度
arr[np.newaxis, :]   # (1, 12)
arr[:, np.newaxis]   # (12, 1)

# 注意：reshape 返回视图（如果可能），flatten 返回副本。修改视图会影响原数组
```

### expand_dims/squeeze

升维/降维

```python
# 升维
np.expand_dims(arr, 0)
# [[[0, 6, 4],
#   [6, 7, 2]]]
np.expand_dims(arr, 1)
# [[[0, 6, 4]],
#  [[6, 7, 2]]]
np.expand_dims(arr, 2)
# [[[0],
#   [6],
#   [4]],
#  [[6],
#   [7],
#   [2]]]

expanded = np.expand_dims(arr, 1)
# [[[0, 6, 4]],
#  [[6, 7, 2]]]
np.squeeze(expanded, axis=1)
# [[0, 6, 4],
#  [6, 7, 2]]
```

## 六、转置与轴操作

### 6.1 转置

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# 二维矩阵转置
arr.T
# [[1, 4],
#  [2, 5],
#  [3, 6]]

# 多维矩阵转置（指定轴顺序）
arr3d = np.arange(24).reshape(2, 3, 4)
np.transpose(arr3d, (2, 0, 1))  # 交换轴的顺序

# swapaxes：交换两个轴
arr3d.swapaxes(0, 2)

# moveaxis：移动轴到新位置
np.moveaxis(arr3d, 0, -1)
```

### 6.2 轴操作技巧

```python
# 在指定轴上操作
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# axis=0：沿着行方向（列之间）
np.sum(arr, axis=0)  # [5, 7, 9]

# axis=1：沿着列方向（行之间）
np.sum(arr, axis=1)  # [6, 15]

# keepdims=True：保持维度
np.sum(arr, axis=0, keepdims=True)  # [[5, 7, 9]]
```

## 七、数组索引与切片

### 7.1 基本索引

```python
arr = np.array([[ 9, 77, 65],
                [44, 43, 86],
                [ 9, 70, 20],
                [10, 53, 97]])

# 单个元素
arr[0, 1]  # 77（相当于 arr[0][1]，但更快）

# 第 1-3 行（不包括第3行）
arr[1:3]
# [[44, 43, 86],
#  [ 9, 70, 20]]

# 第 1-2 行，第 1 列
arr[1:3, 1]  # [43, 70]

# 从开始到第 3 行，第 1-3 列
arr[:3, 1:3]

# 使用步长：第 1、3 行，第 0、2 列
arr[1:4:2, 0:3:2]

# 所有其他维度，第 1 列
arr[..., 1]  # [77, 43, 70, 53]

# 反转数组
arr[::-1]     # 反转行
arr[:, ::-1]  # 反转列
```

### 7.2 高级索引

```python
# 整数数组索引（花式索引）
# 选择第 0, 3 行
arr[[0, 3]]
# [[ 9, 77, 65],
#  [10, 53, 97]]

# 同时指定行和列
arr[[0, 3], [1, 2]]  # [77, 97]

# 使用索引数组
rows = np.array([0, 1, 2])
cols = np.array([0, 1, 2])
arr[rows, cols]  # [9, 43, 20]（对角线元素）

# 布尔索引（掩码）
mask = arr > 50
arr[mask]  # [77, 65, 86, 70, 53, 97]

# 组合使用
arr[arr[:, 0] > 10]  # 第0列大于10的所有行
```

### 7.3 视图 vs 副本

```python
# 切片返回视图（修改会影响原数组）
arr = np.array([1, 2, 3, 4, 5])
view = arr[1:4]
view[0] = 999
print(arr)  # [1, 999, 3, 4, 5]

# 花式索引返回副本（修改不影响原数组）
copy = arr[[1, 2, 3]]
copy[0] = 888
print(arr)  # [1, 999, 3, 4, 5]（未改变）

# 显式复制
true_copy = arr.copy()
```

## 八、数组拼接、堆叠与分割

NumPy 提供了多种方式来组合和分割数组。

```python
rng = np.random.default_rng(42)

arr1 = rng.random((2, 3))
arr2 = rng.random((2, 3))
arr1, arr2

'''
(array([[0.77395605, 0.43887844, 0.85859792],
        [0.69736803, 0.09417735, 0.97562235]]),
 array([[0.7611397 , 0.78606431, 0.12811363],
        [0.45038594, 0.37079802, 0.92676499]]))
'''
```

### 8.1 concatenate（拼接）

```python
# 默认沿 axis=0 连接（垂直拼接）
np.concatenate((arr1, arr2))
# 等价于 vstack（vertical stack）
np.vstack((arr1, arr2))
'''
array([[0.77395605, 0.43887844, 0.85859792],
       [0.69736803, 0.09417735, 0.97562235],
       [0.7611397 , 0.78606431, 0.12811363],
       [0.45038594, 0.37079802, 0.92676499]])
'''

# 沿 axis=1 连接（水平拼接）
np.concatenate((arr1, arr2), axis=1)
# 等价于 hstack（horizontal stack）
np.hstack((arr1, arr2))
'''
array([[0.77395605, 0.43887844, 0.85859792, 0.7611397 , 0.78606431,
        0.12811363],
       [0.69736803, 0.09417735, 0.97562235, 0.45038594, 0.37079802,
        0.92676499]])
'''

# 拼接多个数组
arr3 = rng.random((2, 3))
np.concatenate([arr1, arr2, arr3], axis=0)
```

### 8.2 stack（堆叠）

`stack` 会创建新的维度，而 `concatenate` 是在已有维度上拼接。

```python
# 堆叠，默认根据 axis=0 进行（增加第一维）
np.stack((arr1, arr2))
'''
array([[[0.77395605, 0.43887844, 0.85859792],
        [0.69736803, 0.09417735, 0.97562235]],

       [[0.7611397 , 0.78606431, 0.12811363],
        [0.45038594, 0.37079802, 0.92676499]]])
shape: (2, 2, 3)
'''

# 堆叠，根据 axis=2（在最后增加一维）
np.stack((arr1, arr2), axis=2)
'''
array([[[0.77395605, 0.7611397 ],
        [0.43887844, 0.78606431],
        [0.85859792, 0.12811363]],

       [[0.69736803, 0.45038594],
        [0.09417735, 0.37079802],
        [0.97562235, 0.92676499]]])
shape: (2, 3, 2)
'''

# column_stack：按列堆叠一维数组
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.column_stack((a, b))
# [[1, 4],
#  [2, 5],
#  [3, 6]]

# row_stack：按行堆叠（等价于 vstack）
np.row_stack((a, b))
# [[1, 2, 3],
#  [4, 5, 6]]
```

### 8.3 repeat / tile（重复）

```python
arr = np.array([[1, 2], [3, 4]])

# repeat：沿指定轴重复元素
np.repeat(arr, 2, axis=0)
# [[1, 2],
#  [1, 2],
#  [3, 4],
#  [3, 4]]

np.repeat(arr, 2, axis=1)
# [[1, 1, 2, 2],
#  [3, 3, 4, 4]]

# tile：重复整个数组
np.tile(arr, 2)  # 水平重复2次
# [[1, 2, 1, 2],
#  [3, 4, 3, 4]]

np.tile(arr, (2, 1))  # 垂直重复2次
# [[1, 2],
#  [3, 4],
#  [1, 2],
#  [3, 4]]

np.tile(arr, (2, 3))  # 垂直2次，水平3次
# [[1, 2, 1, 2, 1, 2],
#  [3, 4, 3, 4, 3, 4],
#  [1, 2, 1, 2, 1, 2],
#  [3, 4, 3, 4, 3, 4]]
```

### 8.4 split（分割）

```python
arr = np.arange(12).reshape(4, 3)
# [[ 0,  1,  2],
#  [ 3,  4,  5],
#  [ 6,  7,  8],
#  [ 9, 10, 11]]

# 垂直分割（按行分）
np.split(arr, 2, axis=0)
# 等价于 np.vsplit(arr, 2)
'''
[array([[0, 1, 2],
        [3, 4, 5]]),
 array([[ 6,  7,  8],
        [ 9, 10, 11]])]
'''

# 水平分割（按列分）
np.split(arr, 3, axis=1)
# 等价于 np.hsplit(arr, 3)
'''
[array([[0],
        [3],
        [6],
        [9]]),
 array([[ 1],
        [ 4],
        [ 7],
        [10]]),
 array([[ 2],
        [ 5],
        [ 8],
        [11]])]
'''

# 不等分割（指定分割点）
np.array_split(arr, [1, 3], axis=0)  # 在索引1和3处分割
'''
[array([[0, 1, 2]]),
 array([[3, 4, 5],
        [6, 7, 8]]),
 array([[ 9, 10, 11]])]
'''
```

## 九、数组排序与搜索

### 9.1 排序

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# 返回排序后的数组（不修改原数组）
np.sort(arr)  # [1, 1, 2, 3, 4, 5, 6, 9]

# 就地排序（修改原数组）
arr.sort()

# 返回排序后的索引
np.argsort(arr)  # [1, 3, 6, 0, 2, 4, 7, 5]

# 二维数组排序
arr2d = np.array([[3, 1, 4],
                  [1, 5, 9]])
np.sort(arr2d, axis=0)  # 按列排序
np.sort(arr2d, axis=1)  # 按行排序

# 部分排序（找最小的k个元素）
np.partition(arr, 3)  # 第3个元素左边都比它小，右边都比它大
```

### 9.2 搜索

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# 查找元素（二分查找，要求数组已排序）
np.searchsorted(arr, 5)  # 4（索引）

# 查找多个元素
np.searchsorted(arr, [3, 5, 7])  # [2, 4, 6]

# 查找非零元素的索引
np.nonzero(arr > 5)  # (array([5, 6, 7, 8]),)

# where：返回满足条件的索引
np.where(arr > 5)  # (array([5, 6, 7, 8]),)

# 查找唯一值
arr_dup = np.array([1, 2, 2, 3, 3, 3, 4])
np.unique(arr_dup)  # [1, 2, 3, 4]
np.unique(arr_dup, return_counts=True)  # (array([1,2,3,4]), array([1,2,3,1]))
```

## 十、性能优化技巧

### 10.1 向量化运算

```python
import time

# 慢：Python 循环
arr = np.arange(1000000)
start = time.time()
result = []
for x in arr:
    result.append(x ** 2)
print(f"Python 循环: {time.time() - start:.4f}秒")

# 快：NumPy 向量化
start = time.time()
result = arr ** 2
print(f"NumPy 向量化: {time.time() - start:.6f}秒")
```

### 10.2 内存优化

```python
# 使用合适的数据类型
arr_int8 = np.array([1, 2, 3], dtype=np.int8)    # 1字节
arr_int64 = np.array([1, 2, 3], dtype=np.int64)  # 8字节

# 使用视图而不是副本
view = arr[::2]  # 视图，不占额外内存

# 原地操作
arr += 1  # 比 arr = arr + 1 更省内存

# 使用 out 参数
np.add(a, b, out=a)  # 结果写入 a，不创建新数组
```

### 10.3 避免不必要的复制

```python
# 好：使用视图
view = arr[1:100]

# 差：创建副本
copy = arr[1:100].copy()

# 检查是否共享内存
np.shares_memory(arr, view)  # True
np.shares_memory(arr, copy)  # False
```

## 十一、常见错误与注意事项

### 11.1 浮点数精度问题

```python
# 浮点数比较
a = np.array([0.1 + 0.2])
b = np.array([0.3])
print(a == b)  # [False]（精度问题）

# 正确的比较方式
np.allclose(a, b)  # True
np.isclose(a, b)   # [True]
```

### 11.2 广播陷阱

```python
# 形状不兼容会报错
a = np.ones((3, 4))
b = np.ones((3, 5))
# a + b  # ValueError!

# 检查是否可以广播
try:
    np.broadcast_shapes(a.shape, b.shape)
except ValueError as e:
    print(f"无法广播: {e}")
```

### 11.3 视图 vs 副本

```python
# 切片是视图
a = np.array([1, 2, 3, 4])
b = a[1:3]
b[0] = 999
print(a)  # [1, 999, 3, 4]（被修改！）

# 显式复制
c = a[1:3].copy()
c[0] = 888
print(a)  # [1, 999, 3, 4]（未被修改）
```

## 十二、实用技巧总结

### 12.1 快速生成测试数据

```python
# 随机整数矩阵
np.random.randint(0, 100, (10, 5))

# 正态分布数据
np.random.randn(100, 100)

# 随机打乱数组
arr = np.arange(10)
np.random.shuffle(arr)

# 随机抽样
np.random.choice(arr, size=5, replace=False)
```

### 12.2 数组保存与加载

```python
# 保存单个数组
arr = np.arange(10)
np.save('array.npy', arr)

# 加载数组
arr_loaded = np.load('array.npy')

# 保存多个数组
np.savez('arrays.npz', a=arr1, b=arr2)

# 加载多个数组
data = np.load('arrays.npz')
arr1 = data['a']
arr2 = data['b']

# 保存为文本文件
np.savetxt('array.txt', arr)
arr_loaded = np.loadtxt('array.txt')
```

### 12.3 常用技巧

```python
# 创建对角矩阵
np.diag([1, 2, 3, 4])

# 提取对角线
arr = np.arange(16).reshape(4, 4)
np.diag(arr)  # [0, 5, 10, 15]

# 生成网格
x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 3)
xx, yy = np.meshgrid(x, y)

# 数组比较
np.array_equal(arr1, arr2)  # 是否完全相同
np.allclose(arr1, arr2)     # 是否近似相同

# 条件统计
arr = np.array([1, 2, 3, 4, 5])
np.sum(arr > 2)  # 3（满足条件的元素数量）
```

## 参考

- [NumPy 官方文档](https://numpy.org/doc/)
- [NumPy API 参考](https://numpy.org/doc/stable/reference/)
- [巨硬的NumPy](https://github.com/datawhalechina/powerful-numpy)