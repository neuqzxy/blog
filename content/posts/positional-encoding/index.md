---
title: 位置编码
date: 2026-03-18 09:16:32
tags:
  - Math
  - ML
  - AI
  - LLM
categories:
  - ML
---
## 简介
Transformer 的 encoder 和 decoder 都要求在embedding之后、进入注意力层之前，加入位置编码，使得输入满足：

$$
\text{input} = \text{input\_embedding} + \text{input\_encoding}
$$

位置编码的演进有如下阶段：

### 整数标记

给第一个token标记1，给第二个token标记2...，以此类推，这种方法产生了以下几个主要问题：
  - 模型可能遇见比训练时所用的序列更长的序列。不利于模型的泛化。
  - 模型的位置表示是无界的。随着序列长度的增加，位置值会越来越大。
值越来越大会导致：
1. **尺度爆炸：** embedding一般会做归一化，数值范围较小，而巨大的位置信息会淹没词的语义信息
2. **梯度异常：** 巨大的 $Q \cdot K$ 经过 $\text{softmax}$ 后导致大部分概率被压缩到0，梯度消失
3. **无法表示相对位置：** 1 vs 2 和 100 vs 101 相对关系完全不同，难以学习

### 用[0,1]范围标记位置
将位置值的范围限制在[0, 1]之内，但当序列长度不同时，token间的相对距离不一样。

### 周期函数（sin）

通过一个 $\mathbb{R}^{1 \times d_{model}}$ 维向量来表示，向量的每一个维度代表某个精度下的值，如果位置向量当中的每一个元素都用一个sin函数来表示，则第t个token的位置向量可以表示为：
$$
\text{PE}_t = [sin(\frac{1}{2^0}t), sin(\frac{1}{2^1}t), ..., sin(\frac{1}{2^{d_{model} - 1}}t)]
$$

周期函数避免了梯度爆炸，并且在不同序列长度中，对应位置的值没有变化

不同的周期函数，可以捕获的位置信息也不同

```py
import math
import numpy as np

count = 10000
d_model = 3

def encode(pos,/,dim=d_model):
    return [math.sin(pos / 2**i) for i in range(dim)]

def encode_transformer(pos,/,dim=d_model):
    return [math.sin(pos / 10000 ** (2*i/dim)) for i in range(dim)]

encode_list = np.array([encode(i) for i in range(count)])
encode_list_transformer = np.array([encode_transformer(i) for i in range(count)])
```

![](/img/sin_pe.jpg)

对于 $PE_{t}^{(i)} = \sin(t / 2^i)$。当 $i=0$ 时，波长 $\lambda = 2\pi \cdot 2^0 \approx 6.28$，这意味着每隔约 6 个位置，正弦波就转了一圈。

Transformer 的频率分布： 分母是 $10000^{\frac{2i}{d_{model}}}$。$i=0$ 时，波长 $\lambda \approx 6.28$（快速震荡，负责局部信息）。$i=2$ 时，分母变为 $10000^{4/3} \approx 215443$。波长 $\lambda \approx 1,353,000$。

### 相对位置

{{< alert "file-lines" >}}
在 Transformer 的注意力机制中，位置信息的有用性最终体现在 $Q \cdot K^T$ 的计算结果中。如果位置编码能让这个结果只取决于 $t$ 和 $t+k$ 之间的相对距离 $k$，那么模型就具备了感知相对位置的能力。
{{< /alert >}}

普通的周期函数难以捕获相对位置之间的关联，我们希望不同位置之间能通过线性变换相互转换

$$
PE_{t+k}^{(i)} = M_k \cdot PE_{t}^{(i)}
$$

我们发现，通过sin和cos交替的形式，能对应上三角函数的加法公式：

$\sin(\omega_i (t+k)) = \sin(\omega_i t)\cos(\omega_i k) + \cos(\omega_i t)\sin(\omega_i k)$

$\cos(\omega_i (t+k)) = \cos(\omega_i t)\cos(\omega_i k) - \sin(\omega_i t)\sin(\omega_i k)$

这完全符合矩阵乘以向量的形式。我们可以构造一个旋转矩阵 $M_k$：

$$
\begin{bmatrix} \sin(\omega_i (t+k)) \\ \cos(\omega_i (t+k)) \end{bmatrix} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix} \begin{bmatrix} \sin(\omega_i t) \\ \cos(\omega_i t) \end{bmatrix}
$$

完整的矩阵 $M_k$ 的结构如下：
$$
M_k = \begin{bmatrix}
\color{pink}{\begin{matrix} \cos(\omega_0 k) & \sin(\omega_0 k) \\ -\sin(\omega_0 k) & \cos(\omega_0 k) \end{matrix}} & 0 & \dots & 0 \\
0 & \color{orange}{\begin{matrix} \cos(\omega_1 k) & \sin(\omega_1 k) \\ -\sin(\omega_1 k) & \cos(\omega_1 k) \end{matrix}} & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & \color{lightblue}{\begin{matrix} \cos(\omega_{m} k) & \sin(\omega_{m} k) \\ -\sin(\omega_{m} k) & \cos(\omega_{m} k) \end{matrix}}
\end{bmatrix}
$$

使用这样的方案，位置编码的频率就需要是 $\frac{d_{model}}{2}$ ，每个频率 $\omega_i$ 都会同时生成一个 $\sin$ 和一个 $\cos$，这样拼起来维度仍然是 $d_{model}$。

## 算法

### Sinusoidal 位置编码

在原始 Transformer 中，位置编码是直接加在 Embedding 上的。

$$
PE_{t}^{(i)} = \begin{cases}
sin(\frac{t}{10000^{2k / d_{model}}}) & i = 2k\\\\
cos(\frac{t}{10000^{2k / d_{model}}}) & i = 2k+1
\end{cases}
$$
其中 $i = [1, 2, ..., \frac{d_{model}}{2} - 1]$ ，为了简化证明，我们只看位置编码向量本身的内积 $PE_t \cdot PE_{t+k}$。假设 $PE_t$ 的第 $i$ 组（即两个维度）为：

$$
PE_{t}^{(i)} = \begin{bmatrix} \sin(\omega_i t) \\ \cos(\omega_i t) \end{bmatrix}
$$

计算位置 $t$ 和 $t+k$ 对应维度的内积：

$$
\begin{aligned}
PE_{t}^{(i)} \cdot PE_{t+k}^{(i)} &= \sin(\omega_i t)\sin(\omega_i(t+k)) + \cos(\omega_i t)\cos(\omega_i(t+k)) \\
&= \cos(\omega_i(t+k) - \omega_i t) \quad \text{（利用积化和差公式）} \\
&= \cos(\omega_i k)
\end{aligned}
$$

内积结果 $\cos(\omega_i k)$ 只与相对距离 $k$ 有关，与绝对位置 $t$ 无关。这在数学上证明了 Sinusoidal 编码通过三角函数的对称性，将绝对位置转换为了内积空间中的相对距离关系。

### RoPE (Rotary Positional Embedding)

RoPE 的设计更进一步。它不是将编码“加”上去，而是将 $Q$ 和 $K$ 向量在空间中进行旋转。

假设 $q$ 是位置 $m$ 的查询向量，$k$ 是位置 $n$ 的键向量（这里为了符合论文习惯，用 $m, n$ 表示位置）。RoPE 对它们的操作如下：

$$
\tilde{q}_m = R_{m,\theta} \cdot q
$$

$$
\tilde{k}_n = R_{n,\theta} \cdot k
$$

其中 $R_{m,\theta}$ 是我们之前讨论过的那个分块对角旋转矩阵：

$$
\begin{bmatrix} \cos(\omega_m \theta) & \sin(\omega_m \theta) \\ -\sin(\omega_m \theta) & \cos(\omega_m \theta) \end{bmatrix}
$$

在注意力机制计算点积时：

$$
\text{Score}(m, n) = \tilde{q}_m^T \tilde{k}_n = (R_{m,\theta} q)^T (R_{n,\theta} k) = q^T R_{m,\theta}^T R_{n,\theta} k
$$

由于旋转矩阵是正交矩阵，其转置等于逆矩阵（$R^T = R^{-1}$），且旋转矩阵具有可加性：$R_{m} \cdot R_{n} = R_{m+n}$。那么：

$$
R_{m,\theta}^T R_{n,\theta} = R_{-m,\theta} R_{n,\theta} = R_{n-m, \theta}
$$

带回点积公式：

$$
\text{Score}(m, n) = q^T \underbrace{R_{n-m, \theta}}_{\text{仅与相对位置有关}} k
$$

结论：RoPE 通过矩阵乘法，直接将相对位置信息 $n-m$ 注入到了点积结果中。相比 Sinusoidal 的“相加”，这种“旋转”操作保持了向量的模长不变，只改变相位，在深度网络中具有更好的数值稳定性。
