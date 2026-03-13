---
title: 激活函数
date: 2026-02-15 21:16:48
tags:
  - Math
  - ML
  - AI
categories:
  - ML
---
# 作用
增加非线性因素，从解析式角度来说，通过一系列线性项，去拟合一个非线性的函数，是不可能的。
线性函数满足：
- 可加性：$f(x + y) = f(x) + f(y)$
- 齐次性：$f(kx) = kf(x)$
如果是纯线性层累加，无论多少层，最终都相当于一层
$$
y = W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2) = W_{new}x + b_{new}
$$
这说明：**深度的增加对于纯线性模型没有意义**，它永远只能拟合超平面（高维空间里的 “平坦切片”），永远平坦、无弯曲、无褶皱。
$$
w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b = 0
$$
从矩阵角度看，$Wx + b$是仿射变换，只能对向量空间进行旋转、缩放、剪切、平移操作，无法变出曲面来。

而添加非线性项之后，能够拟合更为复杂的函数，根据**万能近似定理 (Universal Approximation Theorem)**，只要有至少一个非线性隐层，神经网络理论上可以拟合任何闭合区间内的连续函数。

有两种方案：
1. 添加高次项：
  - 维数灾难，即便是二次，额外新增的项也有 $ x_1^2, x_2^2, ..., x_n^2, x_1 x_2, x_1 x_3, ... $
  - 梯度爆炸，在反向传播时容易梯度爆炸
2. 添加激活函数：本质上是通过一个函数来扭曲坐标系。有很多不同的激活函数可选，下面详细介绍

# Sigmoid
![](/images/sigmoid.png)
Sigmoid 函数的图像看起来像一个 S 形曲线。函数表达式如下：
$
\sigma(z)=1/(1+e^{-z})
$
$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$

**缺点：**
- 两侧对比较小的值会压缩的更小，容易梯度消失
- Sigmoid 的导数最大值仅为 $0.25$（当 $z=0$ 时）。这意味着每多经过一层，梯度至少衰减为原来的 $1/4$，在深层网络中梯度会迅速趋近于 $0$，梯度消失！
- 不是 **zero-centered**（不以 0 为中心）

**zero-centered**
$\sigma(z) \ge 0$ 导致每一层的输入都是非负数，那么：
$$
\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial z} \cdot x_i
$$
任意 $x_i \ge 0$ 导致所有参数只取决于 $\frac{\partial L}{\partial z}$ 同号，之字形收敛导致很慢

# ReLU
![](/images/relu.png)

**缺点：**
- Dead ReLU 问题。当输入为负时，ReLU 完全失效，在正向传播过程中，这不是问题。有些区域很敏感，有些则不敏感。但是在反向传播过程中，如果输入负数，则梯度将完全为零，Sigmoid 函数和 tanh 函数也具有类似的问题，但是梯度趋近于 0，被称之为饱和 (Saturation)。Sigmoid 是两端饱和，而 ReLU 是左侧完全硬饱和（梯度直接归零）。
- 不是 **zero-centered**（不以 0 为中心）

# Leaky ReLU
![](/images/Leaky_ReLU.png)

与 ReLU 的不同之处在于负轴保留了非常小的常数leak，使得输入信息小于0时，信息没有完全丢掉，进行了相应的保留

# Tanh

![](/images/tanh.png)

tanh 激活函数的图像也是 S 形，表达式如下：

$
\tanh(x) = \frac{2}{1 + e^{-2x}} - 1
$

**优势：**
- sigmoid 输出都挤在 (0, 1)，数值小、偏软。tanh 输出在 (-1, 1)，动态范围更大
- 是 **zero-centered**

**劣势：**
- 慢
- 梯度消失

对神经网络来说，输出数值跨度越大，信号越强，梯度越明显。

# Softmax

Softmax 适用于多类分类问题的激活函数，在多类分类问题中，超过两个类标签则需要类成员关系。对于长度为 K 的任意实向量，Softmax 可以将其压缩为长度为 K，值在（0，1）范围内，并且向量中元素的总和为 1 的实向量。公式为：
$$
y_i = \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$
$$
\frac{\partial y_i}{\partial x_j} = \begin{cases}
y_i(1 - y_i) & i = j\\\\
-y_i y_j & i \neq j
\end{cases}
$$

Softmax 激活函数的主要缺点是对输入敏感，易受极端值/异常值主导：
- 输入过大：输出概率分布会极度尖锐，导致梯度消失，反向传播时梯度趋近于0
- 输入过小：输出分布趋于均匀，失去区分度
假设某次计算的注意力分数为[1000, 10, 5]，Softmax输出为：$ \text{softmax}([1000,10,5]) \approx [1,0,0] $
此时梯度几乎无法传播到非最大值位置，会出现梯度消失现象，导致模型不能更新。

**优化方案**
1. **Normalization (BN/LN)**：
在进入 Softmax 之前，通过 **Batch Norm** 或 **Layer Norm** 将神经元的输出重新拉回到均值为 0、方差为 1 的标准分布。这保证了输入给 Softmax 的值不会出现极端巨大的量级差异，从而让梯度能够健康地流动。
2. **Temperature Scaling**
在 Transformer 等模型中，注意力分数计算后会除以 $\sqrt{d_k}$。这本质上也是为了防止 Softmax 的输入过大，导致输出分布过于“尖锐”（极端的 0 和 1），从而避开梯度消失区。

# Swish (SiLU)
![](/images/silu.png)
Swish 是由 Google 在 2017 年提出的，后来在研究中发现 $\beta=1$ 时的 Swish（也称为 **SiLU**, Sigmoid Linear Unit）表现最为稳健。
**数学公式**

$$
Swish(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}
$$

通常在深度学习框架中，默认 $\beta = 1$。

**优势：**
- **自门控 (Self-Gated)：** 它的形式可以看作是 $x$ 乘以一个关于 $x$ 的门控值（Sigmoid）。当 $x$ 很大时，门控打开（接近1）；当 $x$ 为负且值较大时，门控关闭（接近0）。
- **非单调性 (Non-monotonicity)：** 与 ReLU 不同，Swish 在 $x < 0$ 的区域有一段平滑的“凹槽”。这意味着即使输入是微小的负值，信息也不会被完全截断，这有助于深层网络中的梯度流动。
- **平滑性：** 它是全域可微的，这使得优化器的表面更加平滑，有助于模型收敛。
- **“死区”问题：** 对于ReLU来说，任何 $x < 0$，$\frac{df}{dx} = 0$。对于 Swish 来说，只要 $x$ 不是负无穷，它的梯度就**永远不会绝对等于 0**。

$f'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \text{Swish}(x) + \sigma(x)(1 - \text{Swish}(x))$

# GLU (Gated Linear Unit)

GLU（门控线性单元）最早出现在卷积神经网络处理语言任务的研究中，它引入了更显式的“门”概念。
**数学公式**
对于输入变量 $x$，GLU 将其通过两个不同的线性变换（矩阵 $W$ 和 $V$），然后进行逐元素相乘：

$$
GLU(x, W, V, b, c) = \sigma(xW + b) \otimes (xV + c)
$$

其中 $\sigma$ 是 Sigmoid 函数，$\otimes$ 是逐元素乘积（Hadamard product）。

**优势：**
- **选择性信息传递：** $xV + c$ 是主要的信息载体，而 $\sigma(xW + b)$ 则充当“动态门控”。模型可以根据当前的输入内容，自主决定让多少信息通过。
- **梯度消失缓解：** 相比于纯粹的 Sigmoid 激活，GLU 的一部分是线性的（即 $xV+c$ 部分），这在很大程度上缓解了深层网络中的梯度消失问题，因为它提供了一条线性的路径。

$\frac{dy}{dx} = \sigma'(xW)W \cdot (xV) + \sigma(xW) \cdot V$

已知$\sigma’ < 0.25$，但后面这一项，如果门控打开（即 $\sigma(xW) \approx 1$），那么这一块的导数就近似等于 **$V$**。只要权重矩阵 $V$ 的谱半径（Spectral Radius）维持在 1 附近，梯度就可以几乎**毫无损耗地**流过这一层，而不会被 $\sigma'$ 的 0.25 魔咒强行截断。

# SwiGLU

SwiGLU 是由 Noam Shazeer 在 2020 年提出的（论文 *GLU Variants Improve Transformer*）。它是 Swish 和 GLU 的结合体，目前是 **Qwen**、**Llama 2/3、Gemma、Mistral** 等主流大模型的标配。

SwiGLU 是 GLU 的变体，它将 Sigmoid 替换为了 Swish，并通常去掉了偏置项：

$$
SwiGLU(x, W, V) = Swish_1(xW) \otimes (xV)
$$

在模型实现中，它通常用于 FFN（前馈网络）层。

**优势：**
- **性能之王：** 研究表明，SwiGLU 在几乎所有 Transformer 任务中都优于 ReLU、GELU 和原始 GLU。
- **更强的表达能力：** 它结合了 Swish 的非单调平滑特性和 GLU 的门控乘法结构。通过两个矩阵 $W$ 和 $V$ 的交互，模型能够学习到比单一激活函数更复杂的特征变换。
- **计算开销与增益：** 相比于 ReLU，SwiGLU 增加了参数量（需要两个权重矩阵），但它带来的模型精度提升通常远超其计算成本的增加，因此成为现代 LLM 的首选。

**用法介绍**
普通的 **FFN** 架构如下：
```
x (d_model)
   ↓ 线性层 W1 (d→d_ff)  【第1次变换：升维】
h (d_ff)
   ↓ 激活函数 ReLU/GELU  【无参数】
   ↓ 线性层 W2 (d_ff→d)  【第2次变换：降维】
out (d_model)
```
**SwiGLU** 架构如下：
```
                    x (d_model)
                   /           \
                  /             \
    线性层 W_gate (d→d_ff)   线性层 W_up (d→d_ff)
          ↓                       ↓
       SiLU(·)                    ·
          \                      /
           \                    /
            逐元素相乘 (⊙)
                     ↓
          线性层 W_down (d_ff→d)  【降维】
                     ↓
                    out
```
