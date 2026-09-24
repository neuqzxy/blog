---
title: Mahalanobis Distance
date: 2026-09-23 21:37:15
tags:
  - Math
  - ML
  - AI
categories:
  - Math
---
# Mahalanobis Distance

## 欧氏距离真的合理吗？

假设有二维数据：

$$
x =
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix},
\qquad
\mu =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
$$

先假设两个维度互相独立，并且：

$$
\sigma_1 = 10, \qquad \sigma_2 = 1
$$

也就是说：

- $x_1$ 本身波动非常大；
- $x_2$ 本身波动非常小。

考虑两个点：

$$
A = (10, 0), \qquad B = (0, 2)
$$

它们到均值的欧氏距离分别是：

$$
\|A - \mu\| = 10, \qquad \|B - \mu\| = 2
$$

如果按照欧氏距离（Euclidean distance）来判断，A 比 B 离均值远得多。

但从统计学角度看，这个结论其实很奇怪：

- **A**：$x_1 = 10$，而 $\sigma_1 = 10$，所以它只偏离了 $\dfrac{10}{10} = 1\sigma$；
- **B**：$x_2 = 2$，而 $\sigma_2 = 1$，所以它偏离了 $\dfrac{2}{1} = 2\sigma$。

因此，A 虽然欧氏距离更远，但从统计意义上说，B 反而更加“异常”。

欧氏距离把每个维度一视同仁，却忽略了不同维度本身的波动尺度。这正是 Mahalanobis distance 要解决的问题：先按各维度的标准差（更一般地，按协方差矩阵）把数据“拉回”同一尺度，再去度量距离。

## 一维 Gaussian 中的“Mahalanobis distance”

在一维 Gaussian 中：

$$
X \sim \mathcal N(\mu, \sigma^2)
$$

我们经常对数据做标准化：

$$
z = \frac{x - \mu}{\sigma}
$$

这个 $z$ 表示 $x$ 距离均值有多少个标准差。

例如，设 $\mu = 100$，$\sigma = 10$，那么 $x = 120$ 对应：

$$
z = \frac{120 - 100}{10} = 2
$$

也就是说，$x$ 距离均值 2 个标准差。

在一维情形下，$|z|$ 就是 Mahalanobis distance：它不再用原始单位衡量“远近”，而是用标准差作为尺子。

## 推广到二维

### 相互独立的两个维度

先从最简单的二维情况开始：两个维度互相独立。假设

$$
X =
\begin{bmatrix}
X_1 \\
X_2
\end{bmatrix},
\qquad
\mu =
\begin{bmatrix}
\mu_1 \\
\mu_2
\end{bmatrix},
\qquad
\Sigma =
\begin{bmatrix}
\sigma_1^2 & 0 \\
0 & \sigma_2^2
\end{bmatrix}
$$

由于 $\Sigma$ 是对角矩阵，它的逆只需把对角元素取倒数：

$$
\Sigma^{-1} =
\begin{bmatrix}
\dfrac{1}{\sigma_1^2} & 0 \\
0 & \dfrac{1}{\sigma_2^2}
\end{bmatrix}
$$

Mahalanobis distance 的平方定义为：

$$
D_M^2 = (x - \mu)^\top \Sigma^{-1} (x - \mu)
$$

代入上面的 $\Sigma^{-1}$，展开得到：

$$
D_M^2 = \frac{(x_1 - \mu_1)^2}{\sigma_1^2} + \frac{(x_2 - \mu_2)^2}{\sigma_2^2}
$$

也就是：

$$
\boxed{
D_M^2 =
\left( \frac{x_1 - \mu_1}{\sigma_1} \right)^2
+
\left( \frac{x_2 - \mu_2}{\sigma_2} \right)^2
}
$$

这个公式非常值得理解。它实际上是在做两步：

1. 先把每个维度减去均值、再除以自己的标准差；
2. 再对标准化后的结果计算普通欧氏距离。

也就是说，先把 $x$ 变成

$$
\tilde x =
\begin{bmatrix}
\dfrac{x_1 - \mu_1}{\sigma_1} \\[2ex]
\dfrac{x_2 - \mu_2}{\sigma_2}
\end{bmatrix}
$$

然后

$$
D_M = \|\tilde x\|
$$

回到开头的例子（$\mu = 0$，$\sigma_1 = 10$，$\sigma_2 = 1$）：

$$
D_M(A) = \sqrt{\left( \frac{10}{10} \right)^2 + 0^2} = 1,
\qquad
D_M(B) = \sqrt{0^2 + \left( \frac{2}{1} \right)^2} = 2
$$

在 Mahalanobis distance 下，B 比 A 更远，这和前面的统计直觉一致。

所以，Mahalanobis distance 做的第一件事就是：**把不同尺度的坐标重新调整到相同的统计尺度。**

### 维度具有相关性

真正有意思的是维度之间存在相关性的情形。此时协方差矩阵不再是对角的：

$$
\Sigma =
\begin{bmatrix}
\sigma_1^2 & \rho \sigma_1 \sigma_2 \\
\rho \sigma_1 \sigma_2 & \sigma_2^2
\end{bmatrix}
$$

其中 $\rho$ 是 $X_1$ 与 $X_2$ 的相关系数。例如：

$$
\Sigma =
\begin{bmatrix}
1 & 0.9 \\
0.9 & 1
\end{bmatrix}
$$

这意味着 $X_1$ 与 $X_2$ 高度正相关。于是二维 Gaussian 的等密度线不再是圆，而是沿着 $x_1 = x_2$ 方向拉长的椭圆。

这时候，只对每个坐标单独做

$$
\frac{x_i - \mu_i}{\sigma_i}
$$

已经不够了。原因在于：不同**方向**上的“自然波动范围”已经不一样了。沿 $x_1 = x_2$ 方向，数据本来就散得很开；而沿与之垂直的 $x_1 = -x_2$ 方向，数据挤得很紧。逐坐标标准化只能照顾到坐标轴方向，照顾不到这些斜向的差异。

### 等距离线是椭圆

这引出 Mahalanobis distance 最重要的几何事实之一。固定一个距离 $c > 0$：

$$
D_M(x) = c
$$

也就是：

$$
(x - \mu)^\top \Sigma^{-1} (x - \mu) = c^2
$$

记 $\delta_1 = x_1 - \mu_1$，$\delta_2 = x_2 - \mu_2$，把 $\Sigma^{-1}$ 写成对称矩阵 $\begin{bmatrix} a & b \\ b & d \end{bmatrix}$，上式展开就是一个二次型：

$$
a \delta_1^2 + 2 b\, \delta_1 \delta_2 + d\, \delta_2^2 = c^2
$$

当 $\Sigma$ 正定时，$\Sigma^{-1}$ 也正定，这个二次曲线是一个椭圆。所以：

$$
\boxed{
D_M(x) = c \iff x \text{ 位于以 } \mu \text{ 为中心的椭圆上}
}
$$

其中特别重要的是 $c = 1$ 的情形：

$$
D_M(x) = 1
$$

它对应的椭圆称为 **1-sigma 椭圆**，是一维中“距离均值 1 个标准差”在二维中的自然推广。

### 特征分解：椭圆的方向与尺度

现在做一个非常重要的数学分解。由于协方差矩阵是对称正定矩阵，可以进行特征分解：

$$
\boxed{
\Sigma = U \Lambda U^\top
}
$$

其中：

- $U = \begin{bmatrix} u_1 & u_2 \end{bmatrix}$ 是特征向量矩阵，$u_1, u_2$ 是两个互相正交的单位向量，因此 $U$ 是正交矩阵（$U^\top U = I$）；
- $\Lambda = \begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix}$ 是特征值矩阵，且 $\lambda_1, \lambda_2 > 0$。

例如：

$$
\Sigma =
\begin{bmatrix}
5 & 4 \\
4 & 5
\end{bmatrix}
= U
\begin{bmatrix}
9 & 0 \\
0 & 1
\end{bmatrix}
U^\top,
\qquad
u_1 = \frac{1}{\sqrt 2}
\begin{bmatrix}
1 \\
1
\end{bmatrix},
\quad
u_2 = \frac{1}{\sqrt 2}
\begin{bmatrix}
1 \\
-1
\end{bmatrix}
$$

这里 $\lambda_1 = 9$，$\lambda_2 = 1$，对应的标准差为：

$$
\sqrt{\lambda_1} = 3, \qquad \sqrt{\lambda_2} = 1
$$

所以：

- 沿 $u_1$（即 $x_1 = x_2$）方向，数据的尺度是 3；
- 沿 $u_2$（即 $x_1 = -x_2$）方向，数据的尺度是 1。

而 $U$ 决定这两个方向在原坐标系中指向哪里。总结起来：

$$
\boxed{
\text{特征向量} = \text{椭圆主轴的方向}
}
\qquad
\boxed{
\sqrt{\text{特征值}} = \text{沿各主轴的标准差尺度}
}
$$

具体地，$D_M(x) = c$ 的椭圆沿 $u_i$ 方向的半轴长为 $c \sqrt{\lambda_i}$。上例中的 1-sigma 椭圆，就是沿 $x_1 = x_2$ 方向半轴长 3、沿 $x_1 = -x_2$ 方向半轴长 1 的椭圆。
