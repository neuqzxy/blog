---
title: 从 L1 Loss 到 SSIM：深入理解图像重建损失函数及其在 3DGS 中的应用
date: 2026-10-11 21:18:26
tags:
  - Math
  - ML
  - AI
  - LLM
categories:
  - ML
---
在学习 3D Gaussian Splatting（3DGS） 时，我们会遇到这样一个损失函数：

$$
\mathcal L=(1-\lambda)\mathcal L_{L1}+\lambda\mathcal L_{D\text{-}SSIM}
$$

其中：
- $L1$ Loss：衡量渲染图像与真实图像之间的像素差异。
- SSIM（Structural Similarity Index）：衡量两张图像的结构相似程度。
- D-SSIM：由 SSIM 构造的损失，表示结构不相似程度。
- $\lambda$：控制两种损失的相对权重。

原始 3DGS 使用 $lambda=0.2$，也就是：

$$
\mathcal L=0.8\mathcal L_{L1}+0.2(1-\operatorname{SSIM})
$$

## 一、L1 Loss：最直接的像素误差

假设真实图像和生成图像分别为：

$$
I,\hat I\in\mathbb R^{H\times W\times3}
$$

其中：
- $H$：图像高度。
- $W$：图像宽度。
- $3$：RGB 三个颜色通道。
L1 Loss 定义为：

$$
\mathcal L_{L1} = \frac{1}{3HW} \sum_{h=1}^{H} \sum_{w=1}^{W} \sum_{c=1}^{3} \left| I_{hwc}-\hat I_{hwc} \right|
$$

本质上就是逐个计算对应像素的绝对误差，然后求平均。

### 1.1 L1 只关注像素误差，不直接考虑局部结构

考虑真实图像的一个局部区域：

$$
X = \begin{bmatrix} 0.2 & 0.2 \\ 0.8 & 0.8 \end{bmatrix}
$$

这是一个很简单的图像结构：上半部分较暗，下半部分较亮。现在有两种预测结果。第一种：

$$
Y_1 = \begin{bmatrix} 0.4 & 0.4 \\ 1.0 & 1.0 \end{bmatrix}
$$

第二种：

$$
Y_2 = \begin{bmatrix} 0.2 & 0.6 \\ 0.4 & 0.8 \end{bmatrix}
$$

对于 $Y_1$，它保留了原图的结构：
- 上半部分仍然比下半部分暗。
- 明暗变化的位置没有改变。
- 只是整体亮度增加了。

对于 $Y_2$，情况不同：
- 原本统一的上半部分出现了明显变化。
- 原本统一的下半部分也出现了变化。
- 图像的局部明暗结构发生了改变。

直觉上，$Y_1$ 更接近原始图像的结构。现在计算 L1。对于第一种预测：

$$
L1(X, Y_1) = \frac{0.2 + 0.2 + 0.2 + 0.2}{4} = 0.2
$$

对于第二种预测：

$$
L1(X, Y_2) = \frac{0 + 0.4 + 0.4 + 0}{4} = 0.2
$$

两种预测的局部结构明显不同，但 L1 给出了完全相同的结果：

$$
L1(X, Y_1) = L1(X, Y_2)
$$

这是因为 L1 计算的是 $\sum_i \left| x_i - y_i \right|$。它不直接分析这些误差在空间上如何组织，也不会把一组像素作为整体来判断其纹理或局部对比度是否保留。L1 可以准确描述像素绝对误差的大小，却不能单独表达我们希望优先保留哪些局部视觉结构。这正是 SSIM 要补充的信息。

## 二、SSIM：结构相似度究竟是什么

SSIM 的全称是：Structural Similarity Index Measure 即结构相似性指标。它来自一篇经典论文：Image Quality Assessment: From Error Visibility to Structural Similarity，Wang et al., 2004。

SSIM 的核心思想是：比较两张图像时，除了直接比较像素值，还应该比较局部亮度、对比度和结构信息。实际上，SSIM 只需要几个非常基础的统计量。它们是：均值、方差和协方差。

$$
\mu_X,\mu_Y,\sigma_X^2,\sigma_Y^2,\sigma_{XY}
$$

首先，取出两张图像中对应的两个局部区域：

$$
X=(x_1,x_2,\ldots,x_n)
$$

$$
Y=(y_1,y_2,\ldots,y_n)
$$

这里的 $X$ 和 $Y$ 可以理解成两张图片中对应位置的两个图像 Patch。例如两个 $11 \times 11$ 的局部窗口。SSIM 分别比较它们的亮度、对比度和结构。

### 2.1 亮度相似度：Luminance

我们希望当 $\mu_X = \mu_Y$ 时，$l(X, Y) = 1$；两个均值差异越大，相似度越低。SSIM 使用：

$$
l(X, Y) = \frac{2\mu_X\mu_Y + C_1}{\mu_X^2 + \mu_Y^2 + C_1}
$$

其中 $C_1$ 是一个很小的正数，用来提高数值稳定性。

理解这个公式之前，需要先了解韦伯定律（Weber's Law）。人类的视觉系统对亮度的感知是相对的，而不是绝对的。

- 场景 A（暗处）：如果在一个全黑的房间里（平均亮度为 10），突然亮起一盏 20 亮度的灯，你会觉得“刺眼”，亮度变化极其显著。绝对误差是 $20 - 10 = 10$。
- 场景 B（亮处）：如果在一个阳光明媚的操场上（平均亮度为 200），再多打一盏亮度为 10 的灯（变成 210），你肉眼根本察觉不到变化。绝对误差依然是 $210 - 200 = 10$。

如果只看绝对误差，这两种情况的惩罚是一样的。这不符合人类的视觉感受，所以需要归一化：把绝对误差除以当前环境的总亮度基数，得到一个相对误差。

回到公式，先看：

$$
\frac{2ab}{a^2 + b^2}
$$

因为 $a^2 + b^2 - 2ab = (a - b)^2$，所以：

$$
\frac{2ab}{a^2 + b^2} = 1 - \frac{(a - b)^2}{a^2 + b^2}
$$

也就是说：

$$
l(X, Y) = 1 - \frac{(\mu_X - \mu_Y)^2}{\mu_X^2 + \mu_Y^2 + C_1}
$$

这个形式更容易理解。它本质上是在减去归一化的亮度误差。用 1 减去这个误差，$\mu_X$ 和 $\mu_Y$ 越相似，$l$ 越大，符合亮度相似度的定义。

### 2.2 对比度相似度：Contrast

亮度只能告诉我们一个区域整体有多亮，不能告诉我们一个区域的明暗变化程度。例如：

$$
X = (0.5, 0.5, 0.5, 0.5)
$$

$$
Y = (0.1, 0.9, 0.1, 0.9)
$$

它们的均值都是 $\mu_X = \mu_Y = 0.5$，但：
- $X$ 完全没有明暗变化。
- $Y$ 有很强的明暗变化。

也就是说，它们的局部对比度不同。衡量这种变化用的是方差：

$$
\sigma_X^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu_X)^2
$$

$$
\sigma_Y^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - \mu_Y)^2
$$

方差越大，说明像素围绕均值的变化越剧烈。于是可以像亮度相似度那样定义：

$$
c(X, Y) = \frac{2\sigma_X\sigma_Y + C_2}{\sigma_X^2 + \sigma_Y^2 + C_2}
$$

其中 $C_2$ 同样是一个小的正数。

两个图像区域即使有相同的均值和标准差，结构也不一定相同。例如：

$$
X = (0, 0, 1, 1)
$$

$$
Y = (1, 1, 0, 0)
$$

两者均值相同，$\mu_X = \mu_Y = 0.5$；方差也相同，$\sigma_X^2 = \sigma_Y^2 = 0.25$。因此 $l(X, Y) = 1$，$c(X, Y) = 1$。但是两个区域的明暗排列完全相反，所以还需要第三个指标。

### 2.3 结构相似度：Structure

协方差描述两个变量偏离各自均值时的共同变化趋势，这是理解 SSIM 的关键。

情况一：两个区域具有相同的明暗变化趋势。例如：

$$
X = (0.2, 0.4, 0.6, 0.8)
$$

$$
Y = (0.3, 0.5, 0.7, 0.9)
$$

它们虽然存在亮度差异，但是：
- $X$ 增大时，$Y$ 也增大。
- $X$ 减小时，$Y$ 也减小。

因此 $\sigma_{XY} > 0$，两个区域具有相似的局部变化趋势。

情况二：两个区域的明暗变化趋势相反。例如：

$$
X = (0.2, 0.4, 0.6, 0.8)
$$

$$
Y = (0.8, 0.6, 0.4, 0.2)
$$

此时：
- $X$ 增大，$Y$ 减小。
- $X$ 减小，$Y$ 增大。

因此 $\sigma_{XY} < 0$，两个区域的变化趋势相反。

统计学中，皮尔逊相关系数定义为：

$$
\rho_{XY} = \frac{\sigma_{XY}}{\sigma_X \sigma_Y}
$$

它描述两个变量之间的线性相关程度。$\rho_{XY} = 1$ 表示两者完全正线性相关，$\rho_{XY} = -1$ 表示两者完全负线性相关，$\rho_{XY} = 0$ 表示两者没有线性相关性。

SSIM 使用了与相关系数非常接近的思想。它的结构相似度定义为：

$$
s(X, Y) = \frac{\sigma_{XY} + C_3}{\sigma_X \sigma_Y + C_3}
$$

当两个区域具有相同的局部变化趋势时，结构相似度会比较高；当两者呈现相反的变化趋势时，结构相似度就会降低。

实际上，SSIM 并没有真正理解图像中的物体结构。它不知道图像里有什么物体，也不知道某条边缘属于人脸还是建筑。它衡量的是对应局部区域内，像素围绕均值的变化模式是否相似。

### 2.4 组合得到完整的 SSIM 公式

现在已经构造了三个指标。亮度：

$$
l(X, Y) = \frac{2\mu_X\mu_Y + C_1}{\mu_X^2 + \mu_Y^2 + C_1}
$$

对比度：

$$
c(X, Y) = \frac{2\sigma_X\sigma_Y + C_2}{\sigma_X^2 + \sigma_Y^2 + C_2}
$$

结构：

$$
s(X, Y) = \frac{\sigma_{XY} + C_3}{\sigma_X\sigma_Y + C_3}
$$

SSIM 将它们组合起来：

$$
\operatorname{SSIM}(X, Y) = l(X, Y)^\alpha \, c(X, Y)^\beta \, s(X, Y)^\gamma
$$

其中 $\alpha, \beta, \gamma$ 是控制各部分相对重要程度的指数参数。在经典的简化形式中，取 $\alpha = \beta = \gamma = 1$，并令 $C_3 = C_2 / 2$，于是 $\operatorname{SSIM} = l \cdot c \cdot s$。

将对比度与结构项相乘：

$$
c \cdot s = \frac{2\sigma_X\sigma_Y + C_2}{\sigma_X^2 + \sigma_Y^2 + C_2} \cdot \frac{\sigma_{XY} + C_2 / 2}{\sigma_X\sigma_Y + C_2 / 2}
$$

注意 $2\sigma_X\sigma_Y + C_2 = 2\left(\sigma_X\sigma_Y + \dfrac{C_2}{2}\right)$，因此可以约去相应的公共因子，得到：

$$
c \cdot s = \frac{2\sigma_{XY} + C_2}{\sigma_X^2 + \sigma_Y^2 + C_2}
$$

最终：

$$
\operatorname{SSIM}(X, Y) = \frac{(2\mu_X\mu_Y + C_1)(2\sigma_{XY} + C_2)}{(\mu_X^2 + \mu_Y^2 + C_1)(\sigma_X^2 + \sigma_Y^2 + C_2)}
$$

其中 $C_1,C_2$ 保证了常见数值情况下分母不会因为零方差等问题直接变成零。这就是最常见的 SSIM 公式。整个公式只依赖五个统计量：

$$
\mu_X, \mu_Y, \sigma_X^2, \sigma_Y^2, \sigma_{XY}
$$

## 三、实际图像中的 SSIM 如何计算？

一张真实图像可能是 $1920 \times 1080 \times 3$。标准 SSIM 用局部滑动窗口（Sliding Window）来计算。经典实现使用一个 $11 \times 11$、标准差为 $1.5$ 的高斯加权窗口，因为窗口中心通常应该比窗口边缘更重要。它满足：
- 最中心的像素（第 6 行第 6 列）权重最大。
- 越往边缘走，权重越小。
- 这 121 个格子的权重总和为 1。

二维高斯权重为：

$$
w(u, v) \propto \exp\left( -\frac{u^2 + v^2}{2\sigma^2} \right)
$$

归一化后满足 $\sum_{u,v} w(u, v) = 1$。把这份权重记作 $G_\sigma$，$*$ 表示卷积。加权均值就是：

$$
\mu_X = G_\sigma * X
$$

方差是偏差平方的加权平均。把它展开：

$$
\sigma_X^2 = G_\sigma * (X - \mu_X)^2 = G_\sigma * (X^2 - 2X\mu_X + \mu_X^2)
$$

局部窗口里 $\mu_X$ 已经是常数，可以提到卷积外面。权重和为 1，常数卷积后仍是它自己。代入 $G_\sigma * X = \mu_X$：

$$
\sigma_X^2 = G_\sigma * (X^2) - 2\mu_X (G_\sigma * X) + \mu_X^2 = G_\sigma * (X^2) - \mu_X^2
$$

协方差从偏差的乘积出发，同样展开：

$$
\sigma_{XY} = G_\sigma * \left((X - \mu_X)(Y - \mu_Y)\right) = G_\sigma * (XY - X\mu_Y - Y\mu_X + \mu_X\mu_Y)
$$

提出常数 $\mu_X$、$\mu_Y$，并代入 $G_\sigma * X = \mu_X$、$G_\sigma * Y = \mu_Y$：

$$
\sigma_{XY} = G_\sigma * (XY) - \mu_Y\mu_X - \mu_X\mu_Y + \mu_X\mu_Y = G_\sigma * (XY) - \mu_X\mu_Y
$$

因此，SSIM 需要的统计量都可以通过卷积实现：

$$
\mu_X = G_\sigma * X
$$

$$
\sigma_X^2 = G_\sigma * (X^2) - \mu_X^2
$$

$$
\sigma_{XY} = G_\sigma * (XY) - \mu_X \mu_Y
$$

其中 $X^2$ 和 $XY$ 都是逐像素运算。SSIM 可以通过几次普通的二维卷积，以及逐元素的加减乘除完成计算，整个过程适合 GPU 并行执行。最后对 SSIM Map 取平均：

$$
\operatorname{SSIM}(I, \hat{I}) = \frac{1}{3HW} \sum_{c,h,w} \operatorname{SSIM}_c(h, w)
$$
