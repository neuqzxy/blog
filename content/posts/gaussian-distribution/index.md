---
title: "深度剖析高斯分布：从 1D Gaussian 到 3D Gaussian Splatting"
date: 2026-09-23 21:55:43
tags:
  - Math
  - ML
  - AI
categories:
  - ML
---
# 深度剖析高斯分布：从 1D Gaussian 到 3D Gaussian Splatting

## 0. 引言：为什么高斯分布如此特殊？

### 0.1 高斯分布无处不在

高斯分布在概率教材里出现得很早，早到容易让人把它当成一条方便画出来的钟形曲线。真正值得停下来的，是它后来出现的场合。测量、滤波、随机过程、生成模型和实时渲染，几乎不共享研究对象，却一次次选中同一种分布。

最早的场合是测量误差。高斯关心的问题很具体：对同一个量做多次观测，什么样的误差律会让算术平均成为最可信的估计。答案就是今天这条曲线。热噪声、传感器读数、天文观测里的残差，很多都可以用它近似。更根本的理由来自中心极限定理。大量彼此独立、方差有限的随机因素加在一起，在相当温和的条件下，总和会趋向高斯分布。世界上不少噪声并不是某一种机制单独生成的，而是许多小扰动叠出来的；叠到最后，形状就成了高斯。

工程里，人们开始主动选用它。经典的 Kalman Filter 假定过程噪声和观测噪声都是高斯的，于是滤波器只要维护一个均值和一个协方差，就能把新的观测递推进去。Gaussian Process 把这件事做到函数上：任意有限个位置的函数值放在一起，联合分布仍是高斯，一条曲线的不确定性由均值函数和协方差核决定。GMM 则换了一个用法，用许多个高斯去拼更复杂的密度，聚类和密度估计里一直能见到它。

到了深度学习，它几乎成了默认的随机性来源。VAE 通常把隐变量的先验取成标准正态，编码器给出的近似后验也是一个高斯。两个高斯之间的 KL 散度有闭式，从中采样也只是对标准正态样本做一次平移和缩放。Diffusion Model 更直接：正向过程不断往数据里加入高斯噪声，直到分布接近标准正态，模型再学习把这些噪声逐步去掉。

三维重建里也能见到它。3D Gaussian Splatting 不用三角网格，也不用神经网络隐式场，而是用大量各向异性的三维高斯表示一个场景。每个高斯带着自己的位置、形状和外观，投影到屏幕上之后可以直接光栅化。

测量里的残差、滤波器里的状态、生成模型里的噪声，以及屏幕上被直接画出来的三维高斯，共享的不是问题，而是同一个选择。全文只追问一件事：为什么大家如此喜欢 Gaussian？

答案先不一次说完。这里只放下后面会反复用到的几条线索。一个高斯分布，用均值和协方差就能完整描述。对它做线性或仿射变换，结果仍是高斯。一组变量的联合分布若是高斯，丢掉其中一部分，剩下的边缘分布仍是高斯；已知另一部分之后，条件分布也仍是高斯。把两个高斯密度乘在一起，结果仍然和一个高斯紧密相关。到了高维，它还有一套干净的几何图像。再往实现看，它还可以被高效地投影和渲染。

## Part I：从一维高斯真正理解 Gaussian

### 1. 一维高斯分布

#### 1.1 从公式开始

若随机变量 $X$ 服从均值为 $\mu$、方差为 $\sigma^2$ 的高斯分布，记作：

$$
X \sim \mathcal N(\mu, \sigma^2)
$$

其概率密度函数为：

$$
f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2 \sigma^2} \right)
$$

- **位置参数 $\mu$**（均值 / 数学期望）：决定曲线的对称中心，改变 $\mu$ 会让曲线整体平移。
- **尺度参数 $\sigma$**（标准差）：$\sigma$ 越小，曲线越窄、峰顶越高，数据越向均值集中；$\sigma$ 越大，曲线越宽，数据越分散。

拆开看，这个公式由两部分拼接而成：

1. **形状项**：$\exp\left( -\dfrac{(x - \mu)^2}{2 \sigma^2} \right)$
   - 分子 $(x - \mu)^2$ 与分母 $2 \sigma^2$ 的量纲都是 $[x]^2$，相除后量纲抵消，所以指数的参数是一个无量纲量。这是必须的：$\exp$、$\ln$、$\sin$ 这类超越函数的参数不能带单位，写 $\exp(5\,\text{m})$ 没有意义。
   - 它决定了整条钟形曲线的起伏：中心最高，向两侧按距离的平方迅速衰减。
2. **归一化常数**：$\dfrac{1}{\sqrt{2 \pi \sigma^2}}$
   - 任何 PDF 在全域上的积分都必须为 1，而形状项的全域积分是

     $$
     \int_{-\infty}^{+\infty} \exp\left( -\frac{(x - \mu)^2}{2 \sigma^2} \right) dx = \sqrt{2 \pi \sigma^2}
     $$

     所以需要除以这个值。

特别地，当 $\mu = 0$、$\sigma = 1$ 时，称为**标准正态分布**，记作 $Z \sim \mathcal N(0, 1)$。任意高斯分布都可以通过标准化转为标准正态分布：

$$
Z = \frac{X - \mu}{\sigma}
$$

#### 1.2 为什么指数中是平方？——距离与几何本质

指数上的核心项是：

$$
\frac{(x - \mu)^2}{\sigma^2} = \left( \frac{x - \mu}{\sigma} \right)^2
$$

其中 $|x - \mu|$ 是样本点 $x$ 到中心 $\mu$ 在实数轴上的欧氏距离，再除以 $\sigma$，就得到“以标准差为单位”的距离。这个标准化后的距离正是一维情形下的 [Mahalanobis distance](/posts/mahalanobis-distance/)，记作 $D_M(x) = \dfrac{|x - \mu|}{\sigma}$。

于是高斯分布的本质可以浓缩为一个极简的形式：

$$
p(x) \propto \exp\left( -\frac{1}{2} D_M(x)^2 \right)
$$

当 $\sigma = 1$ 时，$D_M$ 就退化为普通的欧氏距离。

### 2. 经验法则

> 正态分布几乎所有样本都集中在 $\mu \pm 3\sigma$ 之内，超出 $3\sigma$ 的事件非常罕见，这就是 **3σ 准则**。

经验法则是对任意正态分布 $X \sim \mathcal N(\mu, \sigma^2)$ 概率质量最直观的刻画，包含以下三种情况：

1. **$\mu \pm 1\sigma$**：约 **68.27%** 的数据落在区间内；
2. **$\mu \pm 2\sigma$**：约 **95.45%** 的数据落在区间内；
3. **$\mu \pm 3\sigma$**：约 **99.73%** 的数据落在区间内。

## Part II：从 1D 到 2D——协方差第一次出现

### 3. 二维 Gaussian

#### 3.1 从公式开始

定义二维随机向量：

$$
\mathbf x =
\begin{bmatrix}
x \\
y
\end{bmatrix}
$$

二维 Gaussian 记作：

$$
\mathbf x \sim \mathcal N(\boldsymbol \mu, \Sigma)
$$

其中均值向量为：

$$
\boldsymbol \mu =
\begin{bmatrix}
\mu_x \\
\mu_y
\end{bmatrix}
$$

$\Sigma$ 是一个 $2 \times 2$ 的协方差矩阵。要理解它，先要理解 covariance。

#### 3.2 从 variance 到 covariance

考虑两个随机变量 $X$ 和 $Y$。如果某个样本满足 $X > \mu_x$ 且 $Y > \mu_y$，那么 $(X - \mu_x)(Y - \mu_y) > 0$。当 $X$ 和 $Y$ **经常同时偏离各自均值的同一方向**时，这个乘积的平均值就会是正的。据此定义 covariance：

$$
\operatorname{Cov}(X, Y) = \mathbb E\left[ (X - \mu_x)(Y - \mu_y) \right]
$$

它的符号含义如下：

- $\operatorname{Cov}(X, Y) > 0$：$X$ 高于自己的均值时，$Y$ 往往也高于自己的均值；$X$ 低于均值时，$Y$ 往往也低于均值。
- $\operatorname{Cov}(X, Y) < 0$：$X$ 高于自己的均值时，$Y$ 倾向于低于自己的均值；反之亦然。
- $\operatorname{Cov}(X, Y) = 0$：从 covariance 的角度看，$X$ 和 $Y$ 没有线性相关性。

特别地，$\operatorname{Cov}(X, X) = \operatorname{Var}(X) = \sigma_x^2$，所以 variance 是 covariance 的特例。

但这里一定要强调一个非常重要的数学细节：

$$
\operatorname{Cov}(X, Y) = 0 \not\Rightarrow X, Y \text{ 独立}
$$

也就是说，零协方差只说明没有线性相关性，不代表两个随机变量完全没有关系。例如 $X \sim \mathcal N(0, 1)$、$Y = X^2$，此时 $\operatorname{Cov}(X, Y) = \mathbb E[X^3] = 0$，但 $Y$ 完全由 $X$ 决定。

不过对于**联合 Gaussian** 而言，有一个非常特殊的性质，这也是 Gaussian 与众不同的地方之一：

$$
(X, Y) \text{ 服从联合 Gaussian 且 } \operatorname{Cov}(X, Y) = 0 \Rightarrow X, Y \text{ 独立}
$$

注意前提是**联合** Gaussian：仅仅 $X$ 和 $Y$ 各自服从 Gaussian 还不够。

#### 3.3 协方差矩阵（covariance matrix）

把所有两两之间的 covariance 排成矩阵，就得到协方差矩阵：

$$
\Sigma =
\begin{bmatrix}
\sigma_x^2 & \sigma_{xy} \\
\sigma_{xy} & \sigma_y^2
\end{bmatrix},
\qquad
\sigma_{xy} = \operatorname{Cov}(X, Y)
$$

它是对称矩阵；在非退化的情形下还是正定矩阵，因此可逆。

二维 Gaussian 的概率密度可以写成：

$$
p(\mathbf x) = \frac{1}{2 \pi |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf x - \boldsymbol \mu)^\top \Sigma^{-1} (\mathbf x - \boldsymbol \mu) \right)
$$

其中 $|\Sigma|$ 是 $\Sigma$ 的行列式。前面的系数只负责归一化，真正决定 Gaussian 长什么样的是指数里的二次型：

$$
(\mathbf x - \boldsymbol \mu)^\top \Sigma^{-1} (\mathbf x - \boldsymbol \mu)
$$

#### 3.4 Mahalanobis distance 与椭圆

上面这个二次型正是 Mahalanobis distance 的平方：

$$
D_M(\mathbf x)^2 = (\mathbf x - \boldsymbol \mu)^\top \Sigma^{-1} (\mathbf x - \boldsymbol \mu)
$$

于是二维 Gaussian 与一维情形有完全相同的结构：

$$
p(\mathbf x) \propto \exp\left( -\frac{1}{2} D_M(\mathbf x)^2 \right)
$$

密度只取决于 $D_M$，所以等密度线就是 $D_M(\mathbf x) = c$ 的曲线。当 $\Sigma$ 正定时，它是以 $\boldsymbol \mu$ 为中心的椭圆。

对 $\Sigma$ 做特征分解：

$$
\Sigma = Q \Lambda Q^\top,
\qquad
Q = \begin{bmatrix} q_1 & q_2 \end{bmatrix},
\qquad
\Lambda = \begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix}
$$

- 特征向量 $q_1, q_2$ 给出椭圆主轴的方向；
- $\sqrt{\lambda_1}, \sqrt{\lambda_2}$ 给出沿各主轴的标准差尺度，$D_M = c$ 的椭圆沿 $q_i$ 方向的半轴长为 $c \sqrt{\lambda_i}$。

完整的推导和例子见 [Mahalanobis distance](/posts/mahalanobis-distance/)。

## Part III：3D Gaussian——从椭圆到椭球

### 4. 3D Gaussian 的几何意义

定义三维随机向量：

$$
\mathbf x =
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix},
\qquad
\mathbf x \sim \mathcal N(\boldsymbol \mu, \Sigma)
$$

其中 $\Sigma \in \mathbb R^{3 \times 3}$，展开为：

$$
\Sigma =
\begin{bmatrix}
\sigma_x^2 & \sigma_{xy} & \sigma_{xz} \\
\sigma_{xy} & \sigma_y^2 & \sigma_{yz} \\
\sigma_{xz} & \sigma_{yz} & \sigma_z^2
\end{bmatrix}
$$

讨论等概率密度曲面：

$$
(\mathbf x - \boldsymbol \mu)^\top \Sigma^{-1} (\mathbf x - \boldsymbol \mu) = c^2
$$

它不再是椭圆，而是一个以 $\boldsymbol \mu$ 为中心的**椭球**（ellipsoid）。

和二维一样，通过特征分解：

$$
\Sigma = Q \Lambda Q^\top
$$

可以得到：

- 三个互相正交的主方向 $q_1, q_2, q_3$，即椭球主轴的方向；
- 对应的尺度 $\sqrt{\lambda_1}, \sqrt{\lambda_2}, \sqrt{\lambda_3}$，即沿各主轴的标准差。

这就是 3D Gaussian Splatting 中 Gaussian primitive 的核心几何结构。3DGS 的原始论文把协方差写成 $\Sigma = R S S^\top R^\top$，其中 $R$ 是旋转矩阵，$S$ 是对角缩放矩阵。这和 $Q \Lambda Q^\top$ 是同一件事：$R$ 对应 $Q$，$S$ 的对角元素对应 $\sqrt{\lambda_i}$。这种参数化在优化过程中能始终保证 $\Sigma$ 是合法的（对称半正定）协方差矩阵。
