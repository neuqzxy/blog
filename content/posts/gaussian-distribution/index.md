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

注意前提是**联合** Gaussian：仅仅 $X$ 和 $Y$ 各自服从 Gaussian 还不够。什么样的随机向量才算联合 Gaussian、为什么这个前提不能省，放到 3.4 讨论。

#### 3.3 协方差矩阵与二维 Gaussian 的密度

3.1 只写下了记号 $\mathcal N(\boldsymbol \mu, \Sigma)$，还欠两样东西：

1. $\Sigma$ 到底是什么；
2. 二维 Gaussian 的**概率密度函数** $p(\mathbf x)$ 长什么样。

第二样是一维 1.1 节里 $f(x)$ 的对应物。一维时，有了 $f(x)$ 才能算区间概率、画出钟形曲线；二维时，$p(\mathbf x)$ 告诉我们平面上每一点附近样本出现得有多密集：落在点 $\mathbf x$ 附近一小块面积 $\Delta S$ 里的概率约为 $p(\mathbf x) \, \Delta S$。

**第一步：** 定义 $\Sigma$ 

描述一个二维分布的"散开方式"，需要三个数：$x$ 方向的方差 $\sigma_x^2$，$y$ 方向的方差 $\sigma_y^2$，以及两者如何联动的协方差 $\sigma_{xy} = \operatorname{Cov}(X, Y)$。把它们排成一个矩阵，就是**协方差矩阵**：

$$
\Sigma =
\begin{bmatrix}
\sigma_x^2 & \sigma_{xy} \\
\sigma_{xy} & \sigma_y^2
\end{bmatrix}
$$

第 $i$ 行第 $j$ 列放的是第 $i$ 个分量与第 $j$ 个分量的协方差，对角线上是各自的方差。

这个矩阵还可以写成一个紧凑的期望式。记 $\mathbf v = \mathbf x - \boldsymbol \mu$，它是一个 $2 \times 1$ 的列向量；$\mathbf v \mathbf v^\top$ 是列向量乘行向量，结果是 $2 \times 2$ 矩阵，称为**外积**：

$$
(\mathbf x - \boldsymbol \mu)(\mathbf x - \boldsymbol \mu)^\top =
\begin{bmatrix}
(x - \mu_x)^2 & (x - \mu_x)(y - \mu_y) \\
(y - \mu_y)(x - \mu_x) & (y - \mu_y)^2
\end{bmatrix}
$$

对每个元素取期望，对角线变成方差，非对角线变成协方差，正好就是 $\Sigma$：

$$
\Sigma = \mathbb E\left[ (\mathbf x - \boldsymbol \mu)(\mathbf x - \boldsymbol \mu)^\top \right]
$$

**$\Sigma$ 什么时候可逆。** 先看两条性质。

**对称**：$\operatorname{Cov}(X, Y) = \operatorname{Cov}(Y, X)$，所以 $\Sigma^\top = \Sigma$。

**半正定**：任取一个常向量 $\mathbf a$，$\mathbf a^\top \mathbf x$ 就是 $\mathbf x$ 在方向 $\mathbf a$ 上的投影，它是一个标量随机变量。计算 $\mathbf a^\top \Sigma \mathbf a$：

$$
\mathbf a^\top \Sigma \mathbf a = \mathbb E\left[ \mathbf a^\top (\mathbf x - \boldsymbol \mu) \, (\mathbf x - \boldsymbol \mu)^\top \mathbf a \right] = \mathbb E\left[ \left( \mathbf a^\top (\mathbf x - \boldsymbol \mu) \right)^2 \right] = \operatorname{Var}(\mathbf a^\top \mathbf x) \ge 0
$$

第一个等号把常向量 $\mathbf a$ 移进期望里。第二个等号用到：$\mathbf a^\top (\mathbf x - \boldsymbol \mu)$ 和 $(\mathbf x - \boldsymbol \mu)^\top \mathbf a$ 都是标量，并且互为转置，所以是同一个数，乘起来就是平方。最后一项就是投影的方差，方差不会是负数。

所以 $\mathbf a^\top \Sigma \mathbf a$ 有一个很直观的含义：**$\mathbf x$ 沿方向 $\mathbf a$ 的方差**。它等于 0，意味着 $\mathbf a^\top \mathbf x$ 恒为常数，所有样本都挤在一条直线上，二维分布实际上退化成了一维，这种情况称为**退化**。

排除退化，对任意非零 $\mathbf a$ 都有 $\mathbf a^\top \Sigma \mathbf a > 0$，即 $\Sigma$ **正定**。正定矩阵一定可逆：如果存在非零向量 $\mathbf v$ 使 $\Sigma \mathbf v = \mathbf 0$，那么 $\mathbf v^\top \Sigma \mathbf v = 0$，与正定矛盾。下文都假设非退化，于是 $\Sigma^{-1}$ 存在，行列式 $|\Sigma| > 0$。

**第二步：** 推出二维 Gaussian 的密度

我们知道：**任何一维 Gaussian 都是标准正态平移、缩放得到的**，$X = \mu + \sigma Z$。可以推广到多维：从最简单的二维 Gaussian 出发，用线性变换把它变成一般的二维 Gaussian，同时追踪密度怎么跟着变。

**对于一维 Gaussian。** 设 $Z \sim \mathcal N(0, 1)$，密度为 $\varphi(z) = \frac{1}{\sqrt{2 \pi}} e^{-z^2/2}$，令 $X = \mu + \sigma Z$（$\sigma > 0$）。这个变换把 $Z$ 轴上的点一一对应到 $X$ 轴上：$X$ 轴上的点 $x$，对应 $Z$ 轴上的点

$$
z = \frac{x - \mu}{\sigma}
$$

现在取 $X$ 轴上从 $x$ 开始、长度为 $\Delta x$ 的小区间 $[x, x + \Delta x]$。它的左端点对应 $z$，右端点对应 $\frac{x + \Delta x - \mu}{\sigma} = z + \frac{\Delta x}{\sigma}$；又因为 $\sigma > 0$，对应关系不改变先后顺序。所以事件 $\{x \le X \le x + \Delta x\}$ 和事件 $\{z \le Z \le z + \Delta x / \sigma\}$ 是同一个事件，概率严格相等。

我们可得到区间密度：

$$
f(x) \, \Delta x \approx \varphi(z) \, \frac{\Delta x}{\sigma}
$$

两边除以 $\Delta x$，再令 $\Delta x \to 0$，约等号就成了等号：

$$
f(x) = \frac{1}{\sigma} \, \varphi\left( \frac{x - \mu}{\sigma} \right) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2 \sigma^2} \right)
$$

这正是 1.1 节的公式。可以看到两处变化各有来源：

- **指数里**：把 $z$ 换成 $(x - \mu)/\sigma$，这是"点与点的对应"；
- **系数里**：多出 $1/\sigma$，因为长度被拉长了 $\sigma$ 倍，同样多的概率摊在更长的区间上，密度就要变稀。

二维的推导完全是这两件事的矩阵版本。

**起点：标准二维 Gaussian。** 取两个相互独立的标准正态 $z_1, z_2$，组成 $\mathbf z = (z_1, z_2)^\top$。独立意味着联合密度等于各自密度的乘积：

$$
p_{\mathbf z}(\mathbf z)
= \frac{1}{\sqrt{2 \pi}} e^{-z_1^2/2} \cdot \frac{1}{\sqrt{2 \pi}} e^{-z_2^2/2}
= \frac{1}{2 \pi} \exp\left( -\frac{1}{2} (z_1^2 + z_2^2) \right)
= \frac{1}{2 \pi} \exp\left( -\frac{1}{2} \mathbf z^\top \mathbf z \right)
$$

它的均值是 $\mathbf 0$。协方差矩阵是单位阵 $I$：对角线是各自的方差 1，非对角线是协方差 0（独立）。即 $\mathbb E[\mathbf z \mathbf z^\top] = I$。它的等密度线 $z_1^2 + z_2^2 = c^2$ 是圆。

**变换：把圆变成一般的样子。** 用一个 $2 \times 2$ 可逆矩阵 $A$ 和一个向量 $\boldsymbol \mu$ 做仿射变换：

$$
\mathbf x = \boldsymbol \mu + A \mathbf z
$$

几何上，$A$ 负责旋转和沿不同方向拉伸，把圆拉成椭圆；$\boldsymbol \mu$ 负责把中心平移过去。要求 $A$ 可逆，是因为不可逆的 $A$ 会把整个平面压到一条直线上，正是上面说的退化情形。

变换后的均值和协方差分别是：

$$
\mathbb E[\mathbf x] = \boldsymbol \mu + A \, \mathbb E[\mathbf z] = \boldsymbol \mu
$$

$$
\mathbb E\left[ (\mathbf x - \boldsymbol \mu)(\mathbf x - \boldsymbol \mu)^\top \right]
= \mathbb E\left[ (A \mathbf z)(A \mathbf z)^\top \right]
= \mathbb E\left[ A \mathbf z \mathbf z^\top A^\top \right]
= A \, \mathbb E[\mathbf z \mathbf z^\top] \, A^\top
= A A^\top
$$

反过来，任给一个正定的 $\Sigma$，也总能找到 $A$ 使 $\Sigma = A A^\top$。例如用特征分解 $\Sigma = Q \Lambda Q^\top$，取 $A = Q \Lambda^{1/2}$。因此**任何非退化的二维 Gaussian 都可以由标准二维 Gaussian 这样变换得到**，只要算出这一种构造的密度，就得到了所有二维 Gaussian 的密度。

接下来**追踪密度**， 和一维一样，有两件事要处理。

**点与点的对应。** 平面上的点 $\mathbf x$ 来自唯一的 $\mathbf z = A^{-1}(\mathbf x - \boldsymbol \mu)$。代入 $p_{\mathbf z}$ 的指数：

$$
\mathbf z^\top \mathbf z
= \left( A^{-1} (\mathbf x - \boldsymbol \mu) \right)^\top A^{-1} (\mathbf x - \boldsymbol \mu)
= (\mathbf x - \boldsymbol \mu)^\top (A^{-1})^\top A^{-1} (\mathbf x - \boldsymbol \mu)
$$

而 $(A^{-1})^\top A^{-1} = (A A^\top)^{-1} = \Sigma^{-1}$（矩阵乘积求逆要倒序：$(A A^\top)^{-1} = (A^\top)^{-1} A^{-1}$，且转置与求逆可以交换次序）。于是

$$
\mathbf z^\top \mathbf z = (\mathbf x - \boldsymbol \mu)^\top \Sigma^{-1} (\mathbf x - \boldsymbol \mu)
$$

这就是一维里 $z^2 = (x - \mu)^2 / \sigma^2$ 的二维版本，$\Sigma^{-1}$ 扮演了 $1/\sigma^2$ 的角色。

**面积的伸缩。** 一维里长度被拉长 $\sigma$ 倍；二维里，线性变换 $A$ 会把任意一小块区域的面积放大 $|\det A|$ 倍（例如单位正方形被映成面积为 $|\det A|$ 的平行四边形）。$\mathbf z$ 平面上面积为 $\Delta S$ 的小块，变到 $\mathbf x$ 平面上面积成了 $|\det A| \, \Delta S$，而其中包含的概率不变。和一维一样，小块里的概率约等于"密度 × 面积"：

$$
p(\mathbf x) \cdot |\det A| \, \Delta S \approx p_{\mathbf z}(\mathbf z) \cdot \Delta S
$$

两边除以 $\Delta S$，再令小块收缩到一点，得到

$$
p(\mathbf x) = \frac{p_{\mathbf z}(\mathbf z)}{|\det A|}
$$

再用行列式的性质 $\det(A A^\top) = \det A \cdot \det A^\top = (\det A)^2$，得到 $|\det A| = \sqrt{|\Sigma|} = |\Sigma|^{1/2}$。这对应一维里的 $1/\sigma$。

**合起来。** 把两件事代入 $p_{\mathbf z}$，就得到二维 Gaussian 的概率密度：

$$
p(\mathbf x) = \frac{1}{2 \pi |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf x - \boldsymbol \mu)^\top \Sigma^{-1} (\mathbf x - \boldsymbol \mu) \right)
$$

和一维逐项对照：

| 一维 | 二维 | 含义 |
| --- | --- | --- |
| $\sigma^2$ | $\Sigma$ | 散开程度 |
| $\sigma$ | $A$（$\Sigma = A A^\top$） | 从标准正态出发的缩放 |
| $(x - \mu)^2 / \sigma^2$ | $(\mathbf x - \boldsymbol \mu)^\top \Sigma^{-1} (\mathbf x - \boldsymbol \mu)$ | 点与点的对应 |
| $1/\sigma$ | $1/\lvert\Sigma\rvert^{1/2}$ | 长度 / 面积伸缩带来的归一化 |

还有一个值得注意的细节：最终公式里只出现 $\boldsymbol \mu$ 和 $\Sigma$，$A$ 本身消失了。满足 $A A^\top = \Sigma$ 的 $A$ 并不唯一（特征分解给出的 $Q \Lambda^{1/2}$、Cholesky 分解给出的下三角阵都可以），但只要 $\Sigma$ 相同，得到的就是同一个分布。这正是引言里那句话的来源：**均值和协方差就能完整描述一个 Gaussian**。

最后，前面的系数只负责让总概率为 1，真正决定密度在平面上如何分布、等密度线长什么样的，是指数里的二次型：

$$
(\mathbf x - \boldsymbol \mu)^\top \Sigma^{-1} (\mathbf x - \boldsymbol \mu)
$$

它是一维"以标准差为单位的距离平方"的推广，下一节就来看它的几何含义。

#### 3.4 什么样的随机向量是联合 Gaussian

前面一直在说"联合 Gaussian"，现在给出它的判定。下面的说法对任意维度 $\mathbf x \in \mathbb R^n$ 都成立，二维只是 $n = 2$ 的情形。以下三条等价：

1. **线性组合判定**：对任意常向量 $\mathbf a$，标量 $\mathbf a^\top \mathbf x$ 都服从一维 Gaussian（方差为 0 时退化为常数，也算在内）。
2. **构造判定**：存在向量 $\boldsymbol \mu$ 和矩阵 $A$，使得 $\mathbf x = \boldsymbol \mu + A \mathbf z$，其中 $\mathbf z \sim \mathcal N(\mathbf 0, I)$。此时 $\Sigma = A A^\top$。
3. **密度判定**（仅当 $\Sigma$ 正定）：$\mathbf x$ 的密度为

   $$
   p(\mathbf x) = \frac{1}{(2 \pi)^{n/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf x - \boldsymbol \mu)^\top \Sigma^{-1} (\mathbf x - \boldsymbol \mu) \right)
   $$

第一条是定义，也是最常用的检验方式：只检查每个分量各自是不是 Gaussian 不够，要检查所有方向上的投影。第二条说明联合 Gaussian 就是"独立标准正态经过一次仿射变换"。所以若干独立的 Gaussian 拼在一起，或者再做线性变换，得到的都是联合 Gaussian。3.3 已经从第二条出发推出了第三条的密度（$n = 2$ 时归一化系数就是 $2 \pi |\Sigma|^{1/2}$）。

**反例：边缘都是 Gaussian，联合却不是。** 取 $X \sim \mathcal N(0, 1)$，再取一个与 $X$ 独立、等概率取 $\pm 1$ 的随机符号 $S$，令 $Y = S X$。由对称性 $Y \sim \mathcal N(0, 1)$，并且

$$
\operatorname{Cov}(X, Y) = \mathbb E[S X^2] = \mathbb E[S] \, \mathbb E[X^2] = 0
$$

但 $|Y| = |X|$，两者显然不独立。问题出在线性组合上：$X + Y = (1 + S) X$ 有 $1/2$ 的概率恰好等于 0，不可能是 Gaussian，所以 $(X, Y)$ 不是联合 Gaussian，3.2 的结论不适用。

**为什么联合 Gaussian 下零协方差意味着独立。** 若 $\sigma_{xy} = 0$，则 $\Sigma = \operatorname{diag}(\sigma_x^2, \sigma_y^2)$，$\Sigma^{-1} = \operatorname{diag}(1/\sigma_x^2, 1/\sigma_y^2)$，$|\Sigma| = \sigma_x^2 \sigma_y^2$，代入密度得到

$$
p(x, y) = \frac{1}{\sqrt{2 \pi \sigma_x^2}} e^{-\frac{(x - \mu_x)^2}{2 \sigma_x^2}} \cdot \frac{1}{\sqrt{2 \pi \sigma_y^2}} e^{-\frac{(y - \mu_y)^2}{2 \sigma_y^2}} = p(x) \, p(y)
$$

联合密度等于边缘密度的乘积，这正是独立的定义。高维时同理：$\Sigma$ 是分块对角阵，对应的各组分量就相互独立。

#### 3.5 Mahalanobis distance 与椭圆

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
