---
title: 深度剖析高斯分布：从 1D Gaussian 到 3D Gaussian Splatting
date: 2026-09-23 21:55:43
tags:
  - Math
  - ML
  - AI
categories:
  - ML
---
## 引言：为什么高斯分布如此特殊？

高斯分布在概率教材里出现得很早，早到容易让人把它当成一条方便画出来的钟形曲线。真正值得停下来的，是它后来出现的场合。测量、滤波、随机过程、生成模型和实时渲染，几乎不共享研究对象，却一次次选中同一种分布。

最早的场合是测量误差。高斯关心的问题很具体：对同一个量做多次观测，什么样的误差律会让算术平均成为最可信的估计。答案就是今天这条曲线。热噪声、传感器读数、天文观测里的残差，很多都可以用它近似。更根本的理由来自中心极限定理。大量彼此独立、方差有限的随机因素加在一起，在相当温和的条件下，总和会趋向高斯分布。世界上不少噪声并不是某一种机制单独生成的，而是许多小扰动叠出来的；叠到最后，形状就成了高斯。

工程里，人们开始主动选用它。经典的 Kalman Filter 假定过程噪声和观测噪声都是高斯的，于是滤波器只要维护一个均值和一个协方差，就能把新的观测递推进去。Gaussian Process 把这件事做到函数上：任意有限个位置的函数值放在一起，联合分布仍是高斯，一条曲线的不确定性由均值函数和协方差核决定。GMM 则换了一个用法，用许多个高斯去拼更复杂的密度，聚类和密度估计里一直能见到它。

到了深度学习，它几乎成了默认的随机性来源。VAE 通常把隐变量的先验取成标准正态，编码器给出的近似后验也是一个高斯。两个高斯之间的 KL 散度有闭式，从中采样也只是对标准正态样本做一次平移和缩放。Diffusion Model 更直接：正向过程不断往数据里加入高斯噪声，直到分布接近标准正态，模型再学习把这些噪声逐步去掉。

三维重建里也能见到它。3D Gaussian Splatting 不用三角网格，也不用神经网络隐式场，而是用大量各向异性的三维高斯表示一个场景。每个高斯带着自己的位置、形状和外观，投影到屏幕上之后可以直接光栅化。

测量里的残差、滤波器里的状态、生成模型里的噪声，以及屏幕上被直接画出来的三维高斯，共享的不是问题，而是同一个选择。全文只追问一件事：为什么大家如此喜欢 Gaussian？

答案先不一次说完。这里只放下后面会反复用到的几条线索。一个高斯分布，用均值和协方差就能完整描述。对它做线性或仿射变换，结果仍是高斯。一组变量的联合分布若是高斯，丢掉其中一部分，剩下的边缘分布仍是高斯；已知另一部分之后，条件分布也仍是高斯。把两个高斯密度乘在一起，结果仍然和一个高斯紧密相关。到了高维，它还有一套干净的几何图像。再往实现看，它还可以被高效地投影和渲染。

## 一、一维高斯分布

### 1.1 从公式开始

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

### 1.2 为什么指数中是平方？——距离与几何本质

指数上的核心项是：

$$
\frac{(x - \mu)^2}{\sigma^2} = \left( \frac{x - \mu}{\sigma} \right)^2
$$

其中 $|x - \mu|$ 是样本点 $x$ 到中心 $\mu$ 在实数轴上的欧氏距离，再除以 $\sigma$，就得到"以标准差为单位"的距离。这个标准化后的距离正是一维情形下的 [Mahalanobis distance](/posts/mahalanobis-distance/)，记作 $D_M(x) = \dfrac{|x - \mu|}{\sigma}$。

于是高斯分布的本质可以浓缩为一个极简的形式：

$$
p(x) \propto \exp\left( -\frac{1}{2} D_M(x)^2 \right)
$$

当 $\sigma = 1$ 时，$D_M$ 就退化为普通的欧氏距离。

## 二、经验法则

> 正态分布几乎所有样本都集中在 $\mu \pm 3\sigma$ 之内，超出 $3\sigma$ 的事件非常罕见，这就是 **3σ 准则**。

经验法则是对任意正态分布 $X \sim \mathcal N(\mu, \sigma^2)$ 概率质量最直观的刻画，包含以下三种情况：

1. **$\mu \pm 1\sigma$**：约 **68.27%** 的数据落在区间内；
2. **$\mu \pm 2\sigma$**：约 **95.45%** 的数据落在区间内；
3. **$\mu \pm 3\sigma$**：约 **99.73%** 的数据落在区间内。

## 三、二维 Gaussian：协方差第一次出现

### 3.1 从公式开始

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

### 3.2 从 variance 到 covariance

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

### 3.3 协方差矩阵与二维 Gaussian 的密度

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

### 3.4 什么样的随机向量是联合 Gaussian

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

### 3.5 Mahalanobis distance 与椭圆

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

## 四、3D Gaussian：从椭圆到椭球

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

三维的概率密度可以直接由 3.4 的 $n$ 维公式取 $n = 3$ 得到：

$$
p(\mathbf x) = \frac{1}{(2 \pi)^{3/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf x - \boldsymbol \mu)^\top \Sigma^{-1} (\mathbf x - \boldsymbol \mu) \right)
$$

和一维、二维相比，变化的只有维度、均值向量和协方差矩阵：指数仍然是 Mahalanobis distance 的平方；归一化系数里的 $|\Sigma|^{1/2}$，从二维时的面积伸缩变成了三维时的体积伸缩。

讨论三维 Gaussian 的等密度曲面：

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

这就是 3D Gaussian Splatting 中每个 Gaussian 基元的几何结构：一个有中心、有朝向、三个方向长短不一的椭球。3DGS 论文没有直接用特征分解的记号，而是把协方差写成 $\Sigma = R S S^\top R^\top$。5.2 节会说明这两种写法是同一件事，以及 3DGS 为什么选择后者。

### 4.1 三维的"3σ 椭球"并不包含 99.73% 的概率

用椭球画出一个 Gaussian 的"范围"时，很自然会把第二章的经验法则搬过来，以为 $D_M \le 3$ 的椭球包含 99.73% 的概率。这在三维里并不成立。

为了算出椭球内部的概率，借用 3.3 的构造：任何非退化的 Gaussian 都可以写成 $\mathbf x = \boldsymbol \mu + A \mathbf z$，其中 $\Sigma = A A^\top$，$\mathbf z$ 由相互独立的标准正态组成（三维时有三个分量）。3.3 已经推出，点与点的对应满足

$$
D_M(\mathbf x)^2 = (\mathbf x - \boldsymbol \mu)^\top \Sigma^{-1} (\mathbf x - \boldsymbol \mu) = \mathbf z^\top \mathbf z = z_1^2 + z_2^2 + z_3^2
$$

$n$ 个独立标准正态的平方和服从自由度为 $n$ 的卡方分布 $\chi^2_n$。所以椭球 $D_M(\mathbf x) \le c$ 内部的概率就是 $P(\chi^2_n \le c^2)$。把一维、二维、三维放在一起对比：

| $c$ | 一维 | 二维 | 三维 |
| --- | --- | --- | --- |
| 1 | 68.27% | 39.35% | 19.87% |
| 2 | 95.45% | 86.47% | 73.85% |
| 3 | 99.73% | 98.89% | 97.07% |

维度越高，同样的 $c$ 包住的概率越少：每多一个维度，样本就多一个可以偏离中心的方向。不过，沿任意一个方向单独看，投影仍是一维 Gaussian，仍然遵循 1σ、2σ、3σ 的规律。

这个细节在 3DGS 中直接用得上。数学上 Gaussian 的尾部无限延伸，但实现时必须给每个 Gaussian 划定一个有限的影响范围。官方实现在屏幕上取 $3 \sqrt{\lambda_{\max}}$ 作为半径，其中 $\lambda_{\max}$ 是二维屏幕协方差（6.4 节）的较大特征值。由上表，二维 3σ 椭圆之外只剩约 1.1% 的概率，而这个半径对应的圆还把椭圆整个包在里面。这里的 $c$ 是一个截断阈值，而不是一个固定的概率百分比。

## 五、从概率密度到 3DGS 的 Gaussian 基元

前四章讲的是 Gaussian 本身，从这一章开始进入 3DGS。3DGS 要做的是：用大量 3D Gaussian 表示一个场景，再把它们画成图像。第五到第七章只讲其中和 Gaussian 数学直接相关的四件事：

1. 3DGS 用的 Gaussian 和概率密度有什么不同（5.1 节）；
2. 协方差怎样参数化，才能在训练中始终合法（5.2 节）；
3. 三维椭球怎样投影成屏幕上的二维椭圆（第六章）；
4. 屏幕上的椭圆怎样变成像素的颜色（第七章）。

前面讨论的 $\mathcal N(\boldsymbol \mu, \Sigma)$ 是**概率分布**，描述"一个随机点会落在哪里"。3DGS 里的 Gaussian 则是场景中一个看得见的小椭球，称为基元（primitive），描述"空间中这一块有多少东西"。两者共用均值和协方差的几何，但用途不同，不能直接混为一谈。

### 5.1 为什么 3DGS 使用不带归一化系数的 Gaussian

把三维密度拆成两部分：

$$
p(\mathbf x) = \underbrace{\frac{1}{(2 \pi)^{3/2} |\Sigma|^{1/2}}}_{\text{归一化系数}} \cdot \underbrace{\exp\left( -\frac{1}{2} (\mathbf x - \boldsymbol \mu)^\top \Sigma^{-1} (\mathbf x - \boldsymbol \mu) \right)}_{G(\mathbf x)}
$$

1.1 节说过，归一化系数只负责让总概率为 1，决定形状的是后面的指数部分。3DGS 只使用后一部分，记作

$$
G(\mathbf x) = \exp\left( -\frac{1}{2} (\mathbf x - \boldsymbol \mu)^\top \Sigma^{-1} (\mathbf x - \boldsymbol \mu) \right)
$$

它有三个性质：

- 中心由 $\boldsymbol \mu$ 决定，且 $G(\boldsymbol \mu) = 1$；
- 形状和朝向由 $\Sigma$ 决定，按 Mahalanobis distance 衡量，离中心越远，值越小；
- 取值在 0 到 1 之间，不要求积分为 1。

为什么要去掉归一化系数？看看协方差变大时会发生什么。对概率密度来说，$|\Sigma|$ 变大，归一化系数变小，峰值随之降低：同样多的概率摊到更大的体积上，自然变稀。对 $G(\mathbf x)$ 来说，无论 $\Sigma$ 多大，中心值始终是 1。

这正是渲染需要的。一个 Gaussian 在画面上覆盖多大范围、它有多不透明，应该能分别调节。如果用概率密度，把椭球放大就会自动让它变淡，两件事被绑在了一起。3DGS 让 $\Sigma$ 只管范围，再单独学习一个不透明度参数（7.1 节）来管浓淡。

所以要注意：$G(\mathbf x)$ 不是概率，只是一个 0 到 1 之间的空间权重。

### 5.2 协方差为什么写成旋转 $R$ 与尺度 $S$

训练 3DGS 时，每个 Gaussian 的协方差都要靠梯度下降来调整。问题在于：一个 $3 \times 3$ 矩阵要成为合法的协方差，必须对称且正定（3.3 节），而梯度下降每一步只是把各个元素加减一点，并不保证这两条性质。直接优化矩阵元素，几步之后就可能得到一个不合法的"协方差"，$\Sigma^{-1}$ 和 $G(\mathbf x)$ 也就无从谈起。

3DGS 的办法是换一组参数：

$$
\Sigma = R S S^\top R^\top,
\qquad
S = \operatorname{diag}(s_x, s_y, s_z),
\qquad
R^\top R = I
$$

其中 $R$ 是旋转矩阵，$S$ 是对角缩放矩阵。由于 $S$ 是对角阵，$S S^\top = \operatorname{diag}(s_x^2, s_y^2, s_z^2)$。

这个式子可以从 3.3 的构造直接读出来。3.3 说明，任何 Gaussian 都可以写成 $\mathbf x = \boldsymbol \mu + A \mathbf z$，协方差为 $\Sigma = A A^\top$。取 $A = R S$：

$$
\mathbf x = \boldsymbol \mu + R S \mathbf z
$$

它对应的几何过程是：

1. 标准 Gaussian $\mathbf z$ 的等密度面是球面；
2. 乘以 $S$，沿三个坐标轴分别拉伸，得到与坐标轴对齐的椭球；
3. 乘以 $R$，把椭球旋转到想要的朝向；
4. 加上 $\boldsymbol \mu$，把它平移到场景中的位置。

协方差就是 $A A^\top = (R S)(R S)^\top = R S S^\top R^\top$。

这组参数有两个好处：

- **自动合法**：只要每个 $s_i \ne 0$，$\Sigma$ 就一定对称正定。对称是因为 $\Sigma^\top = \Sigma$；正定是因为对任意非零向量 $\mathbf a$，$\mathbf a^\top \Sigma \mathbf a = \| S^\top R^\top \mathbf a \|^2 > 0$（$R$、$S$ 都可逆，$S^\top R^\top \mathbf a$ 不会是零向量）。
- **含义直观**：$s_i$ 管椭球三个方向的长短，$R$ 管朝向。

实现上，旋转用归一化的四元数表示（四个数，归一化后恰好对应一个三维旋转），尺度写成 $s_i = \exp(\ell_i)$ 以保证为正。训练时优化的是四元数和 $\ell_i$，而不是矩阵元素。

和第四章的特征分解对照：$\Sigma = R S S^\top R^\top$ 与 $\Sigma = Q \Lambda Q^\top$ 形式完全相同，$R$ 对应 $Q$（主轴方向），$s_i^2$ 对应特征值 $\lambda_i$。**3DGS 没有发明新的 Gaussian，只是换了一组更方便优化的参数来描述同一个椭球。**

## 六、三维 Gaussian 如何变成屏幕上的二维 Gaussian

场景里的 Gaussian 是三维椭球，图像却是二维像素网格。要把它画出来，先得知道它投影到屏幕上是什么样子：中心落在哪个像素，椭圆朝哪个方向、有多大。一个 3D Gaussian 投影到屏幕上形成的这个二维椭圆，称为 **splat**（原意是"泼溅出的一团痕迹"），3D Gaussian Splatting 的名字就来源于此。

投影分两步：先从世界坐标转到相机坐标（6.2 节），再从相机坐标透视投影到像素坐标（6.3、6.4 节）。两步都要用到同一个工具：Gaussian 经过线性变换后，均值和协方差怎么变（6.1 节）。

### 6.1 线性变换后的 Gaussian

3.3 推导过一个特殊情形：标准 Gaussian 经过可逆方阵 $A$ 变换后，协方差为 $A A^\top$。投影需要更一般的情形：起点是任意 Gaussian，变换矩阵也可以不是方阵（比如把三维映射到二维）。

设 $\mathbf x \sim \mathcal N(\boldsymbol \mu, \Sigma)$，令

$$
\mathbf y = M \mathbf x + \mathbf b
$$

其中 $M$ 是 $m \times n$ 矩阵，$\mathbf b$ 是常向量。

**均值**：期望是线性的，所以 $\mathbb E[\mathbf y] = M \boldsymbol \mu + \mathbf b$。

**协方差**：由 $\mathbf y - \mathbb E[\mathbf y] = M (\mathbf x - \boldsymbol \mu)$，

$$
\operatorname{Cov}(\mathbf y)
= \mathbb E\left[ M (\mathbf x - \boldsymbol \mu) (\mathbf x - \boldsymbol \mu)^\top M^\top \right]
= M \, \mathbb E\left[ (\mathbf x - \boldsymbol \mu) (\mathbf x - \boldsymbol \mu)^\top \right] M^\top
= M \Sigma M^\top
$$

推导步骤和 3.3 完全相同：$(M \mathbf v)^\top = \mathbf v^\top M^\top$，常数矩阵可以移到期望外面。当 $\mathbf x$ 是标准 Gaussian（$\Sigma = I$）时，结果就退回 3.3 的 $A A^\top$。

**$\mathbf y$ 仍然是 Gaussian**：任取常向量 $\mathbf a$，

$$
\mathbf a^\top \mathbf y = (M^\top \mathbf a)^\top \mathbf x + \mathbf a^\top \mathbf b
$$

这是 $\mathbf x$ 各分量的线性组合再加一个常数。$\mathbf x$ 是联合 Gaussian，按 3.4 的线性组合判定，它服从一维 Gaussian。$\mathbf y$ 的任意线性组合都是一维 Gaussian，所以 $\mathbf y$ 也是联合 Gaussian：

$$
\mathbf y \sim \mathcal N(M \boldsymbol \mu + \mathbf b, \; M \Sigma M^\top)
$$

几何上，线性变换把一个椭球变成另一个椭球；降维时（比如 $M$ 是 $2 \times 3$ 矩阵），就是把三维椭球压成二维椭圆。只要 $M \Sigma M^\top$ 满秩，结果就是非退化的。

一个常用的特例是**只保留部分分量**。比如取 $M = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}$，$\mathbf y$ 就是 $\mathbf x$ 的前两个分量，$M \Sigma M^\top$ 正好是 $\Sigma$ 左上角的 $2 \times 2$ 块。也就是说，Gaussian 丢掉一部分分量后仍是 Gaussian（称为边缘分布），协方差直接取对应的子块。6.4 节会用到这一点。

### 6.2 世界坐标系 → 相机坐标系

场景中的 Gaussian 用世界坐标描述，中心和协方差记为 $\boldsymbol \mu_w$、$\Sigma_w$。渲染某个视角时，先换到相机坐标系：以相机为原点、视线方向为 $Z$ 轴。这个变换由相机外参给出：

$$
\mathbf x_c = R_c \mathbf x_w + \mathbf t_c
$$

其中 $R_c$ 是旋转，$\mathbf t_c$ 是平移。这正是 6.1 中 $M = R_c$、$\mathbf b = \mathbf t_c$ 的情形，直接得到：

$$
\boldsymbol \mu_c = R_c \boldsymbol \mu_w + \mathbf t_c,
\qquad
\Sigma_c = R_c \Sigma_w R_c^\top
$$

平移 $\mathbf t_c$ 只移动中心，不改变协方差：协方差描述的是围绕中心的散开方式，整体平移不影响它。这里的 $R_c$ 是相机的旋转，不要和 5.2 节中 Gaussian 自身的朝向 $R$ 混淆。3DGS 论文把这一步的变换记作 $W$（viewing transformation）。

### 6.3 透视投影不是线性变换：Jacobian 从哪里来

接下来要把相机坐标投影到像素坐标。采用针孔相机模型（忽略镜头畸变），相机坐标 $\mathbf x_c = (X, Y, Z)^\top$（$Z > 0$，即点在相机前方）对应的像素坐标为

$$
\pi(\mathbf x_c) = \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x X / Z + c_x \\ f_y Y / Z + c_y \end{bmatrix}
$$

其中 $f_x, f_y$ 是以像素为单位的焦距，$(c_x, c_y)$ 是图像中心。

问题出在除以 $Z$：$\pi$ 不是线性变换，6.1 的结论不能直接套用。事实上，一个三维 Gaussian 经过精确的透视投影后，一般不再是严格的二维 Gaussian。

3DGS 的处理办法是**在 Gaussian 中心附近把 $\pi$ 近似成线性的**。Gaussian 的权重主要集中在中心附近，只要 $\pi$ 在这个范围内接近线性，近似就足够好。具体做法是在中心 $\boldsymbol \mu_c = (X_0, Y_0, Z_0)^\top$ 处做一阶泰勒展开：

$$
\pi(\mathbf x_c) \approx \pi(\boldsymbol \mu_c) + J (\mathbf x_c - \boldsymbol \mu_c)
$$

其中 $J$ 是 $\pi$ 在 $\boldsymbol \mu_c$ 处的 Jacobian 矩阵，即 $u$、$v$ 分别对 $X$、$Y$、$Z$ 求偏导：

$$
J = \begin{bmatrix} \dfrac{\partial u}{\partial X} & \dfrac{\partial u}{\partial Y} & \dfrac{\partial u}{\partial Z} \\ \dfrac{\partial v}{\partial X} & \dfrac{\partial v}{\partial Y} & \dfrac{\partial v}{\partial Z} \end{bmatrix}_{\boldsymbol \mu_c} = \begin{bmatrix} \dfrac{f_x}{Z_0} & 0 & -\dfrac{f_x X_0}{Z_0^2} \\ 0 & \dfrac{f_y}{Z_0} & -\dfrac{f_y Y_0}{Z_0^2} \end{bmatrix}
$$

逐项看：$u = f_x X / Z + c_x$ 对 $X$ 求导得 $f_x / Z$，对 $Y$ 求导得 0，对 $Z$ 求导得 $-f_x X / Z^2$；$v$ 同理。

$J$ 是 $2 \times 3$ 矩阵，把相机坐标中一个微小的三维位移，映射成屏幕上一个微小的二维位移。注意第三列：沿深度方向移动，像素位置也可能变化（离光轴越远越明显），所以深度方向的分布同样会影响屏幕上的形状。

### 6.4 屏幕上的二维协方差

现在把两步拼起来。在线性近似下，像素坐标 $\mathbf p$ 是 $\mathbf x_c$ 的仿射函数：

$$
\mathbf p \approx \pi(\boldsymbol \mu_c) + J (\mathbf x_c - \boldsymbol \mu_c)
$$

这是 6.1 中 $M = J$、$\mathbf b = \pi(\boldsymbol \mu_c) - J \boldsymbol \mu_c$ 的情形，于是屏幕上二维 Gaussian 的中心和协方差为：

$$
\boldsymbol \mu_{2D} = \pi(\boldsymbol \mu_c),
\qquad
\Sigma_{2D} \approx J \Sigma_c J^\top = J R_c \Sigma_w R_c^\top J^\top
$$

**这是 3DGS 中最核心的一条公式。** 论文中写作 $\Sigma' = J W \Sigma W^\top J^\top$，其中 $W$ 就是这里的 $R_c$。逐项检查矩阵维度：

$$
\underbrace{J}_{2 \times 3}
\underbrace{R_c}_{3 \times 3}
\underbrace{\Sigma_w}_{3 \times 3}
\underbrace{R_c^\top}_{3 \times 3}
\underbrace{J^\top}_{3 \times 2}
\longrightarrow
\underbrace{\Sigma_{2D}}_{2 \times 2}
$$

当 $\Sigma_w$ 正定、$Z_0 > 0$ 时，$J$ 的两行线性无关，$\Sigma_{2D}$ 也是正定的，可以求逆。

**和论文的写法对照。** 论文中的 $J$ 是 $3 \times 3$ 的：除了 $u$、$v$，还多一行深度方向的输出。算完 $J W \Sigma W^\top J^\top$ 得到 $3 \times 3$ 矩阵后，再"去掉第三行和第三列"，得到 $2 \times 2$ 的 $\Sigma'$。按 6.1 末尾的说法，这就是对深度求边缘分布，只保留屏幕上的两个分量。本文的 $J$ 只有 $u$、$v$ 两行，相当于提前去掉了那一行，结果完全相同。

要注意的是，去掉第三行第三列必须在乘过 $J$ 之后做。如果直接去掉 $\Sigma_c$ 的第三行第三列，就等于忽略了 $J$ 的第三列，即假设深度变化不影响像素位置，那是正交投影，而不是透视投影。

**用一个特例检验直觉。** 设中心正好在光轴上，即 $X_0 = Y_0 = 0$，且相机坐标中的协方差是对角阵 $\Sigma_c = \operatorname{diag}(\sigma_X^2, \sigma_Y^2, \sigma_Z^2)$。此时

$$
J =
\begin{bmatrix}
f_x / Z_0 & 0 & 0 \\
0 & f_y / Z_0 & 0
\end{bmatrix},
\qquad
\Sigma_{2D} =
\begin{bmatrix}
(f_x \sigma_X / Z_0)^2 & 0 \\
0 & (f_y \sigma_Y / Z_0)^2
\end{bmatrix}
$$

从中可以读出两点：

- 屏幕上的标准差是 $f \sigma / Z_0$：同样大小的 Gaussian 离相机越远（$Z_0$ 越大），屏幕上的椭圆越小，这就是"近大远小"。
- 在光轴上，深度方向的 $\sigma_Z$ 不影响屏幕上的形状：正对着看一个椭球，看不出它有多厚。离开光轴后，$J$ 的第三列不再为零，$\sigma_Z$ 和交叉协方差就会参与进来。

**近似的适用范围。** 一阶近似要求 $\pi$ 在 Gaussian 的覆盖范围内接近线性。Gaussian 相对它的深度较小、离相机较远时，近似很好；Gaussian 很大、离相机很近或横跨相机近平面时，误差会明显变大。另外，官方实现会给 $\Sigma_{2D}$ 的两个对角元各加上 0.3，即使用 $\Sigma_{2D} + 0.3 I$，保证每个 splat 在屏幕上至少约有一个像素大，起到低通滤波、减轻锯齿的作用。

## 七、从屏幕上的椭圆到像素颜色

第六章得到了每个 splat 在屏幕上的中心和形状。接下来要回答两个问题：对屏幕上某个像素，一个 splat 对它贡献多少？多个 splat 重叠时，又怎么合成最终颜色？

### 7.1 splat 在像素上的权重与不透明度

和 5.1 一样，屏幕上使用不带归一化系数的二维 Gaussian：

$$
G_{2D}(\mathbf p) = \exp\left( -\frac{1}{2} (\mathbf p - \boldsymbol \mu_{2D})^\top \Sigma_{2D}^{-1} (\mathbf p - \boldsymbol \mu_{2D}) \right)
$$

其中 $\mathbf p = (u, v)^\top$ 是像素坐标。$G_{2D}$ 衡量像素离 splat 中心有多近（按二维 Mahalanobis distance），中心处为 1，等值线是椭圆，椭圆的朝向和大小由 $\Sigma_{2D}$ 的特征分解给出（3.5 节）。

注意，第六章借用的是 Gaussian 随机变量在线性变换下的规律，用来求出屏幕上的中心和协方差；真正渲染时用的仍是形状权重 $G_{2D}$，而不是归一化的二维概率密度。

每个 Gaussian 还有一个可学习的不透明度 $o_i \in [0, 1]$（实现中用 sigmoid 把一个无约束的参数映射到这个区间）。第 $i$ 个 splat 在像素 $\mathbf p$ 处的 alpha（不透明程度）为

$$
\alpha_i(\mathbf p) = o_i \, G_{2D, i}(\mathbf p)
$$

于是三组参数各管一件事：$\boldsymbol \mu_{2D}$ 管 splat 在屏幕上的位置，$\Sigma_{2D}$ 管它的大小、拉伸和朝向，$o_i$ 管它中心处有多不透明。这就是 5.1 中"范围与浓淡分开调节"在屏幕上的体现。

### 7.2 多个 splat 重叠：alpha 合成，而不是 GMM

一个像素通常被多个 splat 覆盖。它们不像引言提到的 GMM 那样按权重直接相加，因为前面的 splat 会挡住后面的。3DGS 按 Gaussian 中心的深度把 splat 从近到远排序，然后逐个合成：

$$
C(\mathbf p) = \sum_{i=1}^{N} T_i(\mathbf p) \, \alpha_i(\mathbf p) \, \mathbf c_i + T_{N+1}(\mathbf p) \, C_{\mathrm{bg}},
\qquad
T_i(\mathbf p) = \prod_{j < i} \left( 1 - \alpha_j(\mathbf p) \right)
$$

逐项理解：

- $T_i$ 是光线到达第 $i$ 个 splat 之前还剩下的透过率：每经过一个 splat，就有 $\alpha_j$ 的比例被挡住，只剩 $1 - \alpha_j$ 继续向后；
- 第 $i$ 个 splat 贡献的颜色是"到达它的比例 $T_i$" × "它挡下的比例 $\alpha_i$" × "它的颜色 $\mathbf c_i$"；
- 所有 splat 都经过之后，剩下的 $T_{N+1}$ 由背景色 $C_{\mathrm{bg}}$ 补上。

其中 $\mathbf c_i$ 是第 $i$ 个 Gaussian 的颜色。在 3DGS 中它会随观察方向变化，用球谐函数表示，这里不展开。

## 八、小结：从 Gaussian 数学到 3DGS

从一个三维 Gaussian 到屏幕上的像素，可以串成一条链：

$$
\begin{aligned}
\Sigma_w &= R S S^\top R^\top
&& \text{（三维椭球：旋转与尺度，5.2 节）} \\
\boldsymbol \mu_c &= R_c \boldsymbol \mu_w + \mathbf t_c, \quad \Sigma_c = R_c \Sigma_w R_c^\top
&& \text{（转到相机坐标，6.2 节）} \\
\boldsymbol \mu_{2D} &= \pi(\boldsymbol \mu_c), \quad \Sigma_{2D} \approx J \Sigma_c J^\top
&& \text{（投影成屏幕上的椭圆，6.4 节）} \\
\alpha_i(\mathbf p) &= o_i \, G_{2D, i}(\mathbf p)
&& \text{（像素上的不透明度，7.1 节）}
\end{aligned}
$$

最后再由 7.2 节的 alpha 合成得到像素颜色。

每一步用到的都是前文的工具：协方差的几何意义（第三、四章），线性变换下协方差的变化规律 $M \Sigma M^\top$（6.1 节），以及不带归一化系数的 Gaussian 形状函数（5.1 节）。透视投影是整条链上唯一的非线性环节，3DGS 用 Jacobian 在中心处把它线性化。

理解了第五、六章的三个问题，即为什么用不带归一化系数的 Gaussian、为什么协方差要拆成旋转与尺度、为什么透视投影需要 Jacobian，就足以读懂 3DGS 原始论文中关于 Gaussian 表示与投影的数学部分。

接下来要学的是 3DGS 的算法本身：可微光栅化与训练、用球谐函数表示颜色、按屏幕分块（tile）排序加速，以及 Gaussian 的增密（densification）与剪枝（pruning）。这些可以放到下一篇文章。
