---
title: 从理论到实战：深入理解 3D Gaussian Splatting（3DGS）
date: 2026-10-14 21:16:48
tags:
  - Math
  - ML
  - AI
categories:
  - ML
---
## 一、3DGS 究竟解决了什么问题？

假设围绕一个物体拍摄了 100 张照片。它们的位置和角度不同，但都来自同一个静态场景。目标是做一个能够新视角合成（Novel View Synthesis）的三维表示：给定一个新的相机位置和方向，生成该视角下的图像。

输入只有二维照片，没有物体表面、深度或完整几何。三维结构要从这些多视角图像里反推出来。常见的表示有：

| 表示方法 | 基本思路 | 特点 |
|---|---|---|
| Point Cloud | 使用离散三维点表示场景 | 简单，但点本身没有面积 |
| Mesh | 使用顶点和三角面片表示表面 | 几何明确，适合传统渲染 |
| Voxel | 使用三维网格表示空间 | 容易理解，但存储开销较大 |
| NeRF | 使用神经网络隐式表示场景 | 画质好，但原始方法渲染较慢 |
| 3DGS | 使用大量可优化的三维高斯表示场景 | 显式表示，适合高效可微渲染 |

### 1.1 3DGS 的整体算法流程

点云里的一个点只有位置，没有体积：

$$
P = (x, y, z)
$$

3DGS 为它补上尺度和方向。一个三维高斯可以写成：

$$
G(\mathbf x) = \exp\left( -\frac{1}{2} (\mathbf x - \boldsymbol\mu)^\top \Sigma^{-1} (\mathbf x - \boldsymbol\mu) \right)
$$

其中：
- $\boldsymbol\mu$：高斯中心，确定位置。
- $\Sigma$：协方差矩阵，确定形状、大小和方向。
- $G(\mathbf x)$：位置 $\mathbf x$ 处的高斯权重。

3DGS 用高斯核表示空间里的影响范围，不要求每个 Gaussian 都是积分为 1 的概率密度，所以上式省略了归一化系数。

整个流程分成四步：
- 初始化：用 COLMAP 从多视角照片得到相机参数和稀疏点云，再据此初始化 Gaussian。这一步只做一次。
- 可微渲染：把三维 Gaussian 投影到图像平面，用 alpha 合成（Alpha Compositing）计算像素颜色。
- 参数优化：比较渲染图像和真实照片，用梯度下降更新 Gaussian 参数。
- 调整数量：用增密（Densification）在需要更多细节的区域增加 Gaussian，用剪枝（Pruning）删掉几乎没贡献或尺度过大的 Gaussian。

渲染和参数优化每一步都做。数量调整只插在训练前半段，每隔一段迭代做一次，直到渲染结果接近真实照片。

```
               输入：多视角照片
                       │
                       ▼
              COLMAP / SfM
                       │
              ┌────────┴────────┐
              ▼                 ▼
         相机位姿             稀疏点云
              │                 │
              │                 ▼
              │        初始化 3D Gaussians
              │                 │
              │                 ▼
              │       位置 / 尺度 / 旋转
              │       不透明度 / SH系数
              │                 │
              └────────┬────────┘
                       │
                       ▼
                3D → 2D 投影
                       │
                       ▼
               Gaussian Splatting
                       │
                       ▼
                  Alpha 合成
                       │
                       ▼
                   渲染图像
                       │
                       ▼
              与真实照片计算 Loss
                       │
                       ▼
                   反向传播
                       │
                       ▼
             更新 Gaussian 参数
                       │
                       ▼
             Densification / Pruning
              高斯增密 / 高斯剪枝
                       │
                       ▼
                   重复优化
                       │
                       ▼
               最终 3DGS 场景
```

## 二、3D Gaussian 中的参数

原始 3DGS 里，每个 Gaussian 记为：

$$
G_i = \{ \boldsymbol\mu_i, \mathbf s_i, \mathbf q_i, o_i, \mathbf c_i \}
$$

其中：

| 参数 | 维度 | 作用 |
|---|---|---|
| $\boldsymbol\mu$ | 3 | 三维中心位置 |
| $\mathbf s$ | 3 | 三个主轴方向的尺度 |
| $\mathbf q$ | 4 | 旋转四元数 |
| $o$ | 1 | 不透明度 |
| $\mathbf c$ | 取决于 SH 阶数 | 视角相关的颜色 |

一个 Gaussian 可以移动、变大变小、旋转，也可以改变透明度和颜色。这些自由度让它能够表示场景里不同位置的外观。

### 2.1 位置参数

位置是高斯中心：

$$
\boldsymbol\mu = \begin{bmatrix} \mu_x \\ \mu_y \\ \mu_z \end{bmatrix}
$$

训练时优化 $\boldsymbol\mu$，Gaussian 就会在三维空间里移动。

### 2.2 尺度和旋转

三维高斯的协方差是：

$$
\Sigma = \begin{bmatrix} \sigma_x^2 & \sigma_{xy} & \sigma_{xz} \\ \sigma_{xy} & \sigma_y^2 & \sigma_{yz} \\ \sigma_{xz} & \sigma_{yz} & \sigma_z^2 \end{bmatrix}
$$

直接改这六个独立元素，不能保证矩阵一直合法。**协方差必须对称半正定，原始 3DGS 还希望它非退化、从而正定**。因此不直接优化这六个数，而用尺度和旋转来构造：

$$
\Sigma = R S S^\top R^\top
$$

其中 $S$ 是尺度对角阵，$R$ 是由旋转四元数得到的旋转矩阵：

$$
S = \begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & s_z \end{bmatrix}
$$

$S$ 是对角阵，所以 $S S^\top = \operatorname{diag}(s_x^2, s_y^2, s_z^2)$，上式也可以写成：

$$
\Sigma = R \begin{bmatrix} s_x^2 & 0 & 0 \\ 0 & s_y^2 & 0 \\ 0 & 0 & s_z^2 \end{bmatrix} R^\top
$$

这样得到的 $\Sigma$ 一定半正定。对任意向量 $\mathbf v$，令 $\mathbf a = S^\top R^\top \mathbf v$，则：

$$
\mathbf v^\top \Sigma \mathbf v = \mathbf v^\top R S S^\top R^\top \mathbf v = \mathbf a^\top \mathbf a \geq 0
$$

三个尺度都不为 0 时，$\Sigma$ 还是正定的。几何含义也很直接：
- $s_x, s_y, s_z$：$1\sigma$ 椭球沿三个主轴的半轴长，也就是各方向上的标准差。协方差的特征值是 $s_x^2, s_y^2, s_z^2$。
- $R$：椭球在空间中的朝向。

训练时，尺度先经过指数映射，保证为正；四元数在使用前归一化。

### 2.3 不透明度

每个 Gaussian 还有一个不透明度 $o_i \in [0, 1]$。它决定渲染时这个 Gaussian 能挡住多少：$o_i$ 越大越不透明，越小越透明。

$o_i$ 和高斯函数值不是一回事。$G(\mathbf x)$ 描述空间里的影响范围，$o_i$ 描述这份影响有多强。两者会一起进入像素的 alpha。

### 2.4 颜色：基于球谐函数

普通点云可以为每个点存一组固定颜色 $\mathbf C = (R, G, B)$。真实物体的颜色会随观察方向变化，例如金属反射、玻璃高光，以及不同视角下的材质外观。原始 3DGS 用球谐函数（Spherical Harmonics，SH）表示这种视角相关的颜色。对某个 Gaussian，取从相机中心指向它中心的单位向量作为观察方向 $\mathbf d$：

$$
\hat{\mathbf C}(\mathbf d) = \sum_{l=0}^{L} \sum_{m=-l}^{l} \mathbf c_l^m Y_l^m(\mathbf d)
$$

其中：
- $Y_l^m$：预先定义好的球谐基函数。
- $\mathbf c_l^m$：需要学习的 RGB 系数。
- $\mathbf d$：观察方向。
- $\hat{\mathbf C}(\mathbf d)$：该方向下的颜色。

3 阶 SH 有 $(L + 1)^2 = 16$ 组系数，再乘上 RGB 三个通道，是 48 个标量。加上位置 3、尺度 3、旋转 4、不透明度 1，一个使用 3 阶 SH 的 Gaussian 共有 59 个可优化的标量参数。

## 三、3D Gaussian 的投影

三维 Gaussian 由中心 $\boldsymbol\mu \in \mathbb{R}^3$ 和协方差 $\Sigma \in \mathbb{R}^{3 \times 3}$ 决定。渲染时相机看到的是二维图像，所以要回答：这个三维椭球投到图像上是什么形状。

### 3.1 从世界坐标变换到相机坐标

记 Gaussian 中心的世界坐标为 $\boldsymbol\mu_w$。世界坐标到相机坐标的变换是 $\mathbf x_c = R_c \mathbf x_w + \mathbf t_c$，因此中心变为：

$$
\boldsymbol\mu_c = R_c \boldsymbol\mu_w + \mathbf t_c
$$

高斯分布经过仿射变换 $\mathbf y = A \mathbf x + \mathbf b$ 之后仍是高斯。新的协方差是 $A \Sigma A^\top$，只和线性部分 $A$ 有关，平移 $\mathbf b$ 不出现在这个式子里。这里 $A$ 就是旋转 $R_c$，所以协方差变成：

$$
\Sigma_c = R_c \Sigma_w R_c^\top
$$

### 3.2 透视投影

相机坐标 $\mathbf x_c = (X, Y, Z)^\top$ 用针孔模型投到像素：

$$
u = f_x \frac{X}{Z} + c_x, \quad v = f_y \frac{Y}{Z} + c_y
$$

其中：
- $f_x, f_y$：焦距。
- $c_x, c_y$：主点。
- $Z$：点到相机的深度。

把相机坐标中心 $\boldsymbol\mu_c = (X, Y, Z)^\top$ 代进去，就得到屏幕上的投影中心：

$$
\boldsymbol\mu_{2D} = \begin{bmatrix} f_x X / Z + c_x \\ f_y Y / Z + c_y \end{bmatrix}
$$

中心的位置这样就能确定。椭球的形状还不行，因为除以 $Z$ 使投影不是仿射变换，三维高斯经过精确的透视投影后，一般不再是二维高斯。

3DGS 在中心附近做一阶近似：

$$
\pi(\mathbf x) \approx \pi(\boldsymbol\mu_c) + J (\mathbf x - \boldsymbol\mu_c)
$$

$J$ 是 $\pi$ 在 $\boldsymbol\mu_c$ 处的 Jacobian，即 $u$、$v$ 对 $X$、$Y$、$Z$ 的偏导。$u = f_x X / Z + c_x$ 对 $X$ 求导得 $f_x / Z$，对 $Y$ 得 0，对 $Z$ 得 $-f_x X / Z^2$；$v$ 同理。于是：

$$
J = \begin{bmatrix} \dfrac{f_x}{Z} & 0 & -\dfrac{f_x X}{Z^2} \\ 0 & \dfrac{f_y}{Z} & -\dfrac{f_y Y}{Z^2} \end{bmatrix}
$$

$J$ 是 $2 \times 3$ 矩阵，把相机坐标里的局部位移映射成屏幕上的位移。协方差在这个线性近似下变为：

$$
\Sigma_{2D} \approx J \Sigma_c J^\top = J R_c \Sigma_w R_c^\top J^\top
$$

维度是 $(2 \times 3)(3 \times 3)(3 \times 2) = 2 \times 2$，所以屏幕上的协方差是：

$$
\Sigma_{2D} = \begin{bmatrix} \sigma_u^2 & \sigma_{uv} \\ \sigma_{uv} & \sigma_v^2 \end{bmatrix}
$$

它描述 Gaussian 投到屏幕上的椭圆。实际光栅化时，还会对这个矩阵做数值稳定性和滤波处理。

## 四、Gaussian Splatting

前面已经把三维 Gaussian 投影成了屏幕上的二维高斯。接下来要问：**这个二维高斯怎样影响像素**。

### 4.1 单个 Gaussian 的像素权重

Splatting 把一个空间基元投到图像平面上，让它覆盖一片像素。一个 Gaussian 不只对应一个像素。离投影中心越近，影响越大；越远，影响越小。

设投影中心和屏幕协方差为 $\boldsymbol\mu_{2D} = (u_0, v_0)^\top$、$\Sigma_{2D}$。像素 $\mathbf p = (u, v)^\top$ 相对中心的偏移是 $\boldsymbol\delta = \mathbf p - \boldsymbol\mu_{2D}$。二维高斯核是：

$$
g(\mathbf p) = \exp\left( -\frac{1}{2} \boldsymbol\delta^\top \Sigma_{2D}^{-1} \boldsymbol\delta \right)
$$

$0 < g(\mathbf p) \leq 1$。像素落在投影中心时 $\boldsymbol\delta = \mathbf 0$，$g(\mathbf p) = 1$。再乘上该 Gaussian 自己的不透明度，得到它对这个像素的 alpha：

$$
\alpha_i(\mathbf p) = o_i \, g_i(\mathbf p)
$$

其中：
- $o_i$：该 Gaussian 的不透明度。
- $g_i(\mathbf p)$：它在当前像素上的高斯权重。
- $\alpha_i(\mathbf p)$：它对当前像素的 alpha。

同一个 Gaussian 在不同像素上的 alpha 不同。

### 4.2 多个 Gaussian 的颜色合成

一个像素往往被多个 Gaussian 覆盖。每个 Gaussian 有自己的 alpha、由球谐算出的颜色，以及深度。3DGS 按深度从近到远排序，再用 alpha 合成（Alpha Compositing）。

以前面三个为例，排序后是 $G_1$、$G_2$、$G_3$，颜色是 $\hat{\mathbf C}_1$、$\hat{\mathbf C}_2$、$\hat{\mathbf C}_3$，alpha 是 $\alpha_1$、$\alpha_2$、$\alpha_3$。最近的一个直接贡献 $\alpha_1 \hat{\mathbf C}_1$。第二个要先穿过第一个剩下的透明度，贡献 $(1 - \alpha_1) \alpha_2 \hat{\mathbf C}_2$。第三个再穿过前两个：

$$
\mathbf C = \alpha_1 \hat{\mathbf C}_1 + (1 - \alpha_1) \alpha_2 \hat{\mathbf C}_2 + (1 - \alpha_1)(1 - \alpha_2) \alpha_3 \hat{\mathbf C}_3
$$

$N$ 个 Gaussian 写成：

$$
\mathbf C(\mathbf p) = \sum_{i=1}^{N} T_i(\mathbf p) \, \alpha_i(\mathbf p) \, \hat{\mathbf C}_i(\mathbf d_i)
$$

前面各层剩下的透射率是：

$$
T_i(\mathbf p) = \prod_{j=1}^{i-1} \bigl(1 - \alpha_j(\mathbf p)\bigr)
$$

其中：
- $T_i$：排在前面的 Gaussian 都穿过之后，还剩下的透射率。$i = 1$ 时连乘为空，记 $T_1 = 1$。
- $\alpha_i$：当前 Gaussian 对这个像素的 alpha。
- $\hat{\mathbf C}_i(\mathbf d_i)$：用球谐在该 Gaussian 自己的观察方向上算出的颜色。$\mathbf d_i$ 是从相机中心指向这个 Gaussian 中心的单位向量，同一张图里各个 Gaussian 的方向不同。这个 Gaussian 覆盖的像素共用这一份颜色。

若还有背景色 $\mathbf C_{\mathrm{bg}}$，要加上最后剩下的透射率乘背景：

$$
\mathbf C = \sum_{i=1}^{N} T_i \alpha_i \hat{\mathbf C}_i + T_{N+1} \mathbf C_{\mathrm{bg}}
$$

### 4.3 一个具体数值例子

设某个像素被两个 Gaussian 覆盖，前面是红色，后面是蓝色，背景为黑：

$$
\alpha_1 = 0.6, \quad \hat{\mathbf C}_1 = (1, 0, 0)
$$

$$
\alpha_2 = 0.5, \quad \hat{\mathbf C}_2 = (0, 0, 1)
$$

$$
\mathbf C = 0.6 \, (1, 0, 0) + (1 - 0.6) \times 0.5 \, (0, 0, 1) = (0.6, 0, 0.2)
$$

红色贡献 $0.6$，蓝色贡献 $0.2$，穿过去之后还剩 $0.2$ 的透射率。两个 Gaussian 对调前后，颜色一般会变，所以深度排序是合成的一部分。实际渲染在 GPU 上按图像块排序和光栅化，不会让每个像素去遍历场景里的全部 Gaussian。

## 五、梯度下降

3DGS 要找一组 Gaussian，使各训练视角渲染出的图像接近真实照片。记场景为 $\mathcal G = \{G_1, G_2, \ldots, G_N\}$。相机 $P$ 下的渲染结果是：

$$
\hat{I} = \mathcal{R}(\mathcal G, P)
$$

其中：
- $\mathcal G$：场景中的全部 Gaussian。
- $P$：相机参数。
- $\mathcal{R}$：渲染函数。
- $\hat{I}$：渲染得到的图像。

该视角的真实照片是 $I_{\mathrm{gt}}$，希望 $\hat{I} \approx I_{\mathrm{gt}}$。写成优化问题：

$$
\mathcal G^* = \arg\min_{\mathcal G} \mathcal L\bigl(\mathcal{R}(\mathcal G, P), I_{\mathrm{gt}}\bigr)
$$

$M$ 个训练视角则是把各视角的损失加起来：

$$
\mathcal G^* = \arg\min_{\mathcal G} \sum_{k=1}^{M} \mathcal L\bigl(\mathcal{R}(\mathcal G, P_k), I_k\bigr)
$$

### 5.1 使用 L1 + D-SSIM 作为损失函数

原始 3DGS 使用：

$$
\mathcal L = (1 - \lambda) \mathcal L_{L1} + \lambda \mathcal L_{D\text{-}SSIM}
$$

$\mathcal L_{L1}$ 衡量像素绝对误差，$\mathcal L_{D\text{-}SSIM} = 1 - \operatorname{SSIM}(\hat{I}, I_{\mathrm{gt}})$ 衡量结构不相似。常用 $\lambda = 0.2$，也就是：

$$
\mathcal L = 0.8 \, \mathcal L_{L1} + 0.2 \, \bigl(1 - \operatorname{SSIM}(\hat{I}, I_{\mathrm{gt}})\bigr)
$$

### 5.2 梯度传导

一个像素的颜色是：

$$
\mathbf C(\mathbf p) = \sum_i T_i \, \alpha_i \, \hat{\mathbf C}_i
$$

其中 alpha 来自不透明度和屏幕上的高斯核，$\boldsymbol\delta_i$ 是像素相对该 Gaussian 投影中心的偏移：

$$
\alpha_i = o_i \exp\left( -\frac{1}{2} \boldsymbol\delta_i^\top \Sigma_{2D,i}^{-1} \boldsymbol\delta_i \right)
$$

屏幕协方差又由三维协方差投影而来，三维协方差由尺度和旋转组成：

$$
\Sigma_{2D,i} = J R_c \Sigma_i R_c^\top J^\top, \quad \Sigma_i = R_i S_i S_i^\top R_i^\top
$$

因此像素颜色依赖 alpha，alpha 依赖 $\Sigma_{2D}$，$\Sigma_{2D}$ 依赖 $\Sigma$，而 $\Sigma$ 依赖尺度和旋转。损失对某个尺度 $s_x$ 的梯度可以沿这条链写出来：

$$
\frac{\partial \mathcal L}{\partial s_x} = \frac{\partial \mathcal L}{\partial \mathbf C} \frac{\partial \mathbf C}{\partial \alpha} \frac{\partial \alpha}{\partial \Sigma_{2D}} \frac{\partial \Sigma_{2D}}{\partial \Sigma} \frac{\partial \Sigma}{\partial s_x}
$$

这是链式法则的示意。实际求导是向量和矩阵的导数，还要把各个像素的贡献加起来。位置、不透明度和 SH 系数也在这条计算图里，同样可以由 $\mathcal L$ 更新。

## 六、Gaussian 的数量变化

梯度下降只能改已有 Gaussian 的参数，不能决定该增加多少个。假设初始点云有 10 万个点，就先放 10 万个 Gaussian。窗户、树叶、尖锐边缘和细碎纹理这些地方，往往需要更多个。数量一直不动，只优化现有参数，这些细节仍然可能表达不出来。

3DGS 用自适应密度控制（Adaptive Density Control）增减数量，包括克隆（Clone）、分裂（Split）和删除（Prune）。

哪里该增密，主要看损失对屏幕投影中心的梯度有多大：

$$
\left\| \frac{\partial \mathcal L}{\partial \boldsymbol\mu_{2D}} \right\|
$$

这个值大，说明挪动该 Gaussian 在图像上的位置会明显改变损失，它所在的区域可能还没表示准。原始实现在每个训练视角里，把可见 Gaussian 的这份范数累加起来，并记下被看到的次数。到了增密那一轮，用平均幅度做判断：

$$
\frac{1}{N_{\mathrm{vis}}} \sum \left\| \frac{\partial \mathcal L}{\partial \boldsymbol\mu_{2D}} \right\| > \tau_{\mathrm{pos}}
$$

$N_{\mathrm{vis}}$ 是自上一次清零以来它被看到的次数，默认 $\tau_{\mathrm{pos}} = 0.0002$。除以次数之后，经常入画的 Gaussian 不会只因为出现得多就超过阈值。每次增密结束都会把累加器和计数清零，所以这里平均的是这一段间隔里的梯度。

默认从第 500 轮到第 15000 轮，每 100 轮增密一次。平均幅度超过阈值之后，再用空间尺度决定克隆还是分裂：
- 位置梯度大，而且最长轴还不大：克隆一个新的出来。细节不够，但现有的这个已经不大，复制一份再分开更合适。
- 位置梯度大，而且最长轴已经偏大：把它分裂成更小的。一个大椭球跨过物体边界时，没法同时贴合边界两侧。

梯度大小是启发式指标，不能证明那一块一定缺 Gaussian。

多大算大、多小算小，官方代码只看最长主轴，体积不参与比较。一根细长的高斯可以体积很小，只要有一条轴很长，就按大的处理。

1. 取最大主轴尺度 $\max(s_x, s_y, s_z)$。
2. 阈值是 $\texttt{percent\_dense} \times \texttt{cameras\_extent}$。$\texttt{percent\_dense}$ 默认是 $0.01$。$\texttt{cameras\_extent}$ 在训练开始时由相机位置算出：先取全部训练相机中心的平均位置，再取离这个平均位置最远的相机中心，把这段距离乘上 $1.1$。点云包围盒不进入这个阈值。
3. 平均位置梯度已经超过 $\tau_{\mathrm{pos}}$ 时：
  - $\max(s_x, s_y, s_z) > \texttt{percent\_dense} \times \texttt{cameras\_extent}$：最长轴偏大，甚至可能跨过不同的几何边界，执行分裂（Split）。
  - $\max(s_x, s_y, s_z) \le \texttt{percent\_dense} \times \texttt{cameras\_extent}$：最长轴还不大，执行克隆（Clone）。

不断增加还会让数量膨胀：显存变大，渲染变慢，场景里留下很多几乎没有贡献的 Gaussian。同一次密度控制里还会剪枝。不透明度低于 $0.005$ 的会删掉：

$$
o_i < 0.005
$$

第一次不透明度重置之后，世界空间里最长轴过大的也会删掉：

$$
\max(s_x, s_y, s_z) > 0.1 \times \texttt{cameras\_extent}
$$

代码里还有一条「屏幕半径大于 20 像素就删除」。它写在克隆和分裂之后，而这两步会把记录屏幕半径的数组整个清零，比较时读到的全是 0，这条判断不会成立。作者确认过，这是评测时就已经存在的 bug，论文结果并不依赖它。

不透明度常常会停在 $0.005$ 上面，单靠这条阈值清不掉。所以每 3000 轮会把每个 Gaussian 的不透明度压到不超过 $0.01$。接下来用不到的那些升不回去，就会在后面的剪枝里被删掉。
