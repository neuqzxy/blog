---
title: 归一化
date: 2026-02-22 13:22:36
tags:
  - Math
  - ML
  - AI
  - LLM
categories:
  - ML
---
## Batch Normalization（BN，批归一化）
> 核心思想：跨样本、同特征维度进行归一化，即对一个批次内的所有样本，在每个特征通道上计算均值和方差。

记号约定：

- $B$：batch size；
- $D$：特征维度数；
- $i \in \{1,\dots,B\}$：样本索引；
- $d \in \{1,\dots,D\}$：特征维度索引；
- $X \in \mathbb{R}^{B \times D}$：输入张量，$X_{i,d}$ 表示第 $i$ 个样本在第 $d$ 个特征维度上的取值；
- $Y \in \mathbb{R}^{B \times D}$：归一化后的输出张量。

### 1. 计算跨样本的均值（批次均值）
对每个特征维度 $d$，计算当前批次所有样本在该维度上的均值 $\mu_d$：

$$
\mu_d = \frac{1}{B} \sum_{i=1}^B X_{i,d}
$$

- $X_{i,d}$：第 $i$ 个样本的第 $d$ 个特征值；
- 含义：**同一特征维度在不同样本上的平均水平**。

### 2. 计算跨样本的方差（批次方差）

对每个特征维度 $d$，计算当前批次所有样本在该维度上的方差 $\sigma_d^2$：

$$
\sigma_d^2 = \frac{1}{B} \sum_{i=1}^B (X_{i,d} - \mu_d)^2
$$

- $\epsilon$：极小的常数（如 $10^{-5}$），在后续标准化时加到分母里，避免方差为 0 导致计算错误。

### 3. 标准化（归一化）

将每个特征维度 $d$ 的值标准化为均值 0、方差 1 的分布，得到 $\hat{X}_{i,d}$：

$$
\hat{X_{i,d}} = \frac{X_{i,d} - \mu_d}{\sqrt{\sigma_d^2 + \epsilon}}
$$

### 4. 缩放与平移（可学习参数）

引入可学习参数 $\gamma_d$（缩放因子）和 $\beta_d$（平移因子），恢复特征的表达能力（每个特征维度对应一组参数）：

$$
Y_{i,d} = \gamma_d \cdot \hat{X}_{i,d} + \beta_d
$$

- $\gamma, \beta \in \mathbb{R}^{D}$：按特征维度的一维可学习参数向量，通常初始化为 $\gamma = \mathbf{1}, \beta = \mathbf{0}$；
- 输出 $Y$：归一化后的张量，形状与输入 $X$ 完全一致（$B \times D$）。

## Layer Normalization（LN，层归一化）
> 同一样本、跨特征维度进行归一化，即对单个样本的所有特征维度计算均值和方差

### 1. 对单个样本计算均值

对第 $i$ 个样本（向量维度为 $D$），LN 的均值为：

$$
\mu_i = \frac{1}{D} \sum_{d=1}^{D} X_{i,d}
$$

### 2. 对单个样本计算方差

$$
\sigma_i^2 = \frac{1}{D} \sum_{d=1}^{D} (X_{i,d} - \mu_i)^2
$$

- $\epsilon$：极小的常数（如 $10^{-5}$），在后续标准化时加到分母里，避免方差为 0 导致计算错误。

### 3. 标准化（归一化）

对同一个样本的每个特征维度做标准化：

$$
\hat{X_{i,d}} = \frac{X_{i,d} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}
$$

### 4. 缩放与平移（可学习参数）

LN 同样引入可学习的缩放/平移参数（通常是“按特征维度”一一对应）：

$$
Y_{i,d} = \gamma_d \cdot \hat{X}_{i,d} + \beta_d
$$

- $\gamma, \beta \in \mathbb{R}^{D}$：每个特征维度一组可学习参数；
- 输出 $Y$ 的形状与输入 $X$ 相同。

### 5. 更贴近 Transformer 的写法（按 token 向量做 LN）

以 Transformer 为例，输入常见形状是 $X \in \mathbb{R}^{B \times T \times H}$（batch、序列长度、hidden size）。LN 通常对最后一维 $H$ 做归一化

$$
\mu_{b,t} = \frac{1}{H}\sum_{h=1}^{H} X_{b,t,h},\quad
\sigma_{b,t}^2 = \frac{1}{H}\sum_{h=1}^{H}(X_{b,t,h}-\mu_{b,t})^2 + \epsilon
$$

$$
Y_{b,t,h} = \gamma_h\cdot \frac{X_{b,t,h}-\mu_{b,t}}{\sqrt{\sigma_{b,t}^2}} + \beta_h
$$

- $\gamma, \beta \in \mathbb{R}^{H}$：每个特征维度一组可学习参数；
- 输出 $Y$ 的形状与输入 $X$ 相同。

## RMSNorm（Root Mean Square Layer Normalization）
> 这是目前大模型（如 Llama 2/3、Gopher、Chinchilla）中最常使用的归一化方式。

**核心思想**：RMSNorm 可以视为 LN 的简化版。与 LN 不同，它不减去均值，只做基于二阶矩的缩放归一化（强调“缩放不变性”）。

RMSNorm所需的参数仅为LN的一半，且计算效率更高，模型表现上，LN和RMSNorm相当

### 1. 计算 RMS（按张量的归一化维度）

延续前面对 Transformer 的记号约定：输入 $X \in \mathbb{R}^{B \times T \times H}$（batch、序列长度、hidden size），RMSNorm 也是**对最后一维 $H$** 做归一化。

对每个 token 向量 $X_{b,t,:}$，定义其 RMS 为：

$$
\mathrm{RMS}(X_{b,t,:}) = \sqrt{\frac{1}{H}\sum_{h=1}^{H} X_{b,t,h}^{2} + \epsilon}
$$

其中 $\epsilon$：极小常数（如 $10^{-5}$），避免分母为 0 导致数值不稳定。

### 2. 归一化与缩放（可学习参数）

基于上述 RMS，RMSNorm 的张量形式可以写成：

$$
Y_{b,t,h} = \gamma_h \cdot \frac{X_{b,t,h}}{\mathrm{RMS}(X_{b,t,:})}
$$

其中 $\gamma \in \mathbb{R}^{H}$：沿最后一维（hidden size）逐元素的可学习缩放参数向量，输出张量 $Y \in \mathbb{R}^{B \times T \times H}$ 的形状与输入 $X$ 相同。
