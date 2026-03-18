---
title: 【WIP】位置编码
date: 2026-03-18 09:16:32
tags:
  - Math
  - ML
  - AI
  - LLM
categories:
  - ML
---
# 简介
Transformer 的 encoder 和 decoder 都要求在embedding之后、进入注意力层之前，加入位置编码，使得输入满足：

$$
\text{input} = \text{input\_embedding} + \text{input\_encoding}
$$

位置编码的演进有如下阶段：

## 整数标记

给第一个token标记1，给第二个token标记2...，以此类推，这种方法产生了以下几个主要问题：
  - 模型可能遇见比训练时所用的序列更长的序列。不利于模型的泛化。
  - 模型的位置表示是无界的。随着序列长度的增加，位置值会越来越大。
值越来越大会导致：
1. **尺度爆炸：** embedding一般会做归一化，数值范围较小，而巨大的位置信息会淹没词的语义信息
2. **梯度异常：** 巨大的 $Q \cdot K$ 经过 $\text{softmax}$ 后导致大部分概率被压缩到0，梯度消失
3. **无法表示相对位置：** 1 vs 2 和 100 vs 101 相对关系完全不同，难以学习

## 用[0,1]范围标记位置
将位置值的范围限制在[0, 1]之内，但当序列长度不同时，token间的相对距离不一样。

## 周期函数（sin）

通过一个 $\mathbb{R}^{1 \times d_{model}}$ 维向量来表示，向量的每一个维度代表某个精度下的值，如果位置向量当中的每一个元素都用一个sin函数来表示，则第t个token的位置向量可以表示为：
$$
\text{PE}_t = [sin(\frac{1}{2^0}t), sin(\frac{1}{2^1}t), ..., sin(\frac{1}{2^{d_{model} - 1}}t)]
$$

周期函数避免了梯度爆炸，并且在不同序列长度中，对应位置的值没有变化

## 相对位置

普通的周期函数难以捕获相对位置之间的关联，我们希望不同位置之间能通过线性变换相互转换

$$
PE_{t+k, i} = M_k \cdot PE_{t, i}
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

使用这样的方案，位置编码的频率就需要是 $\frac{d_{model}}{2}$ ，每个频率 $\omega_i$ 都会同时生成一个 $\sin$ 和一个 $\cos$，这样拼起来维度仍然是 $d_{model}$。关于拼接形式，通常有两种排列方式：

- **交错式（Interleaved）：** $[\sin_0, \cos_0, \sin_1, \cos_1, \dots]$（这在旋转位置编码 RoPE 中更常用）。
- **拼接式（Concatenated）：** 原始论文采用的方式，前 $d/2$ 维是 $\sin$，后 $d/2$ 维是对应的 $\cos$。
