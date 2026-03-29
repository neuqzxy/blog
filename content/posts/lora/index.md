---
title: Lora微调
date: 2026-03-20 21:18:26
tags:
  - Math
  - ML
  - AI
  - LLM
categories:
  - ML
---

> LoRA的本质就是通过`低秩近似`，用更少的训练参数来**近似LLM全参数微调所得的增量参数**

## FT 问题定义

给定一个参数为 $\boldsymbol{\Phi}$ 的预训练自回归语言模型 $P_{\boldsymbol{\Phi}}(y|x)$

### 全量微调

全量微调过程中，模型会以预训练的权重 $\boldsymbol{\Phi}_0$ 初始化，然后反复沿着梯度反方向进行更新得到 $\boldsymbol{\Phi}_0 + \boldsymbol{\Delta \Phi}$

$$
\max_{\boldsymbol{\Phi}} \sum_{(x,y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log\left(P_{\boldsymbol{\Phi}}(y_t \mid x, y_{1:t−1})\right)
$$

该方案的主要缺陷是：

1. 针对每个下游任务，都需要学习一组独立的 $\boldsymbol{\Delta \Phi}$ ，且维度和 $\boldsymbol{\Phi_0}$ 完全相等，导致训练规模庞大

2. 全量微调会直接更新全部原始模型参数 $\boldsymbol{\Phi}$ ，每个任务的微调模型都需要保存一份完整参数副本

### Lora微调

Lora微调解决了全量微调的两个缺点：

1. 冻结预训练模型权重，独立的训练增量 $\Delta\Phi(\Theta)$ ，针对下游的每个任务只需要保存一个增量即可，在使用的时候直接和原始参数相加即可 $\Phi_0+\Delta\Phi(\Theta)$

2. 增量 $\Delta\Phi(\Theta)$ 通过低秩近似，显著降低了训练所需的计算和内存

$$
\max_{\Theta} \sum_{(x,y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log\left(P_{\Phi_0+\Delta\Phi(\Theta)}(y_t \mid x, y_{1:t−1})\right)
$$

![](/img/lora.png)

## Lora的实现

对于预训练权重 $\boldsymbol{W}_0 \in \mathbb{R}^{d \times d}$ 和全参微调的增量参数矩阵 $\boldsymbol{\Delta W} \in \mathbb{R}^{d \times d}$ 有如下近似

$$
\boldsymbol{W}_0 + \Delta\boldsymbol{W} = \boldsymbol{W}_0 + \boldsymbol{B}\boldsymbol{A}
$$

微调的参数量从原本 $\boldsymbol{\Delta W}$ 的 $d \times d$ 变成了 $2 * d \times r$。在推理时，直接按上面的式子将 $\boldsymbol{B}\boldsymbol{A}$ 合并到 $\boldsymbol{W}_0$ 中，因此相比原始LLM不存在**推理延时**。
> **推理延时（Inference Latency）** 指：模型从接收输入到输出完整结果所花费的总时间。

### 参数初始化
在训练时，我们要求 $\boldsymbol{B}$ 和 $\boldsymbol{A}$ 中有一个被初始化为全0项。这样初始的 $\boldsymbol{\Delta W}$ 的输出就是0，但 $\boldsymbol{B}$ 和 $\boldsymbol{A}$ 不能都是零矩阵，否则处于鞍点，两个权重的梯度也全为0。

### 参数合并
实际实现时，$\Delta\boldsymbol{W} = \boldsymbol{B}\boldsymbol{A}$ 会乘以系数 $\frac{\alpha}{r}$ 与原始预训练权重合并 $\boldsymbol{W}_0$，$\alpha$ 是一个超参：

$$
\boldsymbol{h} = \left(\boldsymbol{W}_0 + \frac{\alpha}{r}\Delta\boldsymbol{W}\right)\boldsymbol{x}
$$

根据矩阵乘法的性质，两个矩阵相乘后的输出方差与“内积的维度”（即 $r$）成正比。$r$起到了归一化的作用，通过除以 $r$，LoRA 实现了一种**尺度不变性（Scale Invariance）**：

- **归一化：** 它抵消了 $r$ 增加带来的矩阵乘积数值增长。
- **超参数迁移：** 这使得 $\alpha$ 变成了一个相对独立的“强度”开关。如果你在 $r=8$ 时找到了一个很好用的学习率和 $\alpha$，当你决定升级到 $r=16$ 以捕获更多特征时，由于有 $1/r$ 的存在，你往往可以直接沿用之前的学习率，而不需要重新调优。
