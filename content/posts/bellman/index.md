---
title: "贝尔曼方程：强化学习的核心递推关系"
date: 2026-03-10
description: "从状态价值函数出发，推导贝尔曼方程，并对比 MC 与 TD 两种求解思路，分析各自的偏差-方差权衡。"
tags:
  - Math
  - ML
  - AI
categories:
  - ML
series:
  - 强化学习基础
draft: false
---

## 基本公式

在强化学习的任务中，Agent 的目标是寻找一个最佳的 $\pi^{*}$ 使得从状态 $s$ 出发的**"期望累计折扣回报"** $V^{\pi}(s)$ 最大化。因为环境满足马尔科夫性质，所以对任意状态 $s$ 均成立：

$$
P(S_{t+1}|S_t, S_{t-1}, ..., S_0) = P(S_{t+1}|S_t)
$$

$V(s)$ 也被称为**状态价值**，它是累计折扣回报 $G_t$ 的期望：

$$
V_{\pi}(s) = \mathbb{E}_{\pi} [G_t | S_t = s]
$$

累计折扣回报以 $\gamma$ 为折扣因子：

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots + \gamma^{n-1} r_{t+n}
$$

$$
G_t = r_{t+1} + \gamma G_{t+1}
$$

将上式代入状态价值函数，得到**贝尔曼方程（Bellman Equation）**：

$$
V_{\pi}(s) = \mathbb{E}_{\pi} [r_{t+1} + \gamma G_{t+1} | S_t = s]
$$

$$
V_{\pi}(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|a,s) \cdot [r + \gamma \mathbb{E}_{\pi}(G_{t+1}|S_{t+1}=s')]
$$

$$
V_{\pi}(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|a,s) \cdot [r + \gamma V_{\pi}(s')]
$$

## Bellman 方程与时间差分学习（TD）

在真实场景中，我们并不知道系统的转移概率 $p(s',r|a,s)$，因此通过采样来近似：

$$
V_{\pi}(s) \approx \sum_a \pi(a|s) \frac{1}{N} \sum_{n=1}^{N} [r_n + \gamma V_{\pi}(s_n')]
$$

我们的目标是求出使 $V_{\pi}(s)$ 最大的策略 $\pi$：

$$
\arg\max_{\pi} V_{\pi}(s)
$$

首先需要计算 $V_{\pi}(s)$，有两种方案：**MC** 和 **TD**。

### MC（蒙特卡洛）

需要完整走完从状态 $s$ 到结束的轨迹，以 $G_t$ 与 $V_{\pi}(s)$ 的均方误差作为损失：

$$
L = \mathbb{E}[(G_t - V(s;\theta))^2]
$$

**缺点：**

1. 只能用于回合制任务（需要完整轨迹）
2. 不同轨迹的 $G_t$ 波动大，方差极大，难以训练
3. 数据效率低，必须等到完整轨迹才能更新参数

### TD（时间差分）

直接基于贝尔曼方程定义损失函数，每走一步 $(s_t, a_t, r_{t+1}, s_{t+1})$ 即可更新：

$$
L = \mathbb{E}\left[\left((r_{t+1} + \gamma V(s_{t+1};\theta^-)) - V(s_t;\theta)\right)^2\right]
$$

其中 $r_{t+1} + \gamma V(s_{t+1};\theta^-)$ 称为 **TD 目标（TD target）**，$\theta^-$ 表示目标网络参数（停止梯度）。两者之差称为 **TD 误差（TD error）**：

$$
\delta_t = r_{t+1} + \gamma V(s_{t+1};\theta^-) - V(s_t;\theta)
$$

参数更新方向为减小该误差：

$$
\theta \leftarrow \theta + \alpha \cdot \delta_t \cdot \nabla_\theta V(s_t;\theta)
$$

**相比 MC，TD 的优势：**

1. **无需完整轨迹**，可用于连续性任务
2. **方差更小**，每步只依赖一个即时奖励 $r_{t+1}$
3. **数据效率高**，每步交互后即可更新

代价是引入了**偏差（bias）**：TD target 中的 $V(s_{t+1};\theta^-)$ 本身是估计值，存在自举（bootstrapping）带来的偏差。MC 与 TD 在偏差-方差上形成了互补的权衡。
