---
title: 强化学习算法
date: 2026-03-08 13:18:22
tags:
  - Math
  - ML
  - AI
  - LLM
categories:
  - ML
---
# PPO
> OpenAI的GPT-3.5、GPT-4，二者在RLHF阶段均主要采用PPO算法完成偏好对齐，是PPO在LLM领域应用的标杆，也是目前闭源大模型中对齐效果最稳定的案例。

## GAE

PPO基于Actor-Critic基础框架进行优化而来，同样包含`Actor`、`Critic`模型，传统的Actor-Critic会通过TD误差来近似奖励函数：

$$
A \approx \delta_t =  r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

PPO使用**GAE（广义优势估计）**，融合**未来多步 TD 误差**的加权和：

$$
A_t^{GAE_{\lambda}} = \sum_{l=0}^{\inf} (\gamma λ)^l \delta_{t+l}
$$

- $λ=0$：退化为单步 TD 误差（低方差、高偏差）
- $λ=1$：退化为蒙特卡洛估计（低偏差、高方差）

## 裁剪 & 重要性采样（ $L^{CLIP}$ ）

传统的策略梯度 (如 REINFORCE) 用策略 $\pi_θ$ 采样 $\rightarrow$ 计算梯度 $\rightarrow$ 更新 $θ$。而PPO基于训练速度的考量，希望使用离线数据并进行多次学习，针对这种诉求，传统的策略梯度方法就会非常不稳定。如果一次更新让策略 $\pi_θ$ 发生了剧烈变化，可能会导致模型陷入崩溃，无法恢复。
PPO 的核心逻辑是：
- 添加重要性采样，修正新旧两个分布之间的偏差
- 限制策略更新的幅度，给重要性采样打补丁，避免方差爆炸，当新策略 $\pi_θ$ 偏离旧策略 $\pi_{θ_{old}}$ 太远时，直接忽略

我们定义新旧策略的概率比值为 $r_t(θ)$：
$$
r_t(θ) = \frac{\pi_θ(a_t|s_t)}{\pi_{θ_{old}}(a_t|s_t)}
$$

- 如果 $r_t > 1$，说明在当前状态下，新策略选择该动作的概率比旧策略大。
- 如果 $r_t < 1$，则相反。PPO 的目标函数（Clipped Surrogate Objective）如下：

$$
L^{CLIP}(θ) = \hat{\mathbb{E}}_t \left[ \min(r_t(θ) \hat{A}_t, \text{clip}(r_t(θ), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]
$$

| 优势 $\hat{A}_t$ | 动作表现 | 策略变化 $r_t$ | PPO 的态度 |
|------------------|----------|----------------|------------|
| >0 (好) | 优于平均 | 很大 (>1+ϵ) | 压制 (Clip) |
| >0 (好) | 优于平均 | 很小 (<1) | 不限制 |
| <0 (差) | 低于平均 | 很大 (>1) | 猛拉 (Unclipped) |
| <0 (差) | 低于平均 | 很小 (<1−ϵ) | 压制 (Clip) |

## 目标函数（The Total Loss）

PPO的总损失函数是将三个目标合成一个 **Loss 最小化**：

$$
L_t(θ, \phi) = \hat{\mathbb{E}}_t \left[ -L_t^{CLIP}(θ) + c_1 L_t^{VF}(\phi) - c_2 H(s_t) \right]
$$

- **策略损失 (Policy Loss)**：即 $-L_t^{CLIP}(θ)$。通过剪裁机制，确保策略更新平滑。
- **价值损失 (Value Function Loss)**：即 $L_t^{VF}(\phi)$。通常是均方误差（MSE），用来训练 Critic 网络，让 $V_\phi(s)$ 越来越接近实际观测到的奖励回报。

$$
L^{VF} = (V_\phi(s_t) - R_t)^2
$$

- **熵奖励 (Entropy Bonus)**：即 $H$。这是为了鼓励探索。如果策略输出的概率分布太集中（比如只选某一个动作），熵就小。加上这一项能防止模型过早陷入局部最优解。

# DPO (Direct Preference Optimization)
> 它把强化学习问题变成了一个二元交叉熵 (Binary Cross Entropy) 问题，它让模型在看到问题 $x$ 时，最大化“好回答” $y_w$ 出现的概率，同时最小化“烂回答” $y_l$ 出现的概率。

在 RLHF 中，我们的目标是最大化奖励并保留 KL 惩罚：

$$
\max_{\pi} \mathbb{E}_{s \sim D, a \sim \pi} [r(s, a)] - \beta \mathbb{D}_{KL} [\pi(a|s) || \pi_{ref}(a|s)]
$$

数学证明，满足这个目标的最优策略 $\pi_r$ 可以表示为：
$$
\pi_r(a|s) = \frac{1}{Z(s)} \pi_{ref}(a|s) \exp\left(\frac{1}{\beta} r(s, a)\right)
$$

其中 $Z(s)$ 是归一化常数（配分函数）。反过来，我们可以推导出奖励函数 $r(s, a)$ 如何由最优策略表达：
$$
r(s, a) = \beta \log \frac{\pi_r(a|s)}{\pi_{ref}(a|s)} + \beta \log Z(s)
$$

将这个 $r(s, a)$ 代入 **Bradley-Terry 偏好模型**（即人类认为动作 $a_w$ 优于 $a_l$ 的概率 $P(a_w \succ a_l | s) = \sigma(r(s, a_w) - r(s, a_l))$）。常数 $Z(s)$ 被抵消掉了，我们最终得到了：

**DPO 损失函数**

$$
L_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$

- $y_w$：人类选出的好回答 (Winner)。
- $y_l$：人类拒绝的烂回答 (Loser)。
- $\beta$：一个超参数，控制对偏好的敏感度。

**Bradley-Terry 偏好模型**
如果我们有两个竞争者A和B，他们的实力参数分别为 $p(A)$ 和 $P(B)$ ，那么A击败B的概率可以表示为：

$$
P(A beats B) = \frac{p(A)}{p(A) + p(B)}
$$

# GRPO

在标准 PPO 中，为了计算优势函数 $A(s, a)$，我们需要一个和 Actor 同样大的 Critic 网络 来估算 $V(s)$。多维护一个 Critic 模型意味着：
- 显存翻倍：两个庞然大物并存，显存捉襟见肘。
- 训练变慢：需要额外的计算资源来更新 Critic。
- 价值估计难：在推理任务（如奥数、代码）中，状态 $s$ 到奖励 $r$ 的映射非常复杂，Critic 往往很难估准。

GRPO中优势函数不依赖 $V(s)$，而是计算这组分数在当前组内的标准化分值：
$$
A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}
$$

GRPO 损失函数就可以直接删掉critic那一项

$$
L_t(θ, \phi) = \hat{\mathbb{E}}_t \left[ -L_t^{CLIP}(θ) - c_2 H(s_t) \right]
$$

# GDPO
